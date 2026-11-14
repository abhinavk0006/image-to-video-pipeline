"""Run reviewed control guides against a pinned upstream A14B Control backend."""
import argparse,ast,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from runtime import quantize_model,cache_prompts
UPSTREAM='991863c8c3fb9da946f21b675e52e45b0b53ed68'
ENTRY='examples/wan2.2_fun/predict_v2v_control.py'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def reviewed_job(job):
    image=Path(job['reference']).resolve();guide=Path(job['control']).resolve();review=Path(job['control_review'])
    r=json.loads(review.read_text())
    if r.get('decision')!='approved_control_input' or r.get('kind') not in ('canny','depth','pose'):raise ValueError('A separately reviewed model control input is required; movement-only approval is insufficient')
    if r.get('control_sha256')!=sha(guide) or r.get('reference_sha256')!=sha(image):raise ValueError('Stale control/reference review')
    if not r.get('reviewer'):raise ValueError('Control reviewer required')
    return str(image),str(guide)

def rewrite(tree,overrides,quantized):
    found=set()
    class Patch(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in overrides:
                    node.value=ast.parse(overrides[target.id],mode='eval').body;found.add(target.id)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained' and isinstance(node.func.value,ast.Name) and node.func.value.id=='Wan2_2Transformer3DModel':
                return ast.copy_location(ast.Call(func=ast.Name(id='_load_transformer',ctx=ast.Load()),args=[node.func.value]+node.args,keywords=node.keywords),node) if quantized else node
            if isinstance(node.func,ast.Name) and node.func.id=='apply_gpu_memory_mode':
                node.keywords += [ast.keyword(arg='strict',value=ast.Constant(True))]
            return node
    result=Patch().visit(tree)
    if found!=set(overrides):raise ValueError('Upstream assignments changed: '+str(set(overrides)-found))
    return ast.fix_missing_locations(result)

def main():
    invocation_started=time.perf_counter()
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--upstream',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--jobs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--profile',choices=['fp16','int8'],default='fp16');ap.add_argument('--frames',type=int,default=33);ap.add_argument('--height',type=int,default=432);ap.add_argument('--width',type=int,default=768);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--weight-cache',type=Path);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.frames<1 or min(a.width,a.height)<16 or (a.frames-1)%4 or a.width%16 or a.height%16:raise ValueError('Use 4n+1 frames and positive dimensions divisible by 16')
    upstream=a.upstream.resolve();model=a.model.resolve();output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    if rev!=UPSTREAM:raise ValueError('Upstream revision differs from the tested source contract')
    jobs=json.loads(a.jobs.read_text(encoding='utf-8-sig'));validated=[reviewed_job(j) for j in jobs]
    if not jobs:raise ValueError('Empty job list')
    if not a.dry_run and (output/'run.json').exists():raise FileExistsError('Use a fresh output directory; previous attempts must remain preserved')
    if a.steps<1:raise ValueError('steps must be positive')
    for name in ('high_noise_model','low_noise_model','google/umt5-xxl','Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth'):
        if not (model/name).exists():raise FileNotFoundError(model/name)
    for expert in ('high_noise_model','low_noise_model'):
        if not (model/expert/'config.json').is_file() or not list((model/expert).glob('*.safetensors')):raise FileNotFoundError('Missing Control expert config/weights: '+expert)
    entry=upstream/ENTRY;source=entry.read_text(encoding="utf-8-sig");tree=ast.parse(source)
    opts={'GPU_memory_mode':repr('sequential_cpu_offload'),'ulysses_degree':'1','ring_degree':'1','fsdp_dit':'False','fsdp_text_encoder':'False','compile_dit':'False','enable_teacache':'False','cfg_skip_ratio':'0','enable_riflex':'False','model_name':repr(str(model)),'sample_size':repr([a.height,a.width]),'video_length':str(a.frames),'fps':'16','weight_dtype':'torch.float16','num_inference_steps':str(a.steps),'lora_path':'None','lora_high_path':'None'}
    patched=rewrite(tree,opts,a.profile=='int8')
    split=next(i for i,n in enumerate(patched.body) if isinstance(n,ast.With) and any(isinstance(v.context_expr,ast.Call) and isinstance(v.context_expr.func,ast.Attribute) and v.context_expr.func.attr=='no_grad' for v in n.items))
    prefix=compile(ast.Module(body=patched.body[:split],type_ignores=[]),str(entry),'exec');generation=compile(ast.Module(body=patched.body[split:],type_ignores=[]),str(entry),'exec')
    manifest={'upstream':rev,'entry_sha256':sha(entry),'profile':a.profile,'frames':a.frames,'fps':16,'size':[a.height,a.width],'steps':a.steps,'guidance_scale':6.0,'quantization_quality_validated':False,'scientifically_verified':False,'jobs':[],'model_path':str(model),'scientific_accuracy':None,'runtime_notes':['Models stay loaded across this independent job batch.','Sequential layer offload; no automatic two-GPU sharding.','INT8 converts one checkpoint tensor at a time into a content-addressed mmap-backed CPU cache; disk and host peaks still require profiling.','No 4-step Lightning, legacy checkpoint reuse, spatial/temporal VAE streaming port, or fused INT8 kernel.']}
    (output/('preflight.json' if a.dry_run else 'run.json')).write_text(json.dumps(manifest,indent=2))
    if a.dry_run:print('PASS: source AST, required files, reviewed identities and configuration; no model loaded');return
    import torch
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU smoke tests are separate from generation')
        if torch.cuda.get_device_properties(0).total_memory<8*2**30:raise RuntimeError('This experimental A14B profile requires a larger GPU than the detected device; use the target Kaggle GPU session')
    except Exception as e:
        manifest.update(error_type=type(e).__name__,error=str(e),status='hardware_preflight_failed',wall_seconds=time.perf_counter()-invocation_started)
        (output/'run.json').write_text(json.dumps(manifest,indent=2));raise
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    from weight_cache import load_transformer_cached
    cache_path=(a.weight_cache or output.parent/'fun_control_weight_cache').resolve()
    ns={'__file__':str(entry),'__name__':'__main__','_load_transformer':lambda cls,*args,**kwargs:load_transformer_cached(cls,*args,cache_dir=cache_path,**kwargs)}
    previous=Path.cwd();os.chdir(upstream);sys.path.insert(0,str(upstream));started=time.perf_counter()
    try:
        from bootstrap import install
        install(upstream)
        exec(prefix,ns);cache_prompts(ns['pipeline'])
        manifest['startup_seconds']=time.perf_counter()-started
        manifest['startup_peak_allocated_bytes']={str(d):torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
        manifest['quantization']=[getattr(m,'_lab_int8_report',None) for m in (ns['pipeline'].transformer,ns['pipeline'].transformer_2)]
        manifest['weight_cache_keys']=[getattr(m,'_lab_cache_key',None) for m in (ns['pipeline'].transformer,ns['pipeline'].transformer_2)]
        for index,(job,(image,guide)) in enumerate(zip(jobs,validated)):
            dest=output/f'job_{index:03d}';dest.mkdir(exist_ok=False)
            ns.update(prompt=job['prompt'],negative_prompt=job.get('negative_prompt',''),start_image=image,end_image=None,ref_image=None,control_video=guide,control_camera_txt=None,save_path=str(dest),video_length=a.frames)
            ns['generator']=torch.Generator(device=ns['device']).manual_seed(int(job.get('seed',43)))
            for dev in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize();t=time.perf_counter();row={'index':index,'reference_sha256':sha(image),'control_sha256':sha(guide),'seed':int(job.get('seed',43)),'prompt':job['prompt'],'status':'running'};manifest['jobs'].append(row)
            try:
                exec(generation,ns);torch.cuda.synchronize();row['status']='generated_pending_review'
                row['videos']=[{'path':str(p),'sha256':sha(p)} for p in dest.glob('*.mp4')]
                if not row['videos']:raise RuntimeError('Generation returned no MP4')
                metrics=Path(__file__).resolve().parents[2]/'research_metrics.py'
                result=subprocess.run([sys.executable,str(metrics),'--videos']+[v['path'] for v in row['videos']]+['--output-dir',str(dest/'research_metrics')],capture_output=True,text=True)
                row['research_metrics_preparation']={'returncode':result.returncode,'scored':False,'stdout':result.stdout[-2000:],'stderr':result.stderr[-2000:]}
                row['prompt_cache']=dict(ns['pipeline']._lab_prompt_cache_stats)
            except Exception as e:
                row.update(status='failed',error_type=type(e).__name__,error=str(e));raise
            finally:
                row['wall_seconds']=time.perf_counter()-t;row['peak_allocated_bytes']={str(d):torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())}
                manifest['wall_seconds']=time.perf_counter()-invocation_started;(output/'run.json').write_text(json.dumps(manifest,indent=2))
    except Exception as e:
        manifest['error_type']=type(e).__name__;manifest['cuda_oom_detected']=isinstance(e,torch.cuda.OutOfMemoryError);manifest['error']=str(e);(output/'run.json').write_text(json.dumps(manifest,indent=2));raise
    finally:os.chdir(previous)
if __name__=='__main__':main()
