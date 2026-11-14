import json,subprocess
from pathlib import Path
repo=Path.cwd();own=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None))
md('# Wan2.2-Fun A14B Control: motion review and experimental runtime\n\nFresh Kaggle session, Internet and GPUs. This is a separate experimental backend. CPU smoke checks do not establish two-T4 feasibility. Default cells do not download weights or start model inference. Review README limitations before using INT8. No Lightning weights are reused.')
code('''from pathlib import Path
import subprocess,sys,json,hashlib,time
ROOT=Path('/kaggle/working')
BRANCH='feature/fun-control-motion-guides'
PIPELINE=ROOT/'image-to-video-pipeline'
UPSTREAM=ROOT/'VideoX-Fun'
UPSTREAM_SHA='991863c8c3fb9da946f21b675e52e45b0b53ed68'
RUN_MODEL=False
DOWNLOAD_WEIGHTS=False
INSTALL_DEPS=False
FALLBACK_SIZE=None # optional '512x288'; lower resolution requires separate visibility review
PROFILE='int8' # experimental low-memory storage; compare with fp16 on a suitable host; no fused INT8 kernel
MOTION='dropper'
CHUNK=1
REFERENCE=None # exact reviewed photograph path
APPROVE_CONTROL=False # set only after manual control/reference review
REVIEWER=''
PROMPT='A controlled laboratory action matching the reviewed reference and control guide.'
MODEL=ROOT/'Wan2.2-Fun-A14B-Control'
WEIGHT_CACHE=ROOT/'fun_control_weight_cache' # needs room for both converted experts; can be a prepared mounted cache
''')
code('''PIPELINE_SHA="'''+own+'''"
if not PIPELINE.exists():
    subprocess.run(['git','clone','--branch',BRANCH,'https://github.com/abhinavk0006/image-to-video-pipeline.git',str(PIPELINE)],check=True)
subprocess.run(['git','fetch','origin',BRANCH],cwd=PIPELINE,check=True)
subprocess.run(['git','checkout','--detach',PIPELINE_SHA],cwd=PIPELINE,check=True)
if not UPSTREAM.exists():
    subprocess.run(['git','clone','https://github.com/aigc-apps/VideoX-Fun.git',str(UPSTREAM)],check=True)
subprocess.run(['git','checkout','--detach',UPSTREAM_SHA],cwd=UPSTREAM,check=True)
print('Pipeline source:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=PIPELINE,text=True).strip())
print('Upstream source:',UPSTREAM_SHA)
if INSTALL_DEPS:
    subprocess.run([sys.executable,'-m','pip','install','-r',str(PIPELINE/'experimental/fun_control/requirements-control.txt')],check=True)
    subprocess.run([sys.executable,'-m','pip','install','Pillow','imageio-ffmpeg','numpy','huggingface_hub'],check=True)
if DOWNLOAD_WEIGHTS:
    from huggingface_hub import snapshot_download
    snapshot_download('alibaba-pai/Wan2.2-Fun-A14B-Control',local_dir=str(MODEL))
''')
md('## Review before generation\nSelect any of the 24 movements. The local HTML viewer also supports other camera angles and review notes. These are generic motion prototypes, not completed science videos.')
code('''from IPython.display import Video,display
library=PIPELINE/'research/motion_review'
print('Available movements:',list(json.loads((library/'motion_templates.json').read_text())['motions']))
display(Video(str(library/'previews'/f'{MOTION}.mp4'),embed=True,width=768))
GUIDE=library/'guides'/f'{MOTION}_chunk_{CHUNK:02d}.mp4'
display(Video(str(GUIDE),embed=True,width=768))
if REFERENCE:
    from PIL import Image
    import imageio.v3 as iio
    reference=Image.open(REFERENCE).convert('RGB').resize((768,432))
    guide=Image.fromarray(iio.imread(GUIDE,index=0)).convert('RGB').resize((768,432))
    display(Image.blend(reference,guide,0.4))
    print('Inspect this overlay: generic motion geometry is not automatically fitted to the photograph.')
print('Guide interval:',2*(CHUNK-1),'to',2*CHUNK,'seconds; each guide has 33 native frames.')
''')
code('''import os
print('Host RAM availability:',os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/2**30,'GiB')
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,str(PIPELINE/'experimental/fun_control/test_control.py')],check=True)
subprocess.run([sys.executable,str(PIPELINE/'experimental/fun_control/test_upstream.py')],check=True)
''')
md('## Explicit control review and generation\nApproving motion geometry does not establish that these synthetic outlines are suitable conditioning. Check the reference alignment, apparatus, interval and format before enabling this cell. Default remains off. This runner does not auto-chain independent job references.')
code('''if RUN_MODEL:
    if not APPROVE_CONTROL or not REVIEWER.strip() or not REFERENCE:
        raise RuntimeError('Review the control input and photograph and provide reviewer/reference first.')
    ref=Path(REFERENCE).resolve()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out=ROOT/('fun_control_'+PROFILE+'_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    review=out/'control_review.json'
    review.write_text(json.dumps(dict(decision='approved_control_input',kind='canny',reviewer=REVIEWER,control_sha256=sha(GUIDE),reference_sha256=sha(ref),notes='Synthetic outline candidate inspected against the reference.')))
    jobs=out/'jobs.json';jobs.write_text(json.dumps([dict(reference=str(ref),control=str(GUIDE),control_review=str(review),prompt=PROMPT,seed=43)]))
    command=[sys.executable,str(PIPELINE/'experimental/fun_control/run_with_fallback.py'),'--upstream',str(UPSTREAM),'--model',str(MODEL),'--jobs',str(jobs),'--output',str(out),'--profile',PROFILE,'--weight-cache',str(WEIGHT_CACHE)]
    if FALLBACK_SIZE:command+=['--fallback-size',FALLBACK_SIZE]
    subprocess.run(command+['--dry-run'],check=True)
    subprocess.run(command,check=True)
    print('All-attempt telemetry:',out/'suite.json')
    for video in out.rglob('*.mp4'):display(Video(str(video),embed=True,width=768))
else:
    print('Review-only mode. No model is downloaded or loaded.')
''')
for i,c in enumerate(cells):
 if c['cell_type']=='code':compile(c['source'],f'cell{i}','exec')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},cells=cells)
Path('Wan22_Fun_Control_A14B_Motion_Test.ipynb').write_text(json.dumps(nb,indent=1))
print('Built notebook; all cells compile')
