"""Content-addressed, per-tensor INT8 conversion and mmap-backed CPU loading.

Only safetensors experts with exactly matching model tensors are accepted.
No random or silently skipped control weights. Conversion is an offline cache,
not a claim of GPU numeric equivalence or fused INT8 computation.
"""
import hashlib,json,os
from pathlib import Path
from runtime import quantize_model

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def load_transformer_cached(cls,pretrained_model_path,subfolder=None,transformer_additional_kwargs=None,low_cpu_mem_usage=True,torch_dtype=None,*,cache_dir):
 import torch
 from accelerate import init_empty_weights
 from accelerate.utils import set_module_tensor_to_device
 from safetensors import safe_open
 from safetensors.torch import save_file
 path=Path(pretrained_model_path)
 if subfolder:path/=subfolder
 config=json.loads((path/'config.json').read_text())
 if 'has_image_input' in config:config=cls._convert_from_wan_model_config(config)
 extra=dict(transformer_additional_kwargs or {})
 for key,value in extra.pop('dict_mapping',{}).items():extra[value]=config[key]
 with init_empty_weights():
  model=cls.from_config(config,**extra).to(dtype=torch_dtype or torch.float16)
  model=quantize_model(model,True)
 state=model.state_dict()
 qweights={n+'.weight' for n,m in model.named_modules() if hasattr(m,'weight_scale')}
 sources=sorted(path.glob('*.safetensors'))
 if not sources:raise FileNotFoundError('Streaming INT8 conversion requires safetensors expert files')
 identity={'format_version':1,'checkpoint_hashes':{p.name:digest(p) for p in sources},'config':config,'additional':extra,'quantized_weights':sorted(qweights),'dtype':str(torch_dtype),'tensor_shapes':{n:list(t.shape) for n,t in state.items()}}
 key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
 dest=Path(cache_dir)/key;dest.mkdir(parents=True,exist_ok=True)
 marker=dest/'manifest.json'
 if marker.exists():
  manifest=json.loads(marker.read_text())
  if manifest['identity']!=identity:raise ValueError('Quantization cache identity mismatch')
  for record in manifest['shards']:
   if digest(dest/record['file'])!=record['sha256']:raise ValueError('Changed converted weight shard')
 else:
  records=[];loaded=set()
  for source in sources:
   with safe_open(str(source),framework='pt',device='cpu') as f:
    for name in f.keys():
     if name not in state:raise ValueError('Unexpected checkpoint tensor; no silent control-weight omission: '+name)
     if name in loaded:raise ValueError('Duplicate checkpoint tensor: '+name)
     tensor=f.get_tensor(name)
     if list(tensor.shape)!=list(state[name].shape):raise ValueError('Checkpoint/model shape mismatch: '+name)
     if name in qweights:
      w=tensor.float();scale=w.abs().amax(dim=1,keepdim=True).clamp(min=1e-8)/127
      values={name:(w/scale).round().clamp(-127,127).to(torch.int8),name.rsplit('.',1)[0]+'.weight_scale':scale.to(torch_dtype or torch.float16)}
     else:values={name:tensor.to(dtype=state[name].dtype)}
     filename=f'{len(records):05d}.safetensors';temporary=dest/(filename+'.partial')
     save_file({k:v.contiguous() for k,v in values.items()},str(temporary));os.replace(temporary,dest/filename)
     records.append({'file':filename,'keys':list(values),'sha256':digest(dest/filename)});loaded.update(values)
     del tensor,values
     if name in qweights:del w,scale
  if loaded!=set(state):raise ValueError('Missing checkpoint tensors: '+str(sorted(set(state)-loaded)))
  manifest={'identity':identity,'shards':records,'scientifically_verified':False,'numeric_quality_validated':False}
  temporary=dest/'manifest.json.partial';temporary.write_text(json.dumps(manifest,indent=2));os.replace(temporary,marker)
 for record in manifest['shards']:
  with safe_open(str(dest/record['file']),framework='pt',device='cpu') as f:
   for name in record['keys']:
    set_module_tensor_to_device(model,name,'cpu',value=f.get_tensor(name))
 if any(p.device.type=='meta' for p in model.parameters()):raise RuntimeError('Unmaterialized model parameter')
 model._lab_cache_key=key
 return model.eval()
