"""Import only the Wan Control family from the pinned multi-model repository.

Avoids importing unrelated image, speech and training families at startup. This
installs lazy package namespaces in this isolated worker process only; source
files in the upstream checkout remain unchanged.
"""
import importlib,importlib.machinery,sys,types
from pathlib import Path

def install(upstream):
 root=Path(upstream).resolve();sys.path.insert(0,str(root));importlib.import_module('videox_fun')
 def package(name,registry,fallback=None):
  mod=types.ModuleType(name);mod.__path__=[str(root.joinpath(*name.split('.')))];mod.__file__=str(Path(mod.__path__[0])/'__init__.py')
  mod.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True);mod.__spec__.submodule_search_locations=mod.__path__
  def get(attr):
   if attr.startswith('__'):raise AttributeError(attr)
   target=registry.get(attr)
   if target is None and fallback:target=(fallback,attr)
   if target is None:raise AttributeError(name+'.'+attr)
   module,symbol=target;value=getattr(importlib.import_module(module),symbol);setattr(mod,attr,value);return value
  mod.__getattr__=get;sys.modules[name]=mod;return mod
 models={}
 for symbol,module in {'Wan2_2Transformer3DModel':'wan_transformer3d','WanT5EncoderModel':'wan_text_encoder','AutoencoderKLWan':'wan_vae','AutoencoderKLWan3_8':'wan_vae3_8'}.items():models[symbol]=('videox_fun.models.'+module,symbol)
 models['AutoTokenizer']=('transformers','AutoTokenizer');package('videox_fun.models',models)
 package('videox_fun.pipeline',{'Wan2_2FunControlPipeline':('videox_fun.pipeline.pipeline_wan2_2_fun_control','Wan2_2FunControlPipeline')})
 fuser=('get_sequence_parallel_rank','get_sequence_parallel_world_size','get_sp_group','get_world_group','init_distributed_environment','initialize_model_parallel','model_parallel_is_initialized','sequence_parallel_all_gather','sequence_parallel_chunk','set_multi_gpus_devices','ulysses_all_to_all','xFuserLongContextAttention')
 dist={n:('videox_fun.dist.fuser',n) for n in fuser};dist['shard_model']=('videox_fun.dist.fsdp','shard_model');dist['usp_attn_forward']=('videox_fun.dist.wan_xfuser','usp_attn_forward');package('videox_fun.dist',dist)
 utils={'cfg_skip':('videox_fun.utils.cfg_optimization','cfg_skip'),'FlowDPMSolverMultistepScheduler':('videox_fun.utils.fm_solvers','FlowDPMSolverMultistepScheduler'),'FlowUniPCMultistepScheduler':('videox_fun.utils.fm_solvers_unipc','FlowUniPCMultistepScheduler'),'apply_gpu_memory_mode':('videox_fun.utils.gpu_memory_mode','apply_gpu_memory_mode')}
 util=package('videox_fun.utils',utils,'videox_fun.utils.utils')
 def unsupported(*args,**kwargs):raise NotImplementedError('Legacy LoRA/camera mode is intentionally disabled in this Control experiment')
 util.merge_lora=unsupported;util.unmerge_lora=unsupported
 data=package('videox_fun.data',{});data.process_pose_file=unsupported
 # The dense Wan path does not use these optional sparse-attention kernels.
 # Keep their failure explicit rather than replacing any attention arithmetic.
 if importlib.util.find_spec('triton') is None:
  sparse=types.ModuleType('videox_fun.models.attention_kernel')
  def sparse_unavailable(*args,**kwargs):
   raise NotImplementedError('Sparse attention requires the upstream Triton kernels; use the dense SDPA profile')
  for name in ('_sparse_linear_attention','get_block_map','mean_pool','sparse_linear_attention_varlen'):
   setattr(sparse,name,sparse_unavailable)
  sparse._sparse_linear_attention=types.SimpleNamespace(apply=sparse_unavailable)
  sys.modules[sparse.__name__]=sparse
 return root
