"""Smoke tests for the source adapter and mathematical quantization policy."""
import ast,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_control import rewrite,reviewed_job
from runtime import quantize_model
class Tests(unittest.TestCase):
 def test_actual_pinned_source_contract(self):
  src=(ROOT.parents[2]/'VideoX-Fun/examples/wan2.2_fun/predict_v2v_control.py').read_text(encoding='utf-8-sig')
  overrides={'weight_dtype':'torch.float16','GPU_memory_mode':repr('sequential_cpu_offload'),'video_length':'33'}
  tree=rewrite(ast.parse(src),overrides,True);compile(tree,'control','exec')
  self.assertIn('_load_transformer',ast.unparse(tree))
  with self.assertRaises(ValueError):rewrite(ast.parse(src),{'missing_setting':'1'},False)
 def test_int8_math_error_bound(self):
  rng=np.random.default_rng(43);w=rng.normal(size=(32,64)).astype(np.float32);w[0]=0
  scale=np.maximum(np.max(np.abs(w),axis=1,keepdims=True),1e-8)/127
  q=np.clip(np.round(w/scale),-127,127).astype(np.int8);restored=q.astype(np.float32)*scale
  self.assertTrue(np.all(np.isfinite(restored)))
  self.assertTrue(np.all(np.abs(w-restored)<=scale*.501+1e-7))
  self.assertLess(q.nbytes+scale.nbytes,w.nbytes*.3)
 def test_review_not_implicitly_approved(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'r').write_text('r');(p/'g').write_text('g');(p/'review').write_text(json.dumps({'decision':'approved_motion_only'}))
   with self.assertRaises(ValueError):reviewed_job({'reference':str(p/'r'),'control':str(p/'g'),'control_review':str(p/'review')})
 def test_fallback_preserves_aspect_and_descending_area(self):
  from run_with_fallback import sizes
  self.assertEqual(sizes(768,432,['512x288']),[(768,432),(512,288)])
  with self.assertRaises(ValueError):sizes(768,432,['1024x576'])
  with self.assertRaises(ValueError):sizes(768,432,['512x320'])
 @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is not installed on this laptop runtime')
 def test_torch_layer_roundtrip(self):
  import torch
  net=torch.nn.Sequential(torch.nn.Linear(64,32));x=torch.randn(4,64);original=net(x).detach();quantize_model(net,True)
  self.assertEqual(net[0].weight.dtype,torch.int8)
  self.assertLess((net(x)-original).abs().max().item(),.04)

 @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('accelerate'),'CPU Torch/Accelerate required')
 def test_streaming_cache_roundtrip_and_tamper(self):
  import torch
  from safetensors.torch import save_file
  from weight_cache import load_transformer_cached
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.layer=torch.nn.Linear(16,8);self.head=torch.nn.Linear(8,4)
   @classmethod
   def from_config(cls,config,**kwargs):return cls()
   def forward(self,x):return self.head(self.layer(x))
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source';source.mkdir();(source/'config.json').write_text('{}')
   original=Tiny().half();save_file(original.state_dict(),str(source/'weights.safetensors'))
   x=torch.randn(3,16).half();expected=original(x).detach()
   m=load_transformer_cached(Tiny,source,torch_dtype=torch.float16,cache_dir=root/'cache')
   self.assertEqual(m.layer.weight.dtype,torch.int8)
   self.assertEqual(m.head.weight.dtype,torch.float16)
   self.assertLess((m(x)-expected).abs().max().item(),.04)
   second=load_transformer_cached(Tiny,source,torch_dtype=torch.float16,cache_dir=root/'cache')
   self.assertTrue(torch.equal(m(x),second(x)))
   import gc
   del m,second;gc.collect()
   shard=next((root/'cache').rglob('*.safetensors'));shard.write_bytes(b'corrupt')
   with self.assertRaisesRegex(ValueError,'Changed converted'):
    load_transformer_cached(Tiny,source,torch_dtype=torch.float16,cache_dir=root/'cache')
 @unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
 def test_prompt_cache_is_bounded_and_keys_both_prompts(self):
  import torch
  from runtime import cache_prompts
  class TinyPipeline:
   _execution_device='cpu'
   calls=0
   def encode_prompt(self,prompt,negative_prompt='',device=None,prompt_embeds=None,negative_prompt_embeds=None):
    self.calls+=1;return ([torch.ones(2,3)*len(prompt)],[torch.zeros(2,3)])
  p=TinyPipeline();cache=cache_prompts(p,2)
  p.encode_prompt('a','b');p.encode_prompt('a','b');self.assertEqual(p.calls,1)
  p.encode_prompt('a','c');p.encode_prompt('d','e');self.assertEqual(len(cache),2)
  p.encode_prompt('a','b');self.assertEqual(p.calls,4)
if __name__=='__main__':unittest.main()
