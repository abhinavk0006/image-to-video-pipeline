"""Opt-in weight-only INT8 and bounded prompt reuse for pinned VideoX-Fun.

Storage reduction only: dequantizes one Linear weight for FP16 F.linear.
Not a fused INT8 kernel; no trained checkpoint is modified on disk.
"""
from collections import OrderedDict
import inspect
import json

def quantize_model(model, enabled=False):
    if not enabled:return model
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Int8Linear(nn.Module):
        def __init__(self,original):
            super().__init__()
            w=original.weight.detach().float()
            scale=w.abs().amax(dim=1,keepdim=True).clamp(min=1e-8)/127
            self.weight=nn.Parameter((w/scale).round().clamp(-127,127).to(torch.int8),requires_grad=False)
            self.weight_scale=nn.Parameter(scale.to(original.weight.dtype),requires_grad=False)
            self.bias=nn.Parameter(original.bias.detach(),requires_grad=False) if original.bias is not None else None
            self.in_features=original.in_features;self.out_features=original.out_features
        def forward(self,x):
            w=self.weight.to(dtype=x.dtype)*self.weight_scale.to(dtype=x.dtype)
            return F.linear(x,w,None if self.bias is None else self.bias.to(dtype=x.dtype))
    # Preserve the control and conditioning projections during the first quality comparisons.
    exclusions=('before_proj','after_proj','patch_embedding','time_embedding','text_embedding','head','modulation','control')
    replacements=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) and not any(s in name for s in exclusions):replacements.append((name,module))
    for name,module in replacements:
        parent_name,_,leaf=name.rpartition('.');parent=model.get_submodule(parent_name) if parent_name else model
        setattr(parent,leaf,Int8Linear(module))
    model._lab_int8_report={'linear_layers':len(replacements),'excluded_name_fragments':exclusions,'fused_int8':False,'quality_validated':False}
    return model

def cache_prompts(pipeline,limit=2):
    original=pipeline.encode_prompt;sig=inspect.signature(original);cache=OrderedDict()
    pipeline._lab_prompt_cache_stats={'hits':0,'misses':0}
    def move(value,device):
        if hasattr(value,'to'):return value.detach().to(device)
        if isinstance(value,list):return [move(v,device) for v in value]
        if isinstance(value,tuple):return tuple(move(v,device) for v in value)
        return value
    def encode(*args,**kwargs):
        bound=sig.bind(*args,**kwargs);bound.apply_defaults();p=bound.arguments
        if p.get('prompt_embeds') is not None or p.get('negative_prompt_embeds') is not None:return original(*args,**kwargs)
        key=json.dumps({k:str(v) for k,v in p.items() if k not in ('device','prompt_embeds','negative_prompt_embeds')},sort_keys=True)
        device=p.get('device') or pipeline._execution_device
        if key in cache:
            pipeline._lab_prompt_cache_stats['hits']+=1
            cache.move_to_end(key);return move(cache[key],device)
        pipeline._lab_prompt_cache_stats['misses']+=1
        value=original(*args,**kwargs);cache[key]=move(value,'cpu')
        while len(cache)>limit:cache.popitem(last=False)
        return value
    pipeline.encode_prompt=encode
    return cache
