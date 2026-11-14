"""CPU integration with the actual pinned Wan class; no pretrained weights needed."""
import ast
import gc
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from bootstrap import install
from weight_cache import load_transformer_cached


def main():
    upstream=Path(__file__).resolve().parents[3]/'VideoX-Fun'
    install(upstream)
    source=(upstream/'examples/wan2.2_fun/predict_v2v_control.py').read_text(encoding='utf-8-sig')
    imports=ast.Module(body=[n for n in ast.parse(source).body
        if isinstance(n,(ast.Import,ast.ImportFrom))],type_ignores=[])
    exec(compile(imports,'pinned-control-imports','exec'),{})
    from videox_fun.models import Wan2_2Transformer3DModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    torch.manual_seed(43)
    model=Wan2_2Transformer3DModel(
        model_type='t2v',in_dim=4,in_channels=4,dim=64,hidden_size=64,
        ffn_dim=96,freq_dim=16,text_dim=16,out_dim=4,num_heads=4,
        num_layers=1,text_len=8).float().eval()
    # Upstream initializes the output head to zero. Make the output nonzero so
    # a broken quantized backbone cannot pass via an identically zero result.
    torch.nn.init.normal_(model.head.head.weight,std=.02)
    # Random tiny weights check adapter structure, not pretrained model quality.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'source';source.mkdir()
        model.save_config(source)
        save_file(model.state_dict(),str(source/'weights.safetensors'))
        converted=load_transformer_cached(Wan2_2Transformer3DModel,source,
            torch_dtype=torch.float32,cache_dir=root/'cache')
        x=torch.randn(1,4,1,4,4);t=torch.tensor([1.]);context=[torch.randn(4,16)]
        with torch.no_grad():
            expected=model(x,t,context,4)
            actual=converted(x,t,context,4)
        assert actual.shape==expected.shape
        assert torch.isfinite(actual).all()
        assert expected.abs().max().item()>.01
        assert (actual-expected).abs().max().item()<.03
        print('PASS pinned Control imports and tiny CPU forward/cache roundtrip; max error:',
              (actual-expected).abs().max().item())
        del converted
        gc.collect()


if __name__=='__main__':main()
