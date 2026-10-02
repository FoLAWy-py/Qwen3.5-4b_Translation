"""Explicit process-local Windows compiler configuration; no global settings."""
import importlib.util
import os
from pathlib import Path
from functools import wraps


def configure_bundled_toolchain():
    spec=importlib.util.find_spec('triton')
    if spec is None or not spec.origin:
        raise RuntimeError('Compiled runtime requires uv sync --extra compile')
    package=Path(spec.origin).resolve().parent
    compiler=package/'runtime/tcc/tcc.exe'
    cuda=package/'backends/nvidia'
    required=[compiler,cuda/'bin/ptxas.exe',cuda/'include/cuda.h',cuda/'lib/x64/cuda.lib']
    if not all(p.is_file() for p in required):raise RuntimeError('Incomplete bundled Triton toolchain')
    os.environ['CC']=str(compiler);os.environ['CUDA_PATH']=str(cuda)
    return dict(package_directory=str(package),c_compiler=str(compiler),cuda_directory=str(cuda),
        scope='Process-local paths; OS environment unchanged')


def enable_compiled_runtime(translator,cache_dir,*,decode_only=False):
    import torch
    import torch._inductor.config
    toolchain=configure_bundled_toolchain()
    root=Path(cache_dir).resolve()
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor')
    os.environ['TRITON_CACHE_DIR']=str(root/'triton')
    torch._inductor.config.emulate_precision_casts=True
    torch._inductor.config.emulate_divison_rounding=True
    factory=translator._generation_config
    translator._generation_config=lambda **kw:factory(**kw,cache_implementation='static',disable_compile=True,
        **({'max_cache_len':1024} if decode_only else {}))
    base=translator.model.get_base_model() if hasattr(translator.model,'get_base_model') else translator.model
    eager=base.forward
    compiled=torch.compile(eager,backend='inductor',mode='default',fullgraph=False,dynamic=False if decode_only else None)
    if decode_only:
        @wraps(eager)
        def forward(*args,**kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            if ids is not None and ids.shape[-1]==1 and kwargs.get('past_key_values') is not None:
                return compiled(*args,**kwargs)
            return eager(*args,**kwargs)
        base.forward=forward
    else:
        base.forward=compiled
    return dict(toolchain=toolchain,backend='inductor',mode='default',fullgraph=False,dynamic=False if decode_only else None,
        scope='decode_only' if decode_only else 'prefill_and_decode',
        max_cache_len=1024 if decode_only else None,
        forward_signature_preserved=True,
        cache_implementation='static',disable_hf_automatic_compile=True,
        emulate_precision_casts=True,emulate_divison_rounding=True,cache_dir=str(root),
        warmup='Lazy first generation plus whole input round; compilation cost must be reported',quality_validated=False)
