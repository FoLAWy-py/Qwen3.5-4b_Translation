"""Tiny GPU compile test in isolated uv environment; not a model speed claim."""
import argparse
import importlib.metadata
import time
from pathlib import Path
from witrans_tools.common import now, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/compile-smoke.json')
    parser.add_argument('--bundle-toolchain', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Preserve existing compiler evidence')
    report = {'at':now(), 'status':'failed', 'scope':'Toy CUDA function only, not NF4/PEFT model compilation or inference speed.',
        'packages':{name:importlib.metadata.version(name) for name in ('torch','triton-windows')}}
    start = time.perf_counter()
    try:
        if args.bundle_toolchain:
            from scripts.triton_toolchain import configure_bundled_toolchain
            report['toolchain'] = configure_bundled_toolchain()
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        values = torch.linspace(-2,2,2048,device='cuda',dtype=torch.float32)
        def function(x):
            return torch.sin(x)+x*x
        reference = function(values)
        torch.cuda.synchronize()
        compiled = torch.compile(function, fullgraph=True, mode='reduce-overhead')
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled(values).clone()
        torch.cuda.synchronize()
        first_seconds = time.perf_counter()-start
        torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-5)
        torch.compiler.cudagraph_mark_step_begin()
        repeated = compiled(values).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(repeated,reference,rtol=1e-5,atol=1e-5)
        report.update(status='success',first_compile_and_runtime_seconds=first_seconds,
            maximum_abs_error=float((actual-reference).abs().max()),
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:6000])
    report['elapsed_seconds'] = time.perf_counter()-start
    write_json(args.output,report)
    print(report,flush=True)
    if report['status']!='success':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
