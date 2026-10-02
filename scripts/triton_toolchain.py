"""Resolve bundled tools from actual uv overlay, rather than sysconfig platlib."""
import importlib.util
import os
from pathlib import Path

def configure_bundled_toolchain():
    spec = importlib.util.find_spec('triton')
    if spec is None or not spec.origin:
        raise ValueError('Triton package unavailable')
    package = Path(spec.origin).resolve().parent
    compiler = package / 'runtime/tcc/tcc.exe'
    cuda = package / 'backends/nvidia'
    required = [compiler, cuda/'bin/ptxas.exe', cuda/'include/cuda.h', cuda/'lib/x64/cuda.lib']
    if not all(path.is_file() for path in required):
        raise ValueError('Incomplete bundled Triton toolchain')
    os.environ['CC'] = str(compiler)
    os.environ['CUDA_PATH'] = str(cuda)
    return {'package_directory':str(package), 'c_compiler':str(compiler), 'cuda_directory':str(cuda),
        'scope':'Process-local compiler paths; OS environment and project dependencies unchanged.'}

def main():
    configured = configure_bundled_toolchain()
    from triton.windows_utils import find_cuda
    from triton.runtime.build import get_cc
    from witrans_tools.common import now, write_json
    compiler = get_cc()
    cuda_bin, include, libraries = find_cuda()
    if Path(compiler).resolve() != Path(configured['c_compiler']).resolve() or not cuda_bin or not include or not libraries:
        raise ValueError('Bundled tools still not resolved')
    report = {'at':now(), 'configured':configured, 'detected_c_compiler':compiler,
        'detected_cuda_bin':cuda_bin,'detected_cuda_include':include,'detected_cuda_libraries':libraries,
        'gpu_compilation_executed':False}
    write_json('runs/compile-bundled-toolchain-probe.json',report)
    print(report)

if __name__ == '__main__':
    main()
