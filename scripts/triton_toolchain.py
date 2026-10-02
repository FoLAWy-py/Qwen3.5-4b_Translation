"""Resolve bundled tools from actual uv overlay, rather than sysconfig platlib."""
from pathlib import Path
from witrans_tools.compile_runtime import configure_bundled_toolchain

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
