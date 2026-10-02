"""Read compiler prerequisites without importing GPU runtimes or changing env."""
import importlib.metadata
import argparse
import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path
from witrans_tools.common import now, write_json

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/compile-runtime-probe.json')
    args = parser.parse_args()
    include = Path(sysconfig.get_path('include'))
    root = Path(sys.base_prefix)
    report = {'at':now(), 'python':sys.version.split()[0], 'packages':{name:version(name) for name in ('torch','transformers','peft','bitsandbytes','triton-windows','triton')},
        'triton_import_available':importlib.util.find_spec('triton') is not None,
        'python_header_available':(include/'Python.h').exists(), 'python_import_library_available':(root/'libs'/'python312.lib').exists(),
        'cl_on_path':shutil.which('cl.exe') is not None, 'nvcc_on_path':shutil.which('nvcc.exe') is not None,
        'recommended_isolated_triton_series':'3.6 for installed PyTorch2.10; do not take latest3.8',
        'scope':'Read-only prerequisites. Missing PATH compilers alone do not prove GPU compilation impossible; Windows Triton wheels may bundle toolchains. No compilation or speed claim.',
        'sources':['https://github.com/triton-lang/triton-windows', 'https://huggingface.co/docs/transformers/v4.57.1/llm_optims']}
    write_json(args.output, report)
    print(report)

if __name__ == '__main__':
    main()
