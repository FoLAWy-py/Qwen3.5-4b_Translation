"""Process-local CPU isolation for diagnosing native Windows startup failures."""
import faulthandler
import os
import runpy
import sys

import psutil


def configure_cpu():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    faulthandler.enable(all_threads=True)
    process = psutil.Process()
    available = process.cpu_affinity()
    selected = available[-4:]
    process.cpu_affinity(selected)
    print(f"进程内 CPU 隔离: {selected}，CPU 线程数 2；不修改系统设置", flush=True)


def main():
    configure_cpu()
    if len(sys.argv) > 2 and sys.argv[1] == "--module":
        module = sys.argv[2]
        sys.argv = [module, *sys.argv[3:]]
    else:
        module = "witrans_tools"
        sys.argv = [module, *sys.argv[1:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
