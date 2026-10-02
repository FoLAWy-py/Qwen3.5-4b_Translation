"""Inspect only the two task-owned experiment helper launch commands."""
import psutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminate', nargs='*', type=int, default=[])
    requested = set(parser.parse_args().terminate)
    for process in psutil.process_iter(['pid','name','cmdline','cwd']):
        try:
            args = process.info['cmdline'] or []
            if process.info['name'].lower() not in ('uv.exe','cmd.exe'):
                continue
            joined = ' '.join(args)
            if not any(name in joined for name in ('scripts.review_v5_start','scripts.finalize_v5')):
                continue
            print({'pid':process.pid, 'name':process.info['name'], 'parent':process.ppid(), 'status':process.status(), 'affinity':process.cpu_affinity(), 'cwd':process.info['cwd']})
            if process.pid in requested:
                if process.info['name'].lower() != 'cmd.exe' or '/affinity' not in joined.lower() or Path(process.info['cwd']).resolve() != Path.cwd().resolve():
                    raise ValueError('Not the verified task-owned experimental launch')
                for child in reversed(process.children(recursive=True)):
                    child.terminate()
                process.terminate()
                print({'terminated_owned_helper':process.pid})
        except psutil.Error:
            continue

if __name__ == '__main__':
    main()
