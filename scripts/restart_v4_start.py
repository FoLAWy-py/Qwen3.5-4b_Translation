"""Preserve a terminal incomplete comparison before starting a new full run."""
import hashlib
from pathlib import Path
import psutil
from witrans_tools.common import now, read_jsonl, write_json

def main():
    for process in psutil.process_iter(['pid','name','cmdline','cwd']):
        try:
            info = process.info
            if Path(info['cwd'] or '.').resolve() != Path.cwd().resolve():
                continue
            if any(arg in ('scripts.evaluate_v4', 'witrans_tools') for arg in (info['cmdline'] or [])):
                raise ValueError('A project evaluation process is still live; do not restart')
        except (psutil.Error, OSError):
            continue
    path = Path('runs/v4-start-dev.jsonl')
    if path.with_suffix('.summary.json').exists():
        raise ValueError('Complete comparison exists; do not restart')
    rows = read_jsonl(path)
    if len(rows) >= 200:
        raise ValueError('All rows exist; recover metadata before restarting')
    backup = Path('runs/v4-start-dev.interrupted-20261001.jsonl')
    if backup.exists():
        raise ValueError('Prior interruption archive exists')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.rename(backup)
    write_json('runs/v4-start-restart.json', {'at': now(), 'archived': str(backup), 'rows': len(rows), 'sha256': digest,
        'reason': 'Original tool handle missing; no current project evaluation process; no completed summary. Cause unknown.',
        'next': 'Fresh full200-row comparison; interrupted outputs are not merged or used for performance'})
    print({'archived': str(backup), 'rows': len(rows)})

if __name__ == '__main__':
    main()
