"""Run four frozen arms in separate sequential processes; never resume failures silently."""
import json
import subprocess
import sys
from pathlib import Path

from witrans_tools.common import fingerprint, now, write_json


def main():
    path = Path('runs/v12-factorial-progress.json')
    if path.exists():
        raise ValueError('Preserve existing pipeline; inspect failure before explicit new attempt')
    plan = json.loads(Path('data/prepared/v12-factorial-v2/plan.json').read_text(encoding='utf-8'))
    state = {'at': now(), 'phase': 'training', 'plan_hash': fingerprint(plan), 'completed': [],
             'release_approved': False}
    write_json(path, state)
    try:
        for arm in plan['arms']:
            state.update(current_arm=arm['name'], updated_at=now())
            write_json(path, state)
            subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m', 'scripts.low_cpu_run',
                            '--module', 'scripts.train_v12_factorial', '--arm', arm['name']], check=True)
            state['completed'].append(arm['name'])
            write_json(path, state)
        state.update(phase='training_complete_semantic_evaluation_pending', updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence', updated_at=now(),
                     error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(path, state)


if __name__ == '__main__':
    main()
