"""Execute one immutable span-weight ablation after v6 comparison is complete."""
import json
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import fingerprint, read_jsonl
from witrans_tools.critical_spans import validated_annotations

def main():
    plan = json.loads(Path('runs/v7-trial-plan.json').read_text(encoding='utf-8'))
    comparison = json.loads(Path('runs/v6-model-acceptance.json').read_text(encoding='utf-8'))
    if not comparison['comparison_complete']:
        raise ValueError('v6 comparison is incomplete')
    rows = read_jsonl(Path(plan['data_dir']) / 'train.jsonl')
    if fingerprint(rows) != plan['train_hash']:
        raise ValueError('Frozen train pool changed')
    report = json.loads(Path(plan['annotations']).read_text(encoding='utf-8'))
    validated_annotations(rows, report)
    if report['annotation_hash'] != plan['annotation_hash']:
        raise ValueError('Frozen annotations changed')
    if (plan['method'] != 'critical_span_cpo' or plan['learning_rate'] != 1e-5
            or plan['beta'] != .1 or plan['seed'] != 42 or plan['quantization'] != 'nf4'):
        raise ValueError('Unsupported frozen configuration')
    subprocess.run([sys.executable, '-u', '-m', 'scripts.low_cpu_run', '--module','scripts.train_preference',
        '--mode','cpo', '--output',plan['output'], '--data-dir',plan['data_dir'],
        '--starting-adapter',plan['starting_adapter'], '--steps',str(plan['steps']),
        '--accumulation',str(plan['accumulation']), '--critical-spans',plan['annotations']], check=True)

if __name__ == '__main__':
    main()
