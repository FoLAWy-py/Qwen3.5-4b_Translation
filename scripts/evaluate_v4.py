"""Run fixed full200-row development generation, never release testing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('selected', 'start'), required=True)
    parser.add_argument('--selection', default='runs/v4-selection.json')
    parser.add_argument('--input', default='data/prepared/v4/dev.jsonl')
    parser.add_argument('--output')
    args = parser.parse_args()
    destination = Path(args.output or f'runs/v4-{args.role}-dev.jsonl')
    if destination.exists():
        raise ValueError('Generation file exists; preserve prior results')
    adapter = 'models/witrans-4b-v2-selected/adapter' if args.role == 'start' else json.loads(Path(args.selection).read_text(encoding='utf-8'))['selected']['directory']
    subprocess.run([sys.executable, '-u', '-m', 'scripts.low_cpu_run', 'evaluate', '--adapter-dir', adapter,
        '--quantization', 'nf4', '--input', args.input, '--output', str(destination),
        '--max-length', '1024', '--max-new-tokens', '256'], check=True)

if __name__ == '__main__':
    main()
