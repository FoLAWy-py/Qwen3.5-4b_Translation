"""Execute a role from the frozen development comparison."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import fingerprint, read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('cpo','sft_control','critical_cpo','sft'), required=True)
    parser.add_argument('--plan', default='runs/v6-evaluation-plan.json')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    candidate = plan['candidates'][args.role]
    if fingerprint(read_jsonl(plan['input'])) != plan['development_hash']:
        raise ValueError('Frozen development input changed')
    directory = Path(candidate['directory'])
    if hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest() != candidate['adapter_sha256']:
        raise ValueError('Frozen candidate weights changed')
    if Path(plan['outputs'][args.role]).exists():
        raise ValueError('Preserve existing generation evidence')
    subprocess.run([sys.executable, '-u', '-m', 'scripts.low_cpu_run', 'evaluate',
        '--adapter-dir',str(directory), '--quantization','nf4', '--input',plan['input'],
        '--output',plan['outputs'][args.role], '--max-length','1024', '--max-new-tokens','256'], check=True)

if __name__ == '__main__':
    main()
