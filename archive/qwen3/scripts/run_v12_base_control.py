"""Run the original-base control before any further GPU training."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from witrans_tools.common import base_manifest, fingerprint, now, read_jsonl, write_json


def main():
    previous = json.loads(Path('runs/v11-base-dev-plan.json').read_text(encoding='utf-8'))
    plan_path = Path('runs/v12-base-dev-review-plan.json')
    state_path = Path('runs/v12-base-progress.json')
    output = 'runs/v12-base-dev.jsonl'
    if any(p.exists() for p in (plan_path, state_path, Path(output))):
        raise ValueError('Preserve existing v12 evidence; use a fresh run for retries')
    if fingerprint(read_jsonl(previous['input'])) != previous['development_hash']:
        raise ValueError('Frozen DEV changed')
    if base_manifest('models/Qwen3-4B')['revision'] != previous['base']['revision']:
        raise ValueError('Base revision changed')
    v7 = Path('models/witrans-4b-v7-critical-cpo/adapter_model.safetensors')
    if hashlib.sha256(v7.read_bytes()).hexdigest() != '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424':
        raise ValueError('v7 comparison weights changed')
    plan = {**previous, 'at': now(), 'output': output,
            'outputs': {'baseline': output}, 'additional_review_cache_stems': ['v8-selected'],
            'comparison': 'runs/v7-critical-dev.jsonl',
            'scope': 'Known200-row DEV screening only; original base versus existing v7. No release approval.'}
    write_json(plan_path, plan)
    state = {'at': now(), 'phase': 'generating_original_base', 'release_approved': False,
             'plan': str(plan_path), 'plan_hash': fingerprint(plan)}
    write_json(state_path, state)
    try:
        subprocess.run([sys.executable, '-u', '-m', 'scripts.low_cpu_run', 'evaluate',
                        '--baseline', '--quantization', 'nf4', '--input', plan['input'],
                        '--output', output, '--max-length', '1024', '--max-new-tokens', '256'], check=True)
        subprocess.run([sys.executable, '-u', '-m', 'scripts.review_v6', '--role', 'baseline',
                        '--plan', str(plan_path)], check=True)
        state.update(phase='awaiting_individual_semantic_review', updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence', error_type=type(exc).__name__,
                     error=str(exc), updated_at=now())
        raise
    finally:
        write_json(state_path, state)


if __name__ == '__main__':
    main()
