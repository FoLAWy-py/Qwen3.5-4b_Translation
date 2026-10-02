"""One owned GPU job: train fixed16, decode eleven TRAIN repairs, then await review."""
import hashlib
import argparse
import re
import json
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',default='v16-span')
    parser.add_argument('--plan',default='data/prepared/v16-span-trial/plan.json')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',args.prefix)
    prefix = args.prefix
    state_path = Path(f'runs/{prefix}-trial-progress.json')
    assert not state_path.exists(), 'Preserve attempt'
    for path in ('runs/v15-error-optimization-job.json','runs/v15-repair-recall-job.json','runs/v15-repair-signal-job.json'):
        owner = load(path)
        assert owner['status'] == 'finished' and owner['exit_code'] == 0
    plan_path = args.plan
    plan = load(plan_path)
    if plan.get('gpu_predecessor_job'):
        owner = load(plan['gpu_predecessor_job'])
        assert owner['status']=='finished' and owner['exit_code']==0
    state = dict(at=now(),phase='training',plan_hash=fingerprint(plan),release_approved=False)
    write_json(state_path,state)
    try:
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
                        'scripts.train_v15_error_repair','--plan',plan_path],check=True)
        metrics = load(Path(plan['output'])/'metrics.json')
        sha = hashlib.sha256((Path(plan['output'])/'adapter_model.safetensors').read_bytes()).hexdigest()
        assert metrics['plan_hash'] == fingerprint(plan) and metrics['adapter_sha256'] == sha
        assert metrics['selected_checkpoint'] == plan['selected_checkpoint'] == plan['updates']
        assert metrics['full_tokens'] == plan['full_token_budget']
        assert metrics['forward_calls'] == sum(2 + len(s['replay']) for s in plan['schedule'])
        assert metrics['backward_calls'] == sum(1 + len(s['replay']) for s in plan['schedule'])
        assert metrics['peak_reserved_gib'] <= 6.5
        write_json(f'runs/{prefix}-fixed-selection.json',dict(at=now(),training_plan_hash=fingerprint(plan),
            selected=dict(directory=plan['output'],step=plan['selected_checkpoint'],adapter_sha256=sha),selection=plan['selection'],release_approved=False))
        state.update(phase='train_recall_decoding',updated_at=now())
        write_json(state_path,state)
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                        '--adapter-dir',plan['output'],'--input',plan['pairs_path'],
                        '--output',f'runs/{prefix}-recall.jsonl','--quantization','nf4',
                        '--max-length','1024','--max-new-tokens','256'],check=True)
        summary = load(f'runs/{prefix}-recall.summary.json')
        assert summary['adapter_sha256'] == sha and summary['data_hash'] == plan['pairs_hash']
        assert summary['prompt_hash'] == plan['prompt_hash'] and summary['base'] == plan['base']
        state.update(phase='individual_train_recall_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error=str(exc),updated_at=now())
        raise
    finally:
        write_json(state_path,state)


if __name__ == '__main__':
    main()
