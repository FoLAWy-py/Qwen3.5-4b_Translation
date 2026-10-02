"""Reject preserved but invalidated TRAIN trials before loading or accepting weights."""
import json
from pathlib import Path

from .common import fingerprint
from .common import read_jsonl


def require_eligible_trial(plan):
    if plan.get('base', {}).get('model_id') != 'Qwen/Qwen3.5-4B':
        return
    plan_hash = fingerprint(plan)
    evidence = plan.get('evidence', {})
    if evidence.get('focused_negative_audit_path'):
        audit_path = Path(evidence['focused_negative_audit_path'])
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if fingerprint(audit) != evidence['focused_negative_audit_hash']:
            raise ValueError('Frozen focused negative audit changed')
        pairs = read_jsonl(plan['pairs_path'])
        if (fingerprint(pairs) != audit['hashes']['pairs']
                or len(pairs) != audit['retained_major_negatives']
                or set(audit['excluded_ids']) & {r['id'] for r in pairs}):
            raise ValueError('Trial does not use the fully rereviewed true-error pool')
    for path in (Path('runs/qwen35-v3-lexical-negative-correction.json'),
                 Path('runs/qwen35-v3-sweetener-negative-correction.json')):
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding='utf-8'))
        if (plan_hash in report['invalidated_plan_hashes']
                or plan['output'] in report['invalidated_outputs']
                or plan.get('starting_adapter') in report['invalidated_outputs']
                or plan.get('starting_adapter_sha256') == report['quarantined_adapter_sha256']):
            raise ValueError(f'Quarantined trial: {report["reason"]} Evidence: {path}')
