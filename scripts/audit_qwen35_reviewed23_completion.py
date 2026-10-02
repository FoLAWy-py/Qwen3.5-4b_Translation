"""Bind actual successful exit and final64 receipt; efficacy is a separate screen."""
import math
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load, verify_training
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    dest=Path('runs/qwen35-v3-reviewed23-sft-training-completion-audit.json')
    assert not dest.exists()
    plan_path='data/prepared/qwen35-v3-reviewed23-sft-repair/plan.json'
    plan=load(plan_path)
    metrics,sha=verify_training(plan)
    owner=load('runs/qwen35-v3-reviewed23-sft-job.json')
    state=load('runs/qwen35-v3-reviewed23-sft-recall-progress.json')
    assert owner['status']=='finished' and owner['exit_code']==0
    assert state['actual_training_exit_code']==0 and state['adapter_sha256']==sha
    assert state['plan_hash']==fingerprint(plan) and state['owner_pid']==62136
    assert not psutil.pid_exists(62136)
    assert metrics['full_tokens']==435971 and metrics['forward_calls']==1728 and metrics['backward_calls']==1472
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip'])
        and 0<r['peak_reserved_gib']<=6.5 for r in metrics['history'])
    notes=read_jsonl('runs/qwen35-v3-reviewed23-sft-recall-manual.jsonl')
    ids=set(plan['recall_sets']['actual_error_ids'])
    error_notes=[n for n in notes if n['id'] in ids]
    assert len(error_notes)==23 and len({n['id'] for n in error_notes})==23
    from collections import Counter
    counts=dict(Counter(n['verdict'] for n in error_notes))
    assert counts=={'major':14,'pass':4,'minor':5}
    recall_processes=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time','exe']):
        if 'python' not in (p.info['name'] or '').lower():
            continue
        command=p.info['cmdline'] or []
        if 'scripts.evaluate_qwen35' in command and 'models/witrans-qwen35-v3-reviewed23-sft' in command:
            recall_processes.append(dict(pid=p.pid,created=p.info['create_time'],exe=p.info['exe'],command=command))
    report=dict(at=now(),plan_path=plan_path,plan_hash=fingerprint(plan),metrics_hash=fingerprint(metrics),
        owner_job_hash=fingerprint(owner),serial_exit_evidence_hash=fingerprint(state),
        original_training_pid_absent=True,actual_training_exit_code=0,adapter_sha256=sha,
        updates=64,full_tokens=435971,forward_calls=1728,backward_calls=1472,
        training_seconds=metrics['seconds'],peak_reserved_gib=metrics['peak_reserved_gib'],cpu_parameter_count=0,
        predeclared_training_round_completed=True,valid_pool_audit_hash=plan['evidence']['focused_negative_audit_hash'],
        actual_error_counts=counts,error_gate_passed=False,
        recall_process_inventory=recall_processes,remaining_individual_readings=80,
        dev_decoding_allowed=False,candidate_semantic_success=False,
        next='Preserve bounded103 TRAIN outputs and individually review80 preservation/context readings. Reject current candidate on frozen error gate; complete A/C diagnosis before any further trial. No DEV/timing/final freeze/independent400 for this failed candidate.',
        scope='Successful predeclared training execution with fully rereviewed pool; this is not semantic optimization success, resource/speed acceptance or stage Goal completion.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(dest,report)
    print(dict(audit=str(dest),adapter_sha256=sha,error_counts=counts,error_gate_passed=False))


if __name__=='__main__':
    main()
