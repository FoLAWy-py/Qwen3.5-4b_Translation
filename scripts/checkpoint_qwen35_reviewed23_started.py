"""Preserve active engineering trial ownership without claiming semantic success."""
from pathlib import Path
import psutil

from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, write_json


def main():
    dest=Path('runs/qwen35-v3-stage-checkpoint-039.json')
    assert not dest.exists()
    plan_path='data/prepared/qwen35-v3-reviewed23-sft-repair/plan.json'
    plan=load(plan_path)
    job_path='runs/qwen35-v3-reviewed23-sft-job.json'
    job=load(job_path)
    recall_job_path='runs/qwen35-v3-reviewed23-sft-recall-job.json'
    recall_job=load(recall_job_path)
    recall=load('runs/qwen35-v3-reviewed23-sft-recall-progress.json')
    live=load('runs/qwen35-v3-reviewed23-sft-live-audit-001.json')
    process=psutil.Process(62136)
    assert abs(process.create_time()-1790920496.1587167)<.001
    assert 'scripts.train_v15_error_repair' in process.cmdline() and plan_path in process.cmdline()
    assert job['status']=='running' and recall_job['status']=='running'
    assert recall['phase']=='waiting_for_training_exit' and recall['gpu_model_loaded_in_orchestrator'] is False
    assert live['plan_hash']==fingerprint(plan) and live['frozen_budget_prefix_verified']
    write_json(dest,dict(at=now(),previous_checkpoint='runs/qwen35-v3-stage-checkpoint-038.json',
        current_stage='Valid-pool reviewed23 direct SFT executing; semantic success unknown',
        negative_audit='runs/qwen35-v3-focused32-negative-final-audit.json',
        negative_review=dict(reviewed=32,retained_major=23,excluded_minor_or_ambiguous=9,
            original416_current_view=dict(pass_count=328,minor=65,major=23),history_preserved=True),
        plan_path=plan_path,plan_hash=fingerprint(plan),
        preflight='runs/qwen35-v3-reviewed23-sft-preflight.json',
        launch_audit='runs/qwen35-v3-reviewed23-sft-launch-audit.json',
        training=dict(job=job_path,status=job['status'],actual_pid=process.pid,
            created_epoch=process.create_time(),command=process.cmdline(),
            output=plan['output'],updates_planned=64,full_tokens_planned=435971,
            actual_prefix_audit='runs/qwen35-v3-reviewed23-sft-live-audit-001.json',
            prefix_updates=live['completed_updates'],prefix_tokens=live['full_tokens'],
            logged_prefix_peak_reserved_gib=live['reported_peak_reserved_gib']),
        recall=dict(job=recall_job_path,status=recall_job['status'],phase=recall['phase'],
            output='runs/qwen35-v3-reviewed23-sft-recall.jsonl',expected_rows=103,
            serial_owner_pid=recall['owner_pid'],serial_owner_created=recall['owner_created'],
            actual_successful_owner_exit_and_saved_weight_verification_required=True),
        prior_trials_quarantined=True,valid_optimization_round_completed=False,
        quality_success_claimed=False,known_public_candidate_not_run=True,
        next='Continue current actual training and serial recall; never duplicate launch. On finish verify64/token/hash/memory and individually review103 actual outputs. Reject on any frozen TRAIN gate failure. Only then consider known200/public116 and formal own-weight timing; do not collect independent400 before final freeze.',
        scope='First actual update and budget prefix only. Peak logged memory is not completed-run resource acceptance; no loss-based semantic success or algorithm claim.',
        stage_goal_complete=False,release_approved=False,default_promoted=False))
    print(dict(checkpoint=str(dest),training_pid=process.pid,verified_prefix_updates=live['completed_updates']))


if __name__=='__main__':
    main()
