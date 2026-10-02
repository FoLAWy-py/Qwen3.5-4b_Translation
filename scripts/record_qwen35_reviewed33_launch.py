"""Record actual training interpreter identity without loading GPU libraries."""
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, write_json


def main():
    dest=Path('runs/qwen35-v3-reviewed33-launch-identity.json')
    assert not dest.exists()
    owner=load('runs/qwen35-v3-reviewed33-repair-job.json')
    assert owner['status']=='running'
    children=psutil.Process(owner['process_pid']).children(recursive=True)
    matches=[p for p in children if 'python' in p.name().lower()
             and 'scripts.train_v15_error_repair' in p.cmdline()
             and 'data/prepared/qwen35-v3-reviewed33-repair/plan.json' in p.cmdline()
             and '.venv-qwen35' not in p.exe()]
    assert len(matches)==1
    process=matches[0]
    identity=dict(at=now(),pid=process.pid,created=process.create_time(),command=process.cmdline(),
                  interpreter=process.exe(),owner_job_hash=fingerprint(owner),gpu_model_loaded_in_observer=False)
    write_json(dest,identity)
    checkpoint=Path('runs/qwen35-v3-stage-checkpoint-032.json')
    assert not checkpoint.exists()
    plan=load('data/prepared/qwen35-v3-reviewed33-repair/plan.json')
    write_json(checkpoint,dict(at=now(),previous_checkpoint='runs/qwen35-v3-stage-checkpoint-031.json',
        goal_status='active',correction_report='runs/qwen35-v3-lexical-negative-correction.json',
        correction_report_hash=fingerprint(load('runs/qwen35-v3-lexical-negative-correction.json')),
        quarantined_prior_adapter='3e95cd30ad4fa13b0db35102d86f9f0b8bf01630b1da5ba2f0fec9346166df61',
        prior_candidate_revised_counts=dict(pass_count=78,minor=24,major=12),prior_screen_still_failed=True,
        original_train_revised_counts=dict(pass_count=328,minor=55,major=33),
        new_trial=dict(plan='data/prepared/qwen35-v3-reviewed33-repair/plan.json',plan_hash=fingerprint(plan),
            status='training',identity=identity,updates=64,pairs=33,replay=456,pair_visits=256,replay_visits=704,
            full_tokens=305739,forwards=1216,backwards=960,recall_rows=113,quality_pending=True),
        next_steps=['Wait on verified real training PID and actual exit0; no second GPU model.',
            'Verify64/final SHA/frozen full-token budget, then serially generate113 TRAIN recall outputs.',
            'Individually read all actual recall translations; same retention/context/direction gates. No automatic DEV on failure.',
            'Valid round completion/DEV/ownweight speed/final freeze/new400 confirmation all remain pending.'],
        valid_optimization_round_requirement_satisfied=False,stage_goal_complete=False,
        release_approved=False,default_promoted=False))
    print(identity,flush=True)


if __name__=='__main__':
    main()
