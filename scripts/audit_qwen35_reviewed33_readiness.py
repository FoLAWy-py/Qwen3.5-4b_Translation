"""CPU-only check of frozen new data, quarantine gates and serial process inventory."""
from collections import Counter
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load, verify_training
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.qwen35_trial_eligibility import require_eligible_trial


def main():
    dest=Path('runs/qwen35-v3-reviewed33-readiness.json')
    assert not dest.exists()
    plan=load('data/prepared/qwen35-v3-reviewed33-repair/plan.json')
    require_eligible_trial(plan)
    rejected=[]
    old=load('data/prepared/qwen35-v3-reviewed34-repair/plan.json')
    for name,callback in [('train_entry',require_eligible_trial),('recall_and_dev_entry',verify_training)]:
        try:
            callback(old)
        except ValueError as exc:
            assert 'Quarantined trial' in str(exc)
            rejected.append(dict(entry=name,reason=str(exc)))
        else:
            raise AssertionError('Ineligible trial was accepted')
    preflight=load('runs/qwen35-v3-reviewed33-preflight.json')
    assert preflight['plan_hash']==fingerprint(plan)
    assert preflight['gpu_executed'] is False and preflight['optimizer_updates']==0
    assert preflight['full_tokens']==plan['full_token_budget']==305739
    assert preflight['forward_calls']==plan['budget']['forwards']==1216
    assert preflight['backward_calls']==plan['budget']['backwards']==960
    replay=read_jsonl(plan['replay_path']);pairs=read_jsonl(plan['pairs_path'])
    assert fingerprint(replay)==plan['replay_hash'] and fingerprint(pairs)==plan['pairs_hash']
    exposure=Counter(rid for visit in plan['schedule'] for rid in visit['replay'])
    assert sum(exposure.values())==704 and len(exposure)==456
    assert fingerprint(dict(exposure))==plan['budget']['positive_exposure_hash']
    assert all(exposure[r['id']]>=3 for r in replay if r['id'].startswith('q35-v3-supplement-'))
    assert all(exposure[r['id']]>=1 for r in replay)
    assert replay[:416]==read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    assert 'v10-food-01-en' not in {r['id'] for r in pairs}
    assert 'v4-mining-009-zh-CN' in {r['id'] for r in pairs}
    owner=load(plan['owner_job'])
    assert owner['status']=='finished' and owner['exit_code']==0
    family={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())}
    inventory=[]
    terms=('scripts.train_','scripts.evaluate_qwen35','scripts.probe_qwen35_',
           'scripts.diagnose_qwen35_stage','scripts.decode_qwen35_repair_recall')
    for process in psutil.process_iter(['pid','name','cmdline','create_time']):
        if process.pid in family:
            continue
        command=process.info['cmdline'] or []
        if 'python' in (process.info['name'] or '').lower() and any(term in ' '.join(command) for term in terms):
            inventory.append(dict(pid=process.pid,created=process.info['create_time'],command=command))
    assert not inventory, f'Existing model process: {inventory}'
    assert not psutil.pid_exists(58628) and not psutil.pid_exists(63760)
    write_json(dest,dict(at=now(),passed=True,plan_hash=fingerprint(plan),cpu_preflight_hash=fingerprint(preflight),
        quarantine_rejections=rejected,other_model_processes=inventory,prior_train_and_recall_pids_absent=True,
        source_groups=220,pairs=33,replay=456,pair_visits=256,replay_visits=704,
        full_tokens=305739,forward_calls=1216,backward_calls=960,optimizer_updates=0,gpu_executed=False,
        scope='Pretraining CPU readiness only; no quality or resource acceptance inferred.',stage_goal_complete=False))
    print(dict(readiness=str(dest),passed=True,plan_hash=fingerprint(plan)),flush=True)


if __name__=='__main__':
    main()
