"""One method control after stronger CPO shows partial repair: remove preference term."""
import copy
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    destination=Path('data/prepared/v18-sft-control')
    assert not destination.exists(), 'Preserve method control'
    previous=load('data/prepared/v17-strength-trial/plan.json')
    screening=load('runs/v17-strength-recall-screening.json')
    deltas=load('runs/v17-strength-weight-update-diagnostics.json')
    assert not screening['recall_first_gate_passed'] and screening['counts']=={'major':7,'minor':2,'pass':2}
    assert deltas['reports']['v17-strength']['changed_tensors']==504
    assert deltas['reports']['v17-strength']['relative_delta_l2']>0
    plan=copy.deepcopy(previous)
    plan.update(at=now(),output='models/witrans-4b-v18-error-sft',repair_objective='sft',
        method='direct_chosen_critical_span_sft_with_same_replay',
        hypothesis='At the same stronger lr2e-5,16updates and reviewed data, direct chosen critical-span SFT without the preference gradient yields at least3 full TRAIN passes and zero critical; CPO changed paraphrases yet left7 major and2 minor.',
        control=dict(plan='data/prepared/v17-strength-trial/plan.json',plan_hash=fingerprint(previous),
                     adapter_sha256=screening['adapter_sha256']),
        changed_factor='Only repair objective: weighted chosen NLL instead of weighted chosen NLL plus CPO preference term. Samev7,11pairs,128replay,64visit order,lr2e-5,span3,seed42 and16 updates.',
        budget_scope='61450 full prompt+answer+EOS forward tokens,256 forwards,192 backward invocations,16 updates remain fixed. Rejected forwards are score diagnostics under no_grad for SFT. Negative backward graph/FLOPs differ, so do not claim identical FLOPs or algorithm superiority.',
        diagnosis_hash=fingerprint(screening),update_diagnostics_hash=fingerprint(deltas),
        gpu_predecessor_job='runs/v17-strength-trial-job.json',
        scope='Useful same-data/update/token-budget SFT/CPO method control; no further LR grid, no new negatives, one seed. Only concrete candidate engineering results may be claimed.',
        selection='Predeclared final16; same recall-first gate as prior trials. No DEV checkpoint selection.')
    assert plan['full_token_budget']==61450 and plan['updates']==16 and plan['learning_rate']==2e-5
    write_json(destination/'plan.json',plan)
    print({'plan_hash':fingerprint(plan),'hypothesis':plan['hypothesis'],'budget_scope':plan['budget_scope']})


if __name__=='__main__':
    main()
