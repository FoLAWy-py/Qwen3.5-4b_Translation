"""After zero repairs and real parameter updates, change only lr from2e-6 to2e-5."""
import copy
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    destination=Path('data/prepared/v17-strength-trial')
    assert not destination.exists(), 'Preserve experiment'
    previous=load('data/prepared/v16-span-trial/plan.json')
    screening=load('runs/v16-span-recall-screening.json')
    deltas=load('runs/v16-span-weight-update-diagnostics.json')
    assert not screening['recall_first_gate_passed'] and screening['outputs_identical_to_v15']==11
    assert deltas['reports']['v16']['changed_tensors']==504
    plan=copy.deepcopy(previous)
    plan.update(at=now(),output='models/witrans-4b-v17-strength-cpo',learning_rate=2e-5,
        hypothesis='With actual updates but zero TRAIN repairs at2e-6, increasing only lr tenfold at the same16-update budget changes enough decision boundaries to repair at least3/11 errors and clear the critical prohibition reversal.',
        control=dict(plan='data/prepared/v16-span-trial/plan.json',plan_hash=fingerprint(previous),
                     adapter_sha256=screening['adapter_sha256']),
        changed_factor='Only learning_rate:2e-6 to2e-5. Same v7 starting weights,11 pairs,128 replay,64 visit order,chosen span weights3,beta,optimizer,seed42 and61450 full input tokens.',
        diagnosis_hash=fingerprint(screening),gpu_predecessor_job='runs/v16-span-trial-job.json',
        update_diagnostics_hash=fingerprint(deltas),
        scope='One predeclared update-strength trial; same data and budget as failedv16 span trial. No new negatives, no checkpoint search, no original-base evaluation.',
        selection='Predeclared final16; recall-first gate identical to v16. No DEV checkpoint selection.')
    assert plan['full_token_budget']==61450 and plan['updates']==16
    write_json(destination/'plan.json',plan)
    print({'plan_hash':fingerprint(plan),'learning_rate':plan['learning_rate'],'hypothesis':plan['hypothesis']})


if __name__=='__main__':
    main()
