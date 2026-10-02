"""Redirect to the best adapter and reviewed mixed pool; no original-base arms."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, write_json


def main():
    destination = Path('data/prepared/v13-public-optimization/plan.json')
    if destination.exists():
        raise ValueError('Preserve frozen optimization plan')
    previous = json.loads(Path('data/prepared/v12-factorial-v2/plan.json').read_text(encoding='utf-8'))
    arm = {'name':'v13-public-sft','start':'v7','dataset':'mixed',
           'output':'models/witrans-4b-v13-public-sft',
           'selection_output':'runs/v13-public-sft-selection.json'}
    plan = {**previous,'at':now(),'arms':[arm],'datasets':{'mixed':previous['datasets']['mixed']},
            'learning_rate':5e-6,
            'method':'Direct continued SFT from best v7 on reviewed public/old mixed pool',
            'supersedes':'User2026-10-01 revoked further original-base comparisons and requested full focus on optimization',
            'limits':'One optimization trial; LR and data change together, not a causal algorithm ablation. Actual semantic gates versus v7 mandatory; no new base generation.'}
    write_json(destination,plan)
    write_json('runs/optimization-scope-20261001.json',{
        'at':now(),'user_instruction':'不再做意义小的与base比较，全力优化模型',
        'goal_status':'active','base_comparison_enabled':False,'four_arm_factorial_cancelled':True,
        'active_plan':str(destination),'active_plan_hash':fingerprint(plan),
        'starting_adapter':'models/witrans-4b-v7-critical-cpo',
        'necessary_comparison':'Best current v7 versus optimization candidate only; retain useful method controls for future optimization',
        'acceptance':'Keep semantic+format+direction+EOS+constraints+memory+latency gates; final release comparison versus current best adapter. No further base comparison required after this user scope change.',
        'next':'Train public mixed SFT, generate candidate and needed v7 public control, individually accept; then source-conditioned fact optimization if justified',
        'override':'Latest explicit user instruction supersedes base-related clauses of earlier active goal; do not resume stopped v12 factorial or base-evaluation jobs',
        'release_approved':False})
    print({'plan':str(destination),'arm':arm,'lr':plan['learning_rate'],'input_tokens':plan['training_input_tokens_per_arm']},flush=True)


if __name__ == '__main__':
    main()
