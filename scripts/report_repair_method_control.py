"""Report the useful v17/v18 control without an algorithm superiority claim."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    path = Path('runs/v17-v18-repair-method-control.json')
    assert not path.exists(), 'Preserve control evidence'
    cpo = load('data/prepared/v17-strength-trial/plan.json')
    sft = load('data/prepared/v18-sft-control/plan.json')
    equal_fields = ('starting_adapter_sha256','base','prompt_hash','pairs_hash','replay_hash',
        'schedule','seed','updates','accumulation','learning_rate','pair_weight','replay_weight',
        'critical_annotations_hash','full_token_budget','quantization')
    assert all(cpo[k] == sft[k] for k in equal_fields)
    reports = {}
    for prefix,plan in (('v17-strength',cpo),('v18-sft',sft)):
        screening = load(f'runs/{prefix}-recall-screening.json')
        rows,notes = read_jsonl(f'runs/{prefix}-recall.jsonl'),read_jsonl(f'runs/{prefix}-recall-manual.jsonl')
        assert fingerprint(rows) == screening['generation_hash']
        assert fingerprint(notes) == screening['decisions_hash']
        assert fingerprint(plan) == screening['plan_hash']
        metrics = load(Path(plan['output'])/'metrics.json')
        assert metrics['adapter_sha256'] == screening['adapter_sha256']
        delta = load(f'runs/{prefix}-weight-update-diagnostics.json')['reports'][prefix]
        reports[prefix] = dict(counts=screening['counts'],recall_gate=screening['recall_first_gate_passed'],
            adapter_sha256=metrics['adapter_sha256'],peak_reserved_gib=metrics['peak_reserved_gib'],
            relative_parameter_delta_l2=delta['relative_delta_l2'],
            gradient_norm_before_clip_range=delta['recorded_gradient_norm_before_clip_range'],
            steps_with_gradient_clipping=delta['steps_with_gradient_clipping'],
            full_tokens=metrics['full_tokens'],forward_calls=metrics['forward_calls'],backward_calls=metrics['backward_calls'])
    cnotes = {n['id']:n for n in read_jsonl('runs/v17-strength-recall-manual.jsonl')}
    snotes = {n['id']:n for n in read_jsonl('runs/v18-sft-recall-manual.jsonl')}
    differences = [dict(id=rid,cpo_verdict=n['verdict'],sft_verdict=snotes[rid]['verdict'])
                   for rid,n in cnotes.items() if n['verdict'] != snotes[rid]['verdict']]
    write_json(path,dict(at=now(),same_fields=list(equal_fields),reports=reports,verdict_differences=differences,
        result='Both targets repaired2/11 TRAIN errors as full pass; neither passed the frozen3-pass gate. No evidence here that switching to direct SFT resolves under-learning.',
        limitations='One seed,11 TRAIN groups, no DEV run due failed recall gate. Same full input tokens/visits/updates, but SFT rejected forwards under no_grad give different backward graphs/FLOPs. No generalization or algorithm superiority conclusion.',
        next='One frozen broader actual-error/balanced200-family replay engineering trial, not a further LR or method grid.',
        stage_goal_complete=False,release_approved=False))
    print({'reports':reports,'verdict_differences':differences,'stable_method_claim':False})


if __name__ == '__main__':
    main()
