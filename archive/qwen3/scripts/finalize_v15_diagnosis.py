"""Summarize complete bound recall, DEV regressions and measured signal limits."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    destination = Path('runs/v15-error-diagnosis.json')
    assert not destination.exists(), 'Preserve diagnosis'
    report = json.loads(Path('runs/v15-error-model-acceptance.json').read_text(encoding='utf-8'))
    rows = read_jsonl('runs/v15-repair-recall.jsonl')
    decisions = read_jsonl('runs/v15-repair-recall-manual.jsonl')
    refs = read_jsonl('runs/v15-repair-recall-input.jsonl')
    indexed = {r['id']: r for r in rows}
    assert len(rows) == len(decisions) == len(refs) == 11
    assert {r['id'] for r in refs} == {r['id'] for r in decisions} == set(indexed)
    for d in decisions:
        assert d['generation_hash'] == fingerprint(indexed[d['id']])
        assert d['reviewer'] == 'Codex' and d['note'] and d['language_correct']
    signal = read_jsonl('runs/v15-repair-signal.jsonl')
    assert len(signal) == 22
    scores = {(r['role'], r['id']): r for r in signal}
    progress = json.loads(Path('runs/v15-repair-signal-progress.json').read_text(encoding='utf-8'))
    assert progress['phase'] == 'diagnostic_complete'
    deltas = [dict(id=r['id'], chosen_nll_before=scores['v7',r['id']]['chosen_nll'],
                   chosen_nll_after=scores['v15',r['id']]['chosen_nll'],
                   margin_before=scores['v7',r['id']]['summed_margin'],
                   margin_after=scores['v15',r['id']]['summed_margin']) for r in refs]
    strata = {}
    for split, candidate_path, parent_path in [('known','runs/v15-error-known-semantic.jsonl','runs/v15-error-start-dev-semantic.jsonl'),
                                               ('public','runs/v15-error-public-semantic.jsonl','runs/v13-v7-public-semantic.jsonl')]:
        groups = defaultdict(lambda: {'v7': Counter(), 'v15': Counter()})
        for role, path in [('v7',parent_path),('v15',candidate_path)]:
            for d in read_jsonl(path):
                groups[d['category']+'/'+d['target_lang']][role][d['verdict']] += 1
        strata[split] = {k:{role:dict(counts) for role,counts in bucket.items()} for k,bucket in groups.items()}
    result = dict(at=now(), primary_diagnosis='A: none of the eleven TRAIN errors learned under the frozen16 updates',
        recall=dict(rows=11, source_groups=11, counts=dict(Counter(d['verdict'] for d in decisions)),
                    pass_repair_rate=0.0, major_including_critical=11, critical=1,
                    all_original_errors_persist=True, decisions_hash=fingerprint(decisions), generation_hash=fingerprint(rows)),
        known=report['counts'], known_paired_stats=report['comparisons']['cpo']['paired_stats'],
        public=dict(v7={'pass':95,'minor':18,'major':3},v15={'pass':97,'minor':17,'major':2},
                    paired_stats=report['public_auxiliary']['paired_stats']), strata=strata,
        signal=dict(before_after=deltas, all_chosen_nll_improved=all(d['chosen_nll_after']<d['chosen_nll_before'] for d in deltas),
                    all_margins_still_negative=all(d['margin_after']<0 for d in deltas),
                    v7_preference_to_sft_token_score_gradient_ratio_range=[min(r['preference_to_chosen_sft_token_score_gradient_norm_ratio'] for r in signal if r['role']=='v7'), max(r['preference_to_chosen_sft_token_score_gradient_norm_ratio'] for r in signal if r['role']=='v7')],
                    interpretation='CPO token-score derivatives are not near-zero (unlike v14 matching). Likelihood moves in the desired direction but does not change greedy TRAIN semantic outcomes. This does not measure parameter gradients or prove causality.'),
        selection='v15 not promoted; best remains v7', optimization_screen_passed=False,
        limitations='Known DEV200 repeatedly used; public116 is auxiliary. Academic strata each n20 lose one pass (5pp), too small to infer stable capability changes. Public paired95%CI touches zero. Exact identical outputs reuse bound frozen v7 reviews; changed outputs have new explicit readings. No release, independent confirmation, or three-round performance claim.',
        next_hypothesis='A single small same-data/same-budget trial of existing critical-span-weighted chosen-answer supervision, with explicit TRAIN span review and recall-first stopping rule; complete balanced source audit in parallel on CPU.',
        stage_goal_complete=False, release_approved=False)
    write_json(destination, result)
    print({'primary_diagnosis':result['primary_diagnosis'],'recall':result['recall']['counts'],'best':'v7'})


if __name__ == '__main__':
    main()
