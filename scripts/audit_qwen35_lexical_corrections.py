"""Check corrected review views against every immutable actual generation/reference."""
from collections import Counter
from pathlib import Path

from scripts.decode_qwen35_repair_recall import load
from scripts.review_v12_factorial import accept_manual, binding
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record


def main():
    dest=Path('runs/qwen35-v3-lexical-correction-binding-audit.json')
    assert not dest.exists()
    correction=load('runs/qwen35-v3-lexical-negative-correction.json')
    old_trial=load('data/prepared/qwen35-v3-reviewed34-repair/plan.json')
    scenarios=[dict(name='frozen_start416',generation='runs/qwen35-v3-train-discovery-recovery1.jsonl',
                    review=correction['outputs']['original_review'],
                    refs='data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl',expected=416),
               dict(name='quarantined_candidate114',generation='runs/qwen35-v3-reviewed34-recall.jsonl',
                    review=correction['outputs']['candidate_review'],refs=old_trial['recall_path'],expected=114)]
    evidence=[]
    for scenario in scenarios:
        raw=read_jsonl(scenario['generation']);notes=read_jsonl(scenario['review']);refs=read_jsonl(scenario['refs'])
        assert len(raw)==len(notes)==len(refs)==scenario['expected']
        generated={r['id']:r for r in raw};reference={r['id']:r for r in refs}
        assert len(generated)==len(reference)==len({r['id'] for r in notes})==scenario['expected']
        assert set(generated)==set(reference)=={r['id'] for r in notes}
        for note in notes:
            row=generated[note['id']];ref=reference[note['id']]
            accept_manual(row,note)
            validate_record(ref,True,True,purpose='training')
            assert row['input']==ref['input'] and row['reference']==ref['output']
            assert note['binding_hash']==binding(row) and note['generation_hash']==fingerprint(row)
            assert note['chosen_content_hash']==ref['review']['content_hash']
            assert (note['group_id'],note['category'],note['target_lang'])==(ref['group_id'],ref['category'],ref['input']['target_lang'])
        evidence.append(dict(**scenario,raw_hash=fingerprint(raw),review_hash=fingerprint(notes),reference_hash=fingerprint(refs),
                             counts=dict(Counter(n['verdict'] for n in notes)),all_content_bindings_valid=True))
    pairs=read_jsonl(correction['outputs']['clean_pairs'])
    start_raw={r['id']:r for r in read_jsonl(scenarios[0]['generation'])}
    for pair in pairs:
        assert pair['rejected']==start_raw[pair['id']]['prediction']
        assert pair['input']==start_raw[pair['id']]['input']
    assert len(pairs)==33 and len({r['group_id'] for r in pairs})==32
    write_json(dest,dict(at=now(),correction_report_hash=fingerprint(correction),views=evidence,
        clean_pairs_hash=fingerprint(pairs),all33_negatives_are_actual_original_start_predictions=True,
        no_raw_output_modified=True,old_candidate_eligible=False,stage_goal_complete=False))
    print(dict(audit=str(dest),bound_rows=530,actual_negative_rows=33),flush=True)


if __name__=='__main__':
    main()
