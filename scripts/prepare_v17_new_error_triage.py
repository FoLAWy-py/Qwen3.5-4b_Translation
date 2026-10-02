"""Freeze a forty-family balanced TRAIN-only probe outside the previous64-family probe."""
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    dest=Path('data/prepared/v17-new-error-triage')
    assert not dest.exists(), 'Preserve triage selection'
    corpus_path='data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl'
    rows=read_jsonl(corpus_path)
    manifest=json.loads(Path('data/prepared/v16-reviewed-families/resolved-manifest.json').read_text(encoding='utf-8'))
    assert fingerprint(rows)==manifest['train_ready_hash'] and manifest['train_ready_families']==200
    prior_records=read_jsonl('runs/v15-training-hardness/review-selected-records.jsonl')
    excluded={r['group_id'] for r in prior_records}
    scores={s['id']:s for s in read_jsonl('runs/v15-training-hardness/scores.jsonl')}
    old={r['id']:r for r in read_jsonl('data/prepared/v12-factorial-v2/mixed-train.jsonl')}
    selected=[]; used=set(excluded)
    for category in ('daily','travel','food','academic','hard'):
        for direction in ('en','zh-CN'):
            eligible=[r for r in rows if r['category']==category and r['input']['target_lang']==direction
                      and r['group_id'] not in used and r['id'] in scores and 'reference_repair' not in r]
            for r in eligible:
                assert fingerprint(old[r['id']])==scores[r['id']]['record_hash']
                assert r['input']==old[r['id']]['input'] and r['output']==old[r['id']]['output']
            eligible.sort(key=lambda r:(-scores[r['id']]['answer_eos_nll'],r['id']))
            chosen=[]
            for row in eligible:
                if row['group_id'] in used:
                    continue
                chosen.append(row);used.add(row['group_id'])
                if len(chosen)==4:
                    break
            assert len(chosen)==4,(category,direction,len(chosen))
            selected.extend(chosen)
    assert len(selected)==len({r['group_id'] for r in selected})==40
    for row in selected:
        validate_record(row,True,True,purpose='training')
    starting=json.loads(Path('data/prepared/v15-error-repair/plan.json').read_text(encoding='utf-8'))
    write_jsonl(dest/'records.jsonl',selected)
    write_json(dest/'plan.json',dict(at=now(),input=str(dest/'records.jsonl'),output='runs/v17-new-error-triage.jsonl',
        data_hash=fingerprint(selected),corpus_hash=fingerprint(rows),previous_probe_hash=fingerprint(prior_records),
        starting_adapter=starting['starting_adapter'],adapter_sha256=starting['starting_adapter_sha256'],
        base=starting['base'],prompt_hash=starting['prompt_hash'],quantization='nf4',max_length=1024,max_new_tokens=256,
        families=40,rows=40,strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in selected)),
        selection='Four top original TRAIN NLL records per category/direction, one per family, excluding every previous64-probe family; repaired references excluded from stale-NLL selection.',
        scope='Pre-reviewed TRAIN-only source/error diagnosis with current best v7; not a heldout comparison, not an original-base job. NLL cannot assign error labels. Actual outputs must be individually read.',
        negative_training_authorized=False,release_approved=False))
    print({'families':40,'strata':dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in selected)),
           'status':'frozen CPU preparation only; no GPU job started'})


if __name__=='__main__':
    main()
