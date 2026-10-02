"""Complete individually bound TRAIN diagnosis; no held-out efficacy claim."""
import json
from collections import Counter
from pathlib import Path
from archive.qwen3.scripts.review_v12_factorial import accept_manual, binding
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import encode_example


def main():
    destination=Path('runs/qwen35-v3-complete-train-diagnosis.json')
    assert not destination.exists()
    refs=read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    generated=read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')
    notes=read_jsonl('runs/qwen35-v3-train-discovery-recovery1-manual.jsonl')
    pool=read_jsonl('data/prepared/qwen35-v3-current-errors/pairs.jsonl')
    audit=json.loads(Path('runs/qwen35-v3-current-error-pool-final-audit.json').read_text(encoding='utf-8'))
    assert len(refs)==len(generated)==len(notes)==416 and audit['complete_review_validated']
    assert fingerprint(notes)==audit['review_hash'] and fingerprint(refs)==audit['train_hash']
    assert fingerprint(pool)==audit['pairs_hash']
    by_id={r['id']:r for r in generated};refs_by_id={r['id']:r for r in refs}
    assert set(by_id)==set(refs_by_id)=={n['id'] for n in notes}
    for n in notes:
        accept_manual(by_id[n['id']],n)
        assert n['binding_hash']==binding(by_id[n['id']])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    lengths={r['id']:len(encode_example(tokenizer,r,1024)['input_ids']) for r in refs}
    strata={}
    for category in ('daily','travel','food','academic','hard'):
        for direction in ('zh-CN','en'):
            selected=[n for n in notes if n['category']==category and n['target_lang']==direction]
            rows=[refs_by_id[n['id']] for n in selected]
            strata[category+'/'+direction]=dict(rows=len(selected),groups=len({r['group_id'] for r in rows}),
                verdicts=dict(Counter(n['verdict'] for n in selected)),
                format_valid=sum(n['format_valid'] for n in selected),ended=sum(n['ended'] for n in selected),
                direction_correct=sum(n['language_correct'] for n in selected),
                context_rows=sum(bool(r['input'].get('context')) for r in rows),
                glossary_rows=sum(bool(r['input'].get('glossary')) for r in rows),
                chosen_full_tokens=sum(lengths[r['id']] for r in rows),
                longest_chosen_sequence=max(lengths[r['id']] for r in rows))
    errors=[refs_by_id[n['id']] for n in notes if n['verdict'] in ('major','critical')]
    result=dict(at=now(),train_hash=fingerprint(refs),generation_hash=fingerprint(generated),review_hash=fingerprint(notes),
        pool_hash=fingerprint(pool),strata=strata,groups=len({r['group_id'] for r in refs}),
        verdicts=dict(Counter(n['verdict'] for n in notes)),
        direction_failure_ids=[n['id'] for n in notes if not n['language_correct']],
        correct_replay_candidates=[n['id'] for n in notes if n['verdict']=='pass'],
        errors=dict(rows=len(errors),groups=len({r['group_id'] for r in errors}),
            context_rows=sum(bool(r['input'].get('context')) for r in errors),
            glossary_rows=sum(bool(r['input'].get('glossary')) for r in errors)),
        observed='No major hard/en TRAIN error in47 approved rows. Replay correct positives there rather than inventing rejected translations. Errors concentrated infood/zh11 andtravel/zh8; diagnosis does not establish causal benefit of a recipe.',
        required_before_trial_freeze='Actual current chosen/rejected margins and measured critical-span/remaining/preference parameter gradients. This report alone does not select LR, budget or candidate.',
        scope='NLL-prioritized previously trained sources with AI semantic readings. Descriptive TRAIN diagnosis only, not independent performance estimate, DEV gate or release test.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(destination,result)
    print(dict(verdicts=result['verdicts'],direction_failure_ids=result['direction_failure_ids'],
        errors=result['errors'],strata={k:v['verdicts'] for k,v in strata.items()}),flush=True)


if __name__=='__main__':
    main()
