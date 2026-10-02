"""Preserve original labels, apply three explicit reviewed repairs, quarantine overlap hits."""
import copy
import difflib
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

REPAIRS = {
    'v10-food-15-zh-CN': (
        '顾客询问冰茶是没有添加甜味，还是加了蜂蜜增甜。顾客想知道是否没加糖，并不一定要求完全不使用任何甜味剂。',
        '顾客询问冰茶是没有加甜味剂，还是用蜂蜜增甜的。顾客想知道是否没有加糖，而不一定是想确认是否完全没有使用甜味剂。',
        '恢复wanted to know的询问意图，不把是否未加甜味剂改成要求不加。'),
    'v8-multi-031-en': (
        'The report refers to this metric as boundary retention. Please always use the agreed terminology in the abstract; if it could be confused with another metric, do not abbreviate it to retention.',
        'The report calls this metric boundary retention. Use the agreed term throughout the summary; if it could be confused with another metric, do not shorten it to retention.',
        '摘要应为summary，不无依据窄化成学术abstract；保留完整词表术语。'),
    'v2-train-101-en': (
        'The writing of this historical document is much later than the events it describes.',
        'This historical document was written long after the events it describes.',
        '自然表达史料写于其所述事件很久以后，保留时间关系。'),
}


def main():
    destination = Path('data/prepared/v16-reviewed-families')
    assert not destination.exists(), 'Preserve reviewed corpus'
    rows = read_jsonl('data/prepared/v16-family-audit/records.jsonl')
    families = read_jsonl('data/prepared/v16-family-audit/families.jsonl')
    decisions = read_jsonl('runs/v16-family-source-audit.jsonl')
    index = {d['group_id']: d for d in decisions}
    assert len(index) == len(decisions) == len(families) == 200
    for family in families:
        assert index[family['group_id']]['records_hash'] == family['records_hash']
    assert set(REPAIRS) <= {r['id'] for r in rows}
    repaired, repair_receipts = [], []
    for row in rows:
        validate_record(row,True,True,purpose='training')
        decision = index[row['group_id']]
        assert decision['status'] in ('approved','needs_repair')
        record = copy.deepcopy(row)
        if row['id'] in REPAIRS:
            expected, replacement, note = REPAIRS[row['id']]
            assert row['output']['translation'] == expected
            record['output'] = {'translation':replacement}
            record['review'] = dict(status='approved',reviewer='Codex (user-authorized AI acceptance)',at=now(),
                method='Explicit source-grounded reference repair after200 family readings; immutable input/source/group retained.',
                notes=note,content_hash=fingerprint({'input':record['input'],'output':record['output']}))
            receipt = dict(id=row['id'], group_id=row['group_id'], source_record_hash=fingerprint(row),
                           input_hash=fingerprint(row['input']), prior_review=row['review'],
                           prior_output=row['output'], repaired_output=record['output'], note=note,
                           reviewer='Codex',at=now(),content_hash=record['review']['content_hash'])
            repair_receipts.append(receipt)
            record['reference_repair'] = receipt
        record['family_source_audit'] = dict(path='runs/v16-family-source-audit.jsonl',
                                             decision_hash=fingerprint(decision),original_family_records_hash=decision['records_hash'])
        assert record['input'] == row['input'] and record['source'] == row['source'] and record['group_id'] == row['group_id']
        validate_record(record,True,True,purpose='training')
        repaired.append(record)
    for group,d in index.items():
        if d['status'] == 'needs_repair':
            assert any(r['group_id'] == group and r['id'] in REPAIRS for r in rows)
    prior = json.loads(Path('data/prepared/v13-public-optimization/plan.json').read_text(encoding='utf-8'))
    heldout = sum([read_jsonl(prior[k]['path']) for k in ('dev','public_dev')],[])
    forbidden_groups = {r['group_id'] for r in heldout}
    def normalized(row):
        return ' '.join(row['input']['text'].casefold().split())
    forbidden_texts = {normalized(r) for r in heldout}
    hits = []
    for r in repaired:
        assert r['group_id'] not in forbidden_groups and normalized(r) not in forbidden_texts
        source = normalized(r)
        candidates = [(difflib.SequenceMatcher(None,source,normalized(h),autojunk=False).ratio(),h)
                      for h in heldout if h['input']['target_lang'] == r['input']['target_lang']]
        score, h = max(candidates,key=lambda v:v[0])
        if score >= .68:
            hits.append(dict(train_id=r['id'],train_group=r['group_id'],heldout_id=h['id'],
                             heldout_group=h['group_id'],source_similarity=score,
                             decision='quarantine whole TRAIN family pending explicit near-source overlap review'))
    quarantine = {r['train_group'] for r in hits}
    ready = [r for r in repaired if r['group_id'] not in quarantine]
    write_jsonl(destination/'all-reviewed.jsonl',repaired)
    write_jsonl(destination/'train-ready.jsonl',ready)
    write_jsonl(destination/'quarantined.jsonl',[r for r in repaired if r['group_id'] in quarantine])
    write_jsonl(destination/'reference-repairs.jsonl',repair_receipts)
    write_json(destination/'overlap-retrieval.json',dict(at=now(),threshold=.68,hits=hits,
        scope='Source-text retrieval only; no heldout answers read for repair or authoring. High similarity is an audit flag, not proof of leakage. Entire bidirectional/context/paraphrase family quarantined. Prior grouping provenance retained.'))
    write_json(destination/'manifest.json',dict(at=now(),reviewed_families=200,reviewed_rows=len(rows),
        decision_hash=fingerprint(decisions),original_records_hash=fingerprint(rows),reviewed_records_hash=fingerprint(repaired),
        train_ready_hash=fingerprint(ready),train_ready_families=len({r['group_id'] for r in ready}),
        train_ready_rows=len(ready),quarantined_families=sorted(quarantine),reference_repairs=len(repair_receipts),
        strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in ready)),
        source_selection='TRAIN-only NLL priority inside five balanced categories; all references individually read in both directions.',
        data_role='Authorized reviewed existing TRAIN positive sources, not independent confirmation; no new negative labels.',
        new_negative_authorized=False,heldout_hashes={k:prior[k]['hash'] for k in ('dev','public_dev')},release_approved=False))
    print({'reviewed_families':200,'reference_repairs':3,'train_ready_families':len({r['group_id'] for r in ready}),
           'quarantined_families':sorted(quarantine)})


if __name__ == '__main__':
    main()
