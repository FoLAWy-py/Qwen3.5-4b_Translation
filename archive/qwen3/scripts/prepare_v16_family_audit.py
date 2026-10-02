"""Select balanced authorized TRAIN families for source/label reading, not training."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    dest = Path('data/prepared/v16-family-audit')
    if dest.exists():
        raise ValueError('Preserve selected audit families')
    prior = json.loads(Path('data/prepared/v13-public-optimization/plan.json').read_text(encoding='utf-8'))
    pool = read_jsonl(prior['datasets']['mixed']['path'])
    assert fingerprint(pool) == prior['datasets']['mixed']['hash']
    scores = read_jsonl('runs/v15-training-hardness/scores.jsonl')
    indexed = {r['id']: r for r in pool}
    by_id = {s['id']: s for s in scores}
    assert len(indexed) == len(pool) and len(by_id) == len(scores)
    families = defaultdict(list)
    heldout = sum([read_jsonl(prior[k]['path']) for k in ('dev', 'public_dev')], [])
    forbidden_groups = {r['group_id'] for r in heldout}
    forbidden_texts = {r['input']['text'].strip().casefold() for r in heldout}
    for row in pool:
        validate_record(row, True, True, purpose='training')
        assert row['group_id'] not in forbidden_groups
        assert row['input']['text'].strip().casefold() not in forbidden_texts
        if row['id'] in by_id:
            assert by_id[row['id']]['record_hash'] == fingerprint(row)
        families[row['group_id']].append(row)
    buckets = defaultdict(list)
    for group, rows in families.items():
        categories = {r['category'] for r in rows}
        if len(categories) == 1 and {r['input']['target_lang'] for r in rows} == {'en', 'zh-CN'}:
            nll = max((by_id[r['id']]['answer_eos_nll'] for r in rows if r['id'] in by_id), default=-1)
            buckets[next(iter(categories))].append((nll, group, rows))
    selected, records = [], []
    for category in ('daily', 'travel', 'food', 'academic', 'hard'):
        ranked = sorted(buckets[category], key=lambda entry: (-entry[0], entry[1]))
        assert len(ranked) >= 40, (category, len(ranked))
        for nll, group, rows in ranked[:40]:
            records.extend(rows)
            selected.append(dict(group_id=group, category=category, max_answer_nll=nll,
                                 record_ids=[r['id'] for r in rows], records_hash=fingerprint(rows),
                                 audit_status='pending_individual_source_and_label_reading'))
    assert len(selected) == 200
    write_jsonl(dest/'records.jsonl', records)
    write_jsonl(dest/'families.jsonl', selected)
    write_json(dest/'plan.json', dict(at=now(), source_pool=prior['datasets']['mixed'],
        source_score_hash=fingerprint(scores), record_hash=fingerprint(records), family_hash=fingerprint(selected),
        families=200, rows=len(records), strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in records)),
        selection='Top max TRAIN answer NLL within each category; 40 whole bidirectional families per category. NLL selects reading priority only, never an error label. Keep all paraphrases and both directions within their original group.',
        heldout_hashes={k:prior[k]['hash'] for k in ('dev', 'public_dev')},
        leakage_scope='Existing source groups and exact normalized source strings checked; prior source-family provenance retained. Semantic near-overlap still requires reading; no DEV content used for authoring.',
        training_authorized=False, new_negative_authorized=False,
        scope='Pending source/label coverage audit. Not an approved new training corpus.', release_approved=False))
    print({'families': len(selected), 'rows': len(records), 'pending': True})


if __name__ == '__main__':
    main()
