"""Freeze matched authored/real-negative CPO/SFT trial from v5; no old test data."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

def main():
    destination = Path('data/prepared/v6-preference')
    if destination.exists():
        raise ValueError('Trial already frozen')
    train, lineage = [], []
    for name in ('v3/train.jsonl', 'v4-mining/preferences.jsonl', 'v8-mining/preferences.jsonl'):
        path = Path('data/prepared') / name
        rows = read_jsonl(path)
        for row in rows:
            validate_record(row, True, True)
            if row['preference_review']['hash'] != fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')}):
                raise ValueError('Unreviewed preference')
        train.extend(rows)
        lineage.append({'path':str(path), 'rows':len(rows), 'hash':fingerprint(rows)})
    if len({r['id'] for r in train}) != len(train):
        raise ValueError('Duplicate preference ID')
    heldout = []
    for path in Path('data/prepared').glob('**/*.jsonl'):
        if 'short-benchmark' in str(path):
            continue  # Performance-only aliases include approved training rows.
        for row in read_jsonl(path):
            if 'input' in row and (any(tag in path.name for tag in ('dev','test','acceptance'))
                    or row.get('source', {}).get('training_allowed') is False):
                heldout.append(row)
    if {r['group_id'] for r in train} & {r['group_id'] for r in heldout}:
        raise ValueError('Heldout group leaked')
    if {r['input']['text'].strip().casefold() for r in train} & {r['input']['text'].strip().casefold() for r in heldout}:
        raise ValueError('Heldout source leaked')
    dev = read_jsonl('data/prepared/v3/dev.jsonl')
    manifest = {'at':now(), 'lineage':lineage,
        'train':{'rows':len(train), 'groups':len({r['group_id'] for r in train}), 'hash':fingerprint(train)},
        'dev':{'rows':len(dev), 'hash':fingerprint(dev)},
        'protocol':{'starting_adapter':'models/witrans-4b-v5-sft/checkpoint-85',
            'starting_adapter_sha256':'1ed0446f66cdbcec444c95233355e31e7645c8c3bc70a70fe5759a79723b9736',
            'modes':['cpo','sft_control'], 'steps':22, 'accumulation':8, 'lr':1e-5, 'seed':42,
            'record_visits':176, 'beta':0.1,
            'screening':'Known 16-pair dev objective auxiliary. Both candidates must undergo 200-row semantic development screening; no automatic NLL promotion.',
            'release_approved':False},
        'limitations':'Synthetic sources, small real-negative pool; old training positives are replay. Matched updates/two forwards, not equal FLOPs. No efficacy claim before semantic review.'}
    write_jsonl(destination / 'train.jsonl', train)
    write_jsonl(destination / 'dev.jsonl', dev)
    write_json(destination / 'manifest.json', manifest)
    print(manifest)

if __name__ == '__main__':
    main()
