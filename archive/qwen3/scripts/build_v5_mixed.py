"""Freeze next SFT run after individually reviewed training-only extensions."""
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

SOURCES = ('v4', 'v5-seed', 'v6-source', 'v7-context')

def main():
    destination = Path('data/prepared/v5-mixed')
    if destination.exists():
        raise ValueError('Mixed corpus already frozen')
    train, lineage = [], []
    for source in SOURCES:
        path = Path('data/prepared') / source / 'train.jsonl'
        rows = read_jsonl(path)
        for row in rows:
            validate_record(row, True, True)
        train.extend(rows)
        lineage.append({'path': str(path), 'rows': len(rows), 'hash': fingerprint(rows)})
    if len({r['id'] for r in train}) != len(train):
        raise ValueError('Duplicate IDs')
    heldout = []
    for path in Path('data/prepared').glob('**/*.jsonl'):
        for row in read_jsonl(path):
            if 'input' not in row:
                continue
            if any(term in path.name for term in ('dev', 'test', 'acceptance')) or row.get('source', {}).get('training_allowed') is False:
                heldout.append(row)
    if {r['group_id'] for r in train} & {r['group_id'] for r in heldout}:
        raise ValueError('Held-out source group in training')
    if {r['input']['text'].strip().casefold() for r in train} & {r['input']['text'].strip().casefold() for r in heldout}:
        raise ValueError('Held-out text in training')
    dev = read_jsonl('data/prepared/v4/dev.jsonl')
    for row in dev:
        validate_record(row, True, True, purpose='evaluation')
    write_jsonl(destination / 'train.jsonl', train)
    write_jsonl(destination / 'dev.jsonl', dev)
    manifest = {'at': now(), 'lineage': lineage,
        'train': {'rows': len(train), 'groups': len({r['group_id'] for r in train}), 'hash': fingerprint(train),
            'categories': dict(Counter(r['category'] for r in train))},
        'dev': {'rows': len(dev), 'groups': len({r['group_id'] for r in dev}), 'hash': fingerprint(dev)},
        'policy': 'All prepared development/test/acceptance files and training-prohibited rows excluded by group and exact source. Known200-row v4 development reused for screening only; no release test. No complete semantic near-duplicate guarantee.',
        'configuration': {'method': 'continue SFT from v2 NF4; reviewed replay plus context contrasts', 'lr': 1e-5,
            'steps': 85, 'accumulation': 16, 'seed': 42, 'checkpoint_steps': [43, 85],
            'selection': 'Lowest dev answer/EOS NLL before generation; known development results cannot establish release acceptance',
            'record_visits': 1360, 'nominal_epochs': 1360 / len(train)}}
    write_json(destination / 'manifest.json', manifest)
    print(manifest)

if __name__ == '__main__':
    main()
