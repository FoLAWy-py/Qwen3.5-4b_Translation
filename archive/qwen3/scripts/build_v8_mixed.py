"""Freeze a full multi-sentence SFT extension, retaining reviewed replay."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    destination = Path('data/prepared/v8-mixed')
    if destination.exists():
        raise ValueError('Preserve frozen mixed corpus')
    parent = Path('models/witrans-4b-v7-critical-cpo')
    expected_parent = '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424'
    with (parent / 'adapter_model.safetensors').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected_parent:
            raise ValueError('Frozen parent weights changed')
    sources = [('v5-mixed', '9757eaf8a0a666c317326a069dea258fc8271a46e6c6ac12f3fcf9bfafff4c5f'),
        ('v8-multisentence', '28b9b54e5eb122af77a8c90fc2f6c5b217228adbcafeb72f2fedd8a80a8b6245')]
    train, lineage = [], []
    for name, expected in sources:
        path = Path('data/prepared') / name / 'train.jsonl'
        rows = read_jsonl(path)
        if fingerprint(rows) != expected:
            raise ValueError('Reviewed source changed')
        for row in rows:
            validate_record(row, True, True)
        train.extend(rows)
        lineage.append({'path':str(path), 'rows':len(rows), 'hash':expected})
    if len(train) != 758 or len({r['id'] for r in train}) != len(train):
        raise ValueError('Unexpected count or duplicate IDs')
    heldout = []
    for path in Path('data/prepared').glob('**/*.jsonl'):
        for row in read_jsonl(path):
            if 'input' in row and (any(term in path.name for term in ('dev','test','acceptance'))
                    or row.get('source', {}).get('training_allowed') is False):
                heldout.append(row)
    if {r['group_id'] for r in train} & {r['group_id'] for r in heldout}:
        raise ValueError('Held-out group leakage')
    if {r['input']['text'].strip().casefold() for r in train} & {
            r['input']['text'].strip().casefold() for r in heldout}:
        raise ValueError('Held-out exact source leakage')
    dev = read_jsonl('data/prepared/v4/dev.jsonl')
    if fingerprint(dev) != 'a4ec0478d5aca8b263e605bfe3e7ec2b57dd9411510135cfaf227575ede03237':
        raise ValueError('Known development corpus changed')
    for row in dev:
        validate_record(row, True, True, purpose='evaluation')
    configuration = {'method':'Full multi-sentence positive SFT with reviewed replay from v7',
        'starting_adapter':str(parent), 'starting_adapter_sha256':expected_parent,
        'lr':1e-5, 'steps':48, 'accumulation':16, 'seed':42, 'checkpoint_steps':[24,48],
        'selection':'Lowest known development answer/EOS NLL before new generation; no release claim',
        'record_visits':768, 'nominal_epochs':768/len(train)}
    manifest = {'at':now(), 'lineage':lineage,
        'train':{'rows':len(train), 'groups':len({r['group_id'] for r in train}),
            'hash':fingerprint(train), 'categories':dict(Counter(r['category'] for r in train))},
        'dev':{'rows':len(dev), 'groups':len({r['group_id'] for r in dev}), 'hash':fingerprint(dev)},
        'configuration':configuration,
        'policy':'Training-only full approved80-row addition. Known development reused adaptively for screening; all prepared held-out groups and exact sources excluded, no full semantic near-duplicate guarantee. No new release test.'}
    write_jsonl(destination / 'train.jsonl', train)
    write_jsonl(destination / 'dev.jsonl', dev)
    write_json(destination / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
