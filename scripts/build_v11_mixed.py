"""Freeze broadened reviewed SFT data and a lower-LR trial from the best DEV parent."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    destination = Path('data/prepared/v11-mixed')
    if destination.exists():
        raise ValueError('Preserve frozen v11 trial')
    parent = Path('models/witrans-4b-v7-critical-cpo')
    parent_sha = '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424'
    with (parent/'adapter_model.safetensors').open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=parent_sha:
            raise ValueError('Frozen parent changed')
    sources = [
        ('v8-mixed','3096225c7aa6025697354ea24ebde4222d3bf6f6c6f528b32b50a5d723c44390'),
        ('v9-constraints-context-checked','eb0e98c3875e5972d5391fd80f7614e9e5d62d5e0693a1f43db23ec1de630c90'),
        ('v10-source-family-checked','9cd15e368ef3837562a002f6b32ebc1e8c22bacb9d5ddc8901328c4d6103341a'),
    ]
    train,lineage = [],[]
    for name,expected in sources:
        path = Path('data/prepared')/name/'train.jsonl'
        rows = read_jsonl(path)
        if fingerprint(rows)!=expected:
            raise ValueError('Reviewed training source changed')
        for row in rows:
            validate_record(row,True,True)
        train.extend(rows)
        lineage.append({'path':str(path),'rows':len(rows),'hash':expected})
    if len(train)!=992 or len({r['id'] for r in train})!=992:
        raise ValueError('Unexpected count or repeated IDs')
    seen = {}
    for row in train:
        key = (row['input']['text'].strip().casefold(),row['input']['context'],
               fingerprint(row['input']['glossary']),row['input']['target_lang'])
        if key in seen:
            raise ValueError('Repeated identical training input')
        seen[key] = row['group_id']
    heldout = []
    for path in Path('data/prepared').glob('**/*.jsonl'):
        for row in read_jsonl(path):
            if 'input' in row and (any(term in path.name for term in ('dev','test','acceptance'))
                                 or row.get('source',{}).get('training_allowed') is False):
                heldout.append(row)
    if {r['group_id'] for r in train} & {r['group_id'] for r in heldout}:
        raise ValueError('Held-out group leakage')
    if {r['input']['text'].strip().casefold() for r in train} & {r['input']['text'].strip().casefold() for r in heldout}:
        raise ValueError('Held-out exact source leakage')
    dev = read_jsonl('data/prepared/v4/dev.jsonl')
    if fingerprint(dev)!='a4ec0478d5aca8b263e605bfe3e7ec2b57dd9411510135cfaf227575ede03237':
        raise ValueError('Known DEV changed')
    configuration = {'method':'Broader source-grounded SFT with reviewed replay; reduced LR trial',
        'starting_adapter':str(parent),'starting_adapter_sha256':parent_sha,
        'lr':5e-6,'steps':64,'accumulation':16,'seed':42,'checkpoint_steps':[32,64],
        'selection':'Lowest known development answer/EOS NLL before new generation; no release claim',
        'record_visits':1024,'nominal_epochs':1024/len(train)}
    manifest = {'at':now(),'lineage':lineage,
        'train':{'rows':len(train),'groups':len({r['group_id'] for r in train}),
            'hash':fingerprint(train),'categories':dict(Counter(r['category'] for r in train))},
        'dev':{'rows':len(dev),'groups':len({r['group_id'] for r in dev}),'hash':fingerprint(dev)},
        'configuration':configuration,
        'policy':'Training-only broadening. All992 labels previously individually accepted; v10 held-out retrieval families conservatively excluded, training near-families joined. Known DEV screening only. No full global semantic near-duplicate guarantee or new release test. More data and lower LR both change, so not a causal algorithm comparison.'}
    write_jsonl(destination/'train.jsonl',train)
    write_jsonl(destination/'dev.jsonl',dev)
    write_json(destination/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
