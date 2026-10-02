import pytest
from witrans_tools.common import read_jsonl, fingerprint
from witrans_tools.data import validate_record
import json
from pathlib import Path

def test_v4_development_is_large_enough_for_screening_but_cannot_be_training():
    train, dev = [read_jsonl(f'data/prepared/v4/{name}.jsonl') for name in ('train', 'dev')]
    manifest = json.loads(Path('data/prepared/v4/manifest.json').read_text(encoding='utf-8'))
    assert len(dev) == 200 and len({r['group_id'] for r in dev}) == 100
    assert fingerprint(train) == manifest['train']['hash']
    assert fingerprint(dev) == manifest['dev']['hash']
    assert not {r['input']['text'] for r in train} & {r['input']['text'] for r in dev}
    assert not {r['group_id'] for r in train} & {r['group_id'] for r in dev}
    for row in dev:
        validate_record(row, True, True, purpose='evaluation')
        with pytest.raises(ValueError):
            validate_record(row, True, True)
