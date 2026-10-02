import copy
import json

import pytest

from scripts import freeze_v12_public
from archive.qwen3.scripts.prepare_public_short_pilot import DEST
from witrans_tools.common import read_jsonl


def test_corrected_pair_cannot_inherit_another_translation(monkeypatch):
    originals = read_jsonl(DEST / 'candidates.jsonl')
    reviewed = copy.deepcopy(read_jsonl(DEST / 'reviewed-partial.jsonl'))
    reviewed[0]['zh'] = '审核后被替换的译文。'
    monkeypatch.setattr(freeze_v12_public, 'read_jsonl',
                        lambda path: originals if path.name == 'candidates.jsonl' else reviewed)
    with pytest.raises(ValueError, match='without explicit acceptance'):
        freeze_v12_public.checked_rows()


def test_new_family_link_requires_review(monkeypatch, tmp_path):
    audit = json.loads((DEST / 'family-audit.json').read_text(encoding='utf-8'))
    audit['family_links'].append({'a': 'public-short-0001', 'b': 'public-short-0002'})
    (tmp_path / 'family-audit.json').write_text(json.dumps(audit), encoding='utf-8')
    monkeypatch.setattr(freeze_v12_public, 'DEST', tmp_path)
    monkeypatch.setattr(freeze_v12_public, 'read_jsonl', lambda path: read_jsonl(DEST / path.name))
    with pytest.raises(ValueError, match='New family links'):
        freeze_v12_public.checked_rows()
