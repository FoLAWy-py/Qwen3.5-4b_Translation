import pytest
from archive.qwen3.scripts import review_public_short


def test_changed_public_source_cannot_inherit_reading(monkeypatch, tmp_path):
    monkeypatch.setattr(review_public_short, 'DEST', tmp_path)
    monkeypatch.setattr(review_public_short, 'read_jsonl', lambda _: [
        {'id': 'public-short-0001', 'en': 'A replaced source.', 'zh': '替换后的原文。'}])
    with pytest.raises(ValueError, match='Previously read public candidates changed'):
        review_public_short.main()
    assert not (tmp_path / 'decisions.jsonl').exists()
    assert not (tmp_path / 'reviewed-partial.jsonl').exists()
