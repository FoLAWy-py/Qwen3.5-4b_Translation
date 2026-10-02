import json

import pytest

from witrans import WiTrans, make_messages, parse_translation
from witrans_tools.common import fingerprint
from witrans_tools.data import encode_example, split_records


@pytest.mark.parametrize("raw", [
    '{"translation":"a","translation":"b"}',
    '```json\n{"translation":"a"}\n```',
    '{"translation":"a","reason":"b"}',
    '{"translation":null}', '{"translation":"unfinished}',
])
def test_reject_invalid_model_json(raw):
    with pytest.raises(ValueError):
        parse_translation(raw)


def test_literal_think_in_translation_is_preserved():
    assert parse_translation('{"translation":"<think> is a literal tag"}')["translation"].startswith("<think>")


def test_payload_has_no_category_and_validates_glossary():
    payload = json.loads(make_messages("Hello", "en")[1]["content"])
    assert set(payload) == {"text", "target_lang", "context", "glossary"}
    with pytest.raises(ValueError):
        make_messages("Hello", "en", glossary={"a": ""})


class Tokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return list(text.encode())


def record(i, group=None):
    row = {"id": str(i), "group_id": group or str(i), "category": "daily",
           "input": {"text": f"example {i}", "target_lang": "en", "context": "", "glossary": {}},
           "output": {"translation": f"translation {i}"},
           "source": {"name": "unit-test", "license": "original", "training_allowed": True, "external_labeling_allowed": True}}
    row["review"] = {"status": "approved", "reviewer": "unit-test",
                     "content_hash": fingerprint({"input": row["input"], "output": row["output"]})}
    return row


def test_loss_mask_and_no_truncation():
    row = encode_example(Tokenizer(), record(1))
    assert row["labels"][:3] == [-100] * 3
    assert row["labels"][3:] == row["input_ids"][3:]
    assert row["labels"][-1] == 999
    with pytest.raises(ValueError, match="不能截断"):
        encode_example(Tokenizer(), record(1), max_length=4)


def test_split_keeps_related_records_together():
    rows = [record(i, str(i // 2)) for i in range(20)]
    splits = split_records(rows)
    groups = [{r["group_id"] for r in split} for split in splits.values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    assert sum(map(len, splits.values())) == len(rows)
    assert splits == split_records(rows)


def test_unreviewed_and_changed_records_rejected():
    rows = [record(i) for i in range(10)]
    rows[0]["output"]["translation"] = "changed"
    with pytest.raises(ValueError, match="审核"):
        split_records(rows)


def test_missing_adapter_cannot_be_baseline(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="adapter"):
        WiTrans(str(tmp_path), str(tmp_path / "missing"))
