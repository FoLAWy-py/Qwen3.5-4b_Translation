import re
import pytest

from witrans_tools.common import read_jsonl
from witrans_tools.data import validate_record


def test_acceptance_only_rows_are_evaluable_but_not_trainable():
    row = read_jsonl("data/prepared/v2/bf16_test.jsonl")[0]
    assert row["source"]["training_allowed"] is False
    validate_record(row, True, True, purpose="evaluation")
    with pytest.raises(ValueError, match="训练"):
        validate_record(row, True, True)


def test_v2_approved_pairs_have_correct_language_direction():
    # Regression: two authored pairs originally had their language fields reversed.
    for split in ("train", "dev", "test"):
        path = "data/prepared/v2/dev_corrected.jsonl" if split == "dev" else f"data/prepared/v2/{split}.jsonl"
        for row in read_jsonl(path):
            validate_record(row, True, True)
            source_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", row["input"]["text"]))
            target_has_chinese = bool(re.search(r"[\u4e00-\u9fff]", row["output"]["translation"]))
            if row["input"]["target_lang"] == "en":
                assert source_has_chinese and not target_has_chinese, row["id"]
            else:
                assert not source_has_chinese and target_has_chinese, row["id"]


def test_v2_never_replays_old_heldout_sources_and_keeps_context_variants_together():
    datasets = {name: read_jsonl("data/prepared/v2/dev_corrected.jsonl" if name == "dev" else f"data/prepared/v2/{name}.jsonl") for name in ("train", "dev", "test")}
    groups = {name: {r["group_id"] for r in rows} for name, rows in datasets.items()}
    texts = {name: {r["input"]["text"].strip().casefold() for r in rows} for name, rows in datasets.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not groups[a] & groups[b]
        assert not texts[a] & texts[b]
    heldout = read_jsonl("data/prepared/v1/dev.jsonl") + read_jsonl("data/prepared/v1/test.jsonl")
    assert not texts["train"] & {r["input"]["text"].strip().casefold() for r in heldout}
    replay = [r for r in datasets["train"] if r["id"].startswith("v2-replay-")]
    assert len(replay) == 150 and len({r["group_id"] for r in replay}) == 75
    same_source = [r for r in datasets["train"] if r["input"]["text"] == "The port is blocked."]
    assert len(same_source) == 2 and len({r["group_id"] for r in same_source}) == 1
