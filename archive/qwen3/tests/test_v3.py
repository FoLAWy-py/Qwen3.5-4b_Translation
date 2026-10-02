import re

from witrans_tools.common import fingerprint, read_jsonl
from witrans_tools.data import validate_record


def test_v3_preferences_are_bound_and_source_groups_are_isolated():
    datasets = {name: read_jsonl(f"data/prepared/v3/{name}.jsonl") for name in ("train", "dev", "test")}
    groups, texts = [], []
    for name, rows in datasets.items():
        groups.append({r["group_id"] for r in rows})
        texts.append({r["input"]["text"] for r in rows})
        for row in rows:
            validate_record(row, True, True, purpose="evaluation")
            assert row["output"] != row["rejected"]
            assert row["preference_review"]["hash"] == fingerprint({k: row[k] for k in ("input", "output", "rejected", "preference_issue")})
            has_han = bool(re.search(r"[\u4e00-\u9fff]", row["output"]["translation"]))
            assert has_han == (row["input"]["target_lang"] == "zh-CN")
            assert row["source"]["training_allowed"] == (name != "test")
    for i in range(3):
        for j in range(i + 1, 3):
            assert not groups[i] & groups[j]
            assert not texts[i] & texts[j]
