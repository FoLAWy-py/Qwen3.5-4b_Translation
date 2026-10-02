from collections import defaultdict
from witrans_tools.common import read_jsonl
from witrans_tools.data import validate_record


def test_training_mining_keeps_context_families_and_reverse_directions_together():
    rows = read_jsonl("data/prepared/v4-mining/pool.jsonl")
    sources = defaultdict(set)
    families = defaultdict(list)
    for row in rows:
        validate_record(row, True, True)
        assert row["source"]["training_allowed"]
        sources[row["input"]["text"]].add(row["group_id"])
        families[row["group_id"]].append(row)
    assert len(rows) == 48 and len(families) == 20
    assert all(len(groups) == 1 for groups in sources.values())
    assert all({r["input"]["target_lang"] for r in family} == {"en", "zh-CN"} for family in families.values())
    assert sum(len(family) == 4 for family in families.values()) == 4
