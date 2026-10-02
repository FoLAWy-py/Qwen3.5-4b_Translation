import json
from collections import Counter

from data.v1_families import FAMILIES
from witrans_tools.common import read_jsonl, fingerprint
from witrans_tools.data import check_constraints, validate_record


def test_accepted_v1_family_substitutions_preserve_numbers_and_protected_text():
    assert sum(map(len, FAMILIES.values())) == 125
    for pairs in FAMILIES.values():
        for en, zh in pairs:
            for n in (2, 40):
                source, target = en.format(n=n), zh.format(n=n)
                assert not check_constraints(source, target)
                assert not check_constraints(target, source)


def test_v1_frozen_splits_do_not_leak_families_or_reverse_pairs():
    manifest = json.load(open("data/prepared/v1/split_manifest.json", encoding="utf-8"))
    groups = {}
    texts = {}
    for split in ("dev", "test"):
        rows = read_jsonl(f"data/prepared/v1/{split}.jsonl")
        assert len(rows) == 1000
        assert fingerprint(rows) == manifest["splits"][split]["sha256"]
        for row in rows:
            validate_record(row, True, True)
        groups[split] = {r["group_id"] for r in rows}
        texts[split] = {r["input"]["text"] for r in rows}
        assert Counter(r["input"]["target_lang"] for r in rows) == {"en": 500, "zh-CN": 500}
    training = read_jsonl("data/v1_label_inputs.jsonl")
    groups["train"] = {r["group_id"] for r in training}
    texts["train"] = {r["input"]["text"] for r in training}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not groups[a] & groups[b]
        assert not texts[a] & texts[b]
