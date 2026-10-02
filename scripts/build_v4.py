"""Freeze reviewed v4 mixed SFT corpus and separately authored development set."""
from collections import Counter
from pathlib import Path
from data.v4_dev_pairs import PAIRS
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

def main():
    destination = Path("data/prepared/v4")
    if destination.exists():
        raise ValueError("v4 already frozen")
    old = {r['input']['text'].strip().casefold() for p in Path("data/prepared").glob("**/*.jsonl") for r in read_jsonl(p) if 'input' in r}
    dev, seen = [], set()
    for i, (category, en, zh, context) in enumerate(PAIRS, 1):
        for target, text, answer in (("zh-CN", en, zh), ("en", zh, en)):
            key = text.strip().casefold()
            if key in old or key in seen:
                raise ValueError(f"Development source duplicates old source: {text}")
            seen.add(key)
            row = {"id": f"v4-dev-{i:03}-{target}", "group_id": f"v4-dev-{i:03}", "category": category,
                "input": {"text": text, "target_lang": target, "context": context, "glossary": {}}, "output": {"translation": answer},
                "source": {"name": "Original Codex-authored v4 development reference", "license": "Original synthetic project data",
                    "training_allowed": False, "external_labeling_allowed": False}}
            reviewed(row, "Individually checked bilingual development reference before any model outputs; not training permitted")
            validate_record(row, True, True, purpose="evaluation")
            dev.append(row)
    train = []
    for path in ("data/prepared/v2/train.jsonl", "data/prepared/v3/train.jsonl", "data/prepared/v4-mining/pool.jsonl"):
        for original in read_jsonl(path):
            row = {k: v for k, v in original.items() if k not in ('rejected', 'preference_review', 'preference_issue')}
            validate_record(row, True, True)
            train.append(row)
    if len({r['id'] for r in train}) != len(train):
        raise ValueError("Duplicate training IDs")
    if {r['input']['text'].strip().casefold() for r in train} & seen:
        raise ValueError("Training/development overlap")
    if {r['group_id'] for r in train} & {r['group_id'] for r in dev}:
        raise ValueError("Group leakage")
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "dev.jsonl", dev)
    manifest = {"at": now(), "train": {"rows": len(train), "groups": len({r['group_id'] for r in train}), "hash": fingerprint(train)},
        "dev": {"rows": len(dev), "groups": len({r['group_id'] for r in dev}), "hash": fingerprint(dev), "categories": dict(Counter(r['category'] for r in dev))},
        "policy": "New dev sources exact-deduplicated against all existing prepared files; both directions same group; not a release test; no full semantic near-duplicate guarantee",
        "configuration": {"method": "continue SFT from v2 NF4; mixed reviewed positive rows", "lr": 1e-5, "steps": 31, "accumulation": 16, "seed": 42,
            "selection": "Save at16 and31; lowest mean per-row answer NLL on newdev; inspect selected dev outputs; no release test until candidate passes screening"}}
    write_json(destination / "manifest.json", manifest)
    print(manifest)

if __name__ == "__main__":
    main()
