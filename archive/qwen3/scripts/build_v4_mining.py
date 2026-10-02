"""Freeze new training-only mining inputs without changing old dev/test sources."""
from pathlib import Path
from collections import Counter
from data.v4_mining_pairs import PAIRS
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

def main():
    destination = Path("data/prepared/v4-mining")
    if destination.exists():
        raise ValueError("Mining pool already frozen")
    old = {r["input"]["text"].strip().casefold() for path in Path("data/prepared").glob("**/*.jsonl") for r in read_jsonl(path) if "input" in r}
    rows = []
    for index, (group, category, en, zh, context) in enumerate(PAIRS, 1):
        for target, text, answer in (("zh-CN", en, zh), ("en", zh, en)):
            if text.strip().casefold() in old:
                raise ValueError(f"Old source duplicated: {text}")
            row = {"id": f"v4-mining-{index:03}-{target}", "group_id": f"v4-mining-{group}", "category": category,
                "input": {"text": text, "target_lang": target, "context": context, "glossary": {}}, "output": {"translation": answer},
                "source": {"name": "Codex-authored training-only real-error mining pool", "license": "Original synthetic project data",
                    "training_allowed": True, "external_labeling_allowed": True}}
            reviewed(row, "Individually composed and checked bilingual reference; context-family kept together; not teacher output")
            validate_record(row, True, True)
            rows.append(row)
    write_jsonl(destination / "pool.jsonl", rows)
    write_json(destination / "manifest.json", {"at": now(), "rows": len(rows), "groups": len({r['group_id'] for r in rows}),
        "categories": dict(Counter(r['category'] for r in rows)), "data_hash": fingerprint(rows),
        "scope": "Training-only mining; exact old-source exclusion, not semantic near-duplicate guarantee; no release evaluation",
        "starting_adapter": "models/witrans-4b-v2-selected/adapter", "quantization": "nf4",
        "selection": "Codex must individually inspect real outputs; only genuine semantic errors become rejected translations"})
    print({"rows": len(rows), "groups": len({r['group_id'] for r in rows})})

if __name__ == "__main__":
    main()
