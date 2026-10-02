"""Freeze a small CPO feasibility corpus before generation or training."""
from pathlib import Path
from collections import Counter

from data.v3_preferences import TRAIN, DEV, TEST
from data.v2_pairs import p
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    destination = Path("data/prepared/v3")
    if destination.exists():
        raise ValueError("v3 已冻结")
    old = []
    for path in Path("data/prepared").glob("**/*.jsonl"):
        old += [r for r in read_jsonl(path) if "input" in r]
    seen = {r["input"]["text"].strip().casefold() for r in old}
    splits = {}
    for name, pairs in (("train", TRAIN), ("dev", DEV), ("test", TEST)):
        rows = []
        for index, (category, en, zh, bad_zh, bad_en, issue, context) in enumerate(pairs, 1):
            p(category, en, zh, context)
            for lang, source, answer, wrong in (("zh-CN", en, zh, bad_zh), ("en", zh, en, bad_en)):
                key = source.strip().casefold()
                if key in seen:
                    raise ValueError(f"重复原文 {source}")
                seen.add(key)
                row = {"id": f"v3-{name}-{index:03}-{lang}", "group_id": f"v3-{name}-{index:03}", "category": category,
                    "input": {"text": source, "target_lang": lang, "context": context, "glossary": {}},
                    "output": {"translation": answer}, "rejected": {"translation": wrong}, "preference_issue": issue,
                    "source": {"name": "Original Codex-authored semantic contrast feasibility corpus", "license": "Original synthetic project data",
                        "training_allowed": name != "test", "external_labeling_allowed": True}}
                reviewed(row, "Bilingual source and deliberate semantic error individually checked before generation")
                validate_record(row, True, True, purpose="evaluation")
                rows.append(row)
        splits[name] = rows
        write_jsonl(destination / f"{name}_references.jsonl", rows)
    write_jsonl("data/v3_label_inputs.jsonl", [{k: v for k, v in r.items() if k not in ("output", "review", "rejected", "preference_issue")} for r in splits["train"]])
    write_json("runs/v3-plan.json", {"at": now(), "scope": "32 new training source pairs, not the planned full 3000–5000-row expansion",
        "splits": {name: {"rows": len(rows), "groups": len({r['group_id'] for r in rows}), "hash": fingerprint(rows),
            "categories": dict(Counter(r["category"] for r in rows))} for name, rows in splits.items()},
        "experiment": "Same v2 starting adapter, NF4, CPO versus paired positive-only SFT, 8 optimizer updates, accumulation8, LR1e-5, seed42, beta0.1; choose on new dev then single new test",
        "gate": "New selected candidate must not increase severe errors or reduce semantic passes versus starting adapter; formal requires zero major and >=95% passes",
        "limitations": "Small synthetic semantic contrast pilot; no claim of algorithmic novelty or broad translation quality"})
    print({k: len(v) for k,v in splits.items()})


if __name__ == "__main__":
    main()
