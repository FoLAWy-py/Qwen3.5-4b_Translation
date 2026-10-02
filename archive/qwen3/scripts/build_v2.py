"""Freeze diverse v2 pairs and a deduplicated subset of v1 training only."""
import json
import random
from collections import Counter
from pathlib import Path

from data.v2_pairs import TRAIN, DEV, TEST
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def reviewed(row, method):
    row["review"] = {"status": "approved", "reviewer": "Codex (user-authorized AI acceptance)",
                     "method": method, "at": now(), "content_hash": fingerprint({"input": row["input"], "output": row["output"]})}
    return row


def main():
    destination = Path("data/prepared/v2")
    if destination.exists():
        raise ValueError("v2 已冻结，不能覆盖")
    splits = {}
    for split, pairs in (("train", TRAIN), ("dev", DEV), ("test", TEST)):
        rows = []
        for index, pair in enumerate(pairs, 1):
            # Identical source with different contexts stays in one family.
            group = "v2-" + fingerprint(pair["english"])[:16]
            for target, source, translation in (("zh-CN", pair["english"], pair["chinese"]), ("en", pair["chinese"], pair["english"])):
                terms = pair["terms"] if target == "zh-CN" else {v: k for k, v in pair["terms"].items()}
                row = {"id": f"v2-{split}-{index:03}-{target}", "group_id": group, "category": pair["category"],
                       "input": {"text": source, "target_lang": target, "context": pair["context"], "glossary": terms},
                       "output": {"translation": translation},
                       "source": {"name": "Codex-authored diverse bilingual v2", "license": "Original synthetic project data",
                                  "training_allowed": True, "external_labeling_allowed": True,
                                  "construction": "Individually authored semantic pair; reverse pair remains in same group"}}
                rows.append(reviewed(row, "Individually authored and source-grounded bilingual pair acceptance"))
        splits[split] = rows
    replay = []
    for row in read_jsonl("data/prepared/v1/train.jsonl"):
        if "-02-" in row["id"]:
            row["id"] = "v2-replay-" + row["id"]
            row["input"]["context"] = ""
            row["source"]["construction"] = "One bilingual pair per v1 training family; generic context removed, original text and answer retained"
            replay.append(reviewed(row, "Previously accepted v1 training pair; no v1 dev/test replay"))
    if len(replay) != 150:
        raise ValueError("旧数据训练来源组数量变化")
    old_heldout = read_jsonl("data/prepared/v1/dev.jsonl") + read_jsonl("data/prepared/v1/test.jsonl")
    old_texts = {r["input"]["text"].strip().casefold() for r in old_heldout}
    for rows in splits.values():
        for row in rows:
            validate_record(row, True, True)
            if row["input"]["text"].strip().casefold() in old_texts:
                raise ValueError("新数据重复旧开发/测试原文")
    families = {name: {r["group_id"] for r in rows} for name, rows in splits.items()}
    texts = {name: {r["input"]["text"].strip().casefold() for r in rows} for name, rows in splits.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if families[a] & families[b] or texts[a] & texts[b]:
            raise ValueError("新数据来源或原文泄漏")
    write_jsonl("data/v2_train_references.jsonl", splits["train"])
    write_jsonl("data/v2_label_inputs.jsonl", [{k: v for k, v in r.items() if k not in ("output", "review")} for r in splits["train"]])
    write_jsonl(destination / "replay.jsonl", replay)
    for name in ("dev", "test"):
        write_jsonl(destination / f"{name}.jsonl", splits[name])
    # Short development generations include both directions and all categories.
    write_jsonl(destination / "dev_screen.jsonl", splits["dev"])
    manifest = {"at": now(), "new_pair_counts": {"train": len(TRAIN), "dev": len(DEV), "test": len(TEST)},
                "replay_rows": len(replay), "planned_train_rows": len(splits["train"]) + len(replay),
                "source_hash": fingerprint({"train": TRAIN, "dev": DEV, "test": TEST}),
                "splits": {name: {"count": len(rows), "groups": len(families[name]), "hash": fingerprint(rows),
                                  "categories": dict(Counter(r["category"] for r in rows)),
                                  "context_rows": sum(bool(r["input"]["context"]) for r in rows)} for name, rows in splits.items()},
                "experiment": {"base": "Original Qwen3-4B; fresh LoRA", "learning_rate": 3e-5, "lora_dropout": 0.05,
                               "epochs": 1, "rank": 16, "max_length": 1024, "eval_steps": 8,
                               "selection": "Minimum new-development loss; examine dev generations before independent test",
                               "test_use": "Single acceptance after model choice; never train or choose checkpoints with test"},
                "limitations": "Focused small optimization trial, 120 new training pairs plus 75 old train pairs; not a 3000-row independent corpus"}
    write_json(destination / "split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
