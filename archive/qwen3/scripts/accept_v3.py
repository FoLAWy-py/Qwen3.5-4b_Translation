"""Apply Codex's completed source-grounded review of 64 teacher candidates."""
from pathlib import Path
import hashlib
from collections import Counter

from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

FIXES = {
    "001-en": "未说明耳机所有权，去掉 her。",
    "005-zh-CN": "省略比较对象的照片容易理解为实物橱柜，明确两边都是照片。",
    "010-en": "接驳车未明确是bus，保留shuttle。",
    "011-zh-CN": "platform/concourse被误译为平台/站台层，恢复站台/大厅位置关系。",
    "011-en": "大厅未明确是atrium中庭，使用hall。",
    "013-zh-CN": "departure time保留出发时间，不改成起飞时间。",
    "014-en": "小壶应为small jug，不泛化为pot。",
    "015-zh-CN": "visible pieces未明确是贝壳碎片，去掉无依据限定。",
    "018-zh-CN": "crust未明确是酥皮，保留泛指外皮。",
    "021-en": "物理量质量是mass，不能用quality。",
}


def main():
    destination = Path("data/prepared/v3/train.jsonl")
    if destination.exists():
        raise ValueError("已验收训练数据禁止覆盖")
    if hashlib.sha256(Path("data/generated/v3_candidates.jsonl").read_bytes()).hexdigest() != "829fac3d6e7b3cdc9ff20857cde43859e1774db99c71a3e6a7761e34a08c6a09":
        raise ValueError("候选变化，必须重新阅读验收")
    candidates = read_jsonl("data/generated/v3_candidates.jsonl")
    refs = {r["id"]: r for r in read_jsonl("data/prepared/v3/train_references.jsonl")}
    assert len(candidates) == len(refs) == 64
    rows, decisions = [], []
    for row in candidates:
        ref = refs[row["id"]]
        assert row["input"] == ref["input"]
        key = row["id"].removeprefix("v3-train-")
        if key[:3] in ("031", "032"):
            decisions.append({"id": row["id"], "action": "exclude_preference_pair", "note": "拒绝负例：错误词义导致表述不自然，可能让模型只学流畅度；教师正例本身可接受，原记录保留。"})
            continue
        note = FIXES.get(key)
        if note:
            row["output"] = ref["output"] if key != "011-en" else {"translation": "The platform is above the hall, not below it."}
        row.update(rejected=ref["rejected"], preference_issue=ref["preference_issue"])
        reviewed(row, "Codex read every teacher output and retained equivalent variants; specific corrections recorded")
        row["preference_review"] = {"reviewer": "Codex", "at": now(), "hash": fingerprint({k: row[k] for k in ("input", "output", "rejected", "preference_issue")})}
        rows.append(row)
        decisions.append({"id": row["id"], "action": "correct" if note else "accept_teacher", "note": note or "逐条阅读并核对原文和语境，接受等义措辞。",
            "reviewed_hash": row["preference_review"]["hash"]})
    write_jsonl(destination, rows)
    for name in ("dev", "test"):
        heldout = read_jsonl(f"data/prepared/v3/{name}_references.jsonl")
        for row in heldout:
            row["preference_review"] = {"reviewer": "Codex", "at": now(), "hash": fingerprint({k: row[k] for k in ("input", "output", "rejected", "preference_issue")})}
        write_jsonl(f"data/prepared/v3/{name}.jsonl", heldout)
    write_jsonl("data/generated/v3_decisions.jsonl", decisions)
    write_json("runs/v3-data-acceptance.json", {"at": now(), "candidate_rows_read": 64, "training_pairs": len(rows), "groups": len(rows)//2,
        "actions": dict(Counter(d["action"] for d in decisions)), "train_hash": fingerprint(rows), "candidate_hash": fingerprint(candidates),
        "scope": "Small CPO feasibility trial; four pairs excluded for weak negatives, no test feedback used"})
    print({"training_pairs": len(rows), "actions": dict(Counter(d["action"] for d in decisions))})


if __name__ == "__main__":
    main()
