from __future__ import annotations

from pathlib import Path

from .common import fingerprint, now, read_jsonl, write_jsonl
from .data import validate_record


def review(args):
    """Apply explicit reviewer decisions; never auto-approve teacher output."""
    rows = read_jsonl(args.input)
    decisions = read_jsonl(args.decisions)
    ids = {r["id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("候选样本 id 重复")
    by_id = {d["id"]: d for d in decisions}
    if len(by_id) != len(decisions) or set(by_id) - ids:
        raise ValueError("审核 id 重复或不存在")
    approved = []
    for row in rows:
        decision = by_id.get(row["id"])
        if decision is None or decision.get("status") != "approved":
            continue
        if not decision.get("reviewer") or not decision.get("notes"):
            raise ValueError("审核记录需有 reviewer 与 notes")
        content_hash = fingerprint({"input": row["input"], "output": row["output"]})
        if decision.get("content_hash") != content_hash:
            raise ValueError("审核内容指纹不一致；重新审阅当前版本")
        if "translation" in decision:
            row["output"] = {"translation": decision["translation"]}
        row["review"] = {**decision, "at": now(), "content_hash": fingerprint({"input": row["input"], "output": row["output"]})}
        validate_record(row, require_output=True, require_review=True)
        approved.append(row)
    if Path(args.output).exists():
        raise ValueError("审核输出已存在；使用新路径")
    write_jsonl(args.output, approved)
    print(f"保存 {len(approved)} 条已审核样本")
