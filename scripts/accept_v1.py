"""Accept against independent Codex-reviewed family references, not self-grading.

Any non-identical teacher wording is normalized to the reviewed reference. That
is a reference replacement, not a claim that every difference is a mistranslation.
Teacher candidates and input/output fingerprints remain available for audit.
"""
import json
from collections import Counter
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import check_constraints, validate_record

REFERENCES_HASH = "21f7e33c1b98ef19b57e48d86d22d27ef56b7d2aafdb2c1be29289dbbc2537c8"


def main():
    references = read_jsonl("data/v1_references.jsonl")
    if fingerprint(references) != REFERENCES_HASH:
        raise ValueError("已验收参考有变更，必须重新验收")
    destination = Path("data/prepared/v1")
    manifest = json.loads((destination / "split_manifest.json").read_text(encoding="utf-8"))
    families = read_jsonl("data/v1_family_acceptance.jsonl")
    if fingerprint(families) != manifest["family_hash"]:
        raise ValueError("句式验收记录有变更")
    train_groups = {f["group_id"] for f in families if f["split"] == "train"}
    expected = {r["id"]: r for r in references if r["group_id"] in train_groups}
    candidates = read_jsonl("data/generated/v1_candidates.jsonl")
    if len({r["id"] for r in candidates}) != len(candidates) or {r["id"] for r in candidates} != set(expected):
        raise ValueError("标注不完整或有重复/额外 id")
    accepted, decisions = [], []
    for row in candidates:
        validate_record(row, require_output=True)
        ref = expected[row["id"]]
        if row["input"] != ref["input"] or row["group_id"] != ref["group_id"]:
            raise ValueError("候选输入与已验收参考不一致")
        replacement = row["output"] != ref["output"] or row.get("label_metadata", {}).get("teacher_format_rejected", False)
        warnings = check_constraints(row["input"]["text"], row["output"]["translation"])
        final = {**row, "output": ref["output"], "review": {
            **ref["review"], "at": now(),
            "teacher_candidate_hash": fingerprint({"input": row["input"], "output": row["output"]}),
            "action": "replace_with_reviewed_reference" if replacement else "accept_exact_reference",
            "teacher_constraint_warnings": warnings}}
        if check_constraints(final["input"]["text"], final["output"]["translation"]):
            raise ValueError(f"参考保留项检查失败: {row['id']}")
        validate_record(final, require_output=True, require_review=True)
        accepted.append(final)
        decisions.append({"id": row["id"], **final["review"],
                          "teacher_translation": row["output"]["translation"],
                          "accepted_translation": final["output"]["translation"]})
    accepted.sort(key=lambda r: r["id"])
    import random
    random.Random(20260930).shuffle(accepted)
    if (destination / "train.jsonl").exists():
        raise ValueError("训练数据已冻结，不得覆盖")
    write_jsonl(destination / "train.jsonl", accepted)
    write_jsonl("data/generated/v1_decisions.jsonl", decisions)
    report = {"at": now(), "reviewer": "Codex, user-authorized quality acceptance",
              "total_accepted": len(accepted), "train_source_families": len(train_groups),
              "actions": dict(Counter(d['action'] for d in decisions)),
              "teacher_constraint_warning_count": sum(bool(d['teacher_constraint_warnings']) for d in decisions),
              "teacher_format_rejected_and_reference_corrected": sum(bool(r['label_metadata'].get('teacher_format_rejected')) for r in candidates),
              "method": "Independently reviewed bilingual families plus validated deterministic substitution; all final targets match accepted references",
              "not_an_error_rate": "Reference replacement includes equally valid alternative wording; count is not semantic error rate",
              "train_hash": fingerprint(accepted),
              "usage": {k: sum(r['label_metadata']['usage'].get(k, 0) for r in candidates)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost")}}
    write_json("runs/v1-data-acceptance.json", report)
    manifest["splits"]["train"]["sha256"] = fingerprint(accepted)
    manifest["training_labels_accepted_at"] = now()
    write_json(destination / "split_manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
