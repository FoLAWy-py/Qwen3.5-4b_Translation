"""Version objective bilingual deadline/arrival corrections without altering inputs.

These corrections follow source semantics, not desired model predictions. The
original frozen files remain available and no training examples are changed.
"""
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    edits = []
    for name in ("test", "test_screen"):
        rows = read_jsonl(f"data/prepared/v1/{name}.jsonl")
        output = Path(f"data/prepared/v1/{name}_corrected.jsonl")
        if output.exists():
            raise ValueError("修订参考已冻结")
        for row in rows:
            group = row["group_id"]
            target = row["input"]["target_lang"]
            old = row["output"]["translation"]
            new = old
            if group == "v1-daily-021":
                if target == "zh-CN":
                    new = old.replace("星期五之前", "最迟星期五")
                else:
                    new = old.replace("by Friday", "before Friday")
                reason = "by Friday 与 before Friday 的截止边界不同；按当前原文严格翻译"
            elif group == "v1-travel-001":
                if target == "zh-CN":
                    new = old.replace("入住前", "抵达前")
                else:
                    new = old.replace("before arrival", "before check-in")
                reason = "arrival 不应无条件收窄为 check-in；按当前原文区别抵达与入住"
            if new != old:
                edits.append({"set": name, "id": row["id"], "original": old, "corrected": new, "reason": reason})
                row["output"] = {"translation": new}
                row["review"]["reference_correction"] = reason
                row["review"]["content_hash"] = fingerprint({"input": row["input"], "output": row["output"]})
        write_jsonl(output, rows)
    write_json("runs/v1-reference-corrections.json", {
        "at": now(), "reviewer": "Codex", "training_data_changed": False,
        "test_inputs_changed": False, "original_frozen_files_preserved": True, "corrections": edits})
    # Preserve the original report and derive a report with corrected references.
    rows = read_jsonl("runs/v1-baseline-screen.jsonl")
    refs = {r["id"]: r["output"] for r in read_jsonl("data/prepared/v1/test_screen_corrected.jsonl")}
    for row in rows:
        row["reference"] = refs[row["id"]]
    write_jsonl("runs/v1-baseline-screen-corrected.jsonl", rows)
    print(f"参考修订 {len(edits)} 条记录（完整测试与筛查存在重叠），不改输入与训练数据")


if __name__ == "__main__":
    main()
