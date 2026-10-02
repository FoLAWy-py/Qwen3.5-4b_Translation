"""Record Codex's source-grounded review of the 100 observed baseline outputs."""
from collections import Counter

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

ISSUES = {
    "v1-academic-012-13-zh-CN": ("major", "输出混入 The，中文句法不自然，目标语言未完整实现。"),
    "v1-daily-013-05-zh-CN": ("major", "‘我如果不介意’破坏主句和条件从句关系；未自然表达愿意等待。"),
    "v1-daily-013-13-zh-CN": ("major", "‘我如果不介意’破坏主句和条件从句关系；未自然表达愿意等待。"),
    "v1-daily-022-05-zh-CN": ("major", "available 指人有空，不是‘我无法使用’。"),
    "v1-daily-022-13-zh-CN": ("major", "available 指人有空，不是‘我无法使用’。"),
    "v1-food-005-05-zh-CN": ("major", "一壶热茶误译为一碗热茶，改变容器/分量。"),
    "v1-food-005-13-zh-CN": ("major", "一壶热茶误译为一杯热茶，改变容器/分量。"),
    "v1-daily-021-05-en": ("major", "原文‘星期五之前’译为 by Friday，截止边界扩大。"),
    "v1-daily-021-13-en": ("major", "原文‘星期五之前’译为 by Friday，截止边界扩大。"),
    "v1-daily-021-05-zh-CN": ("major", "by Friday 应表达最迟星期五，译为‘周五前’收窄截止边界。"),
    "v1-daily-021-13-zh-CN": ("major", "by Friday 应表达最迟星期五，译为‘周五前’收窄截止边界。"),
    "v1-daily-008-13-en": ("minor", "把商量语气译成确定的 we will，宜保留 let's 的提议语气。"),
    "v1-academic-042-05-zh-CN": ("minor", "‘应假设的10点’措辞生硬，宜直接表达先列出假设。"),
    "v1-academic-042-13-zh-CN": ("minor", "‘应满足’是原文未明确表达的关系，宜直接表达先列出假设。"),
    "v1-food-006-05-en": ("minor", "share them 的代词容易指盘子，宜明确分着吃食物。"),
    "v1-food-006-13-en": ("minor", "share them 的代词容易指盘子，宜明确分着吃食物。"),
    "v1-food-006-05-zh-CN": ("minor", "‘盘子用来分享’生硬，宜译为方便分着吃。"),
    "v1-food-006-13-zh-CN": ("minor", "‘盘子用于分享’生硬，宜译为方便分着吃。"),
    "v1-travel-008-05-en": ("minor", "疑问句英语结构不自然，宜用 Does the fare cover ...。"),
    "v1-travel-008-13-en": ("minor", "疑问句英语结构不自然，宜用 Does the fare cover ...。"),
}


def main():
    rows = read_jsonl("runs/v1-baseline-screen-corrected.jsonl")
    if fingerprint(rows) != "1f86f700457db7ba7529a049a54e80dc862cebb452b14d8ff088d84d0175b9e0":
        raise ValueError("评测输出有变更，必须重新逐条验收")
    decisions = []
    for row in rows:
        verdict, note = ISSUES.get(row["id"], ("pass", "已逐条对照当前原文，含义、方向和主要约束可接受；格式另行判定。"))
        decisions.append({"id": row["id"], "reviewer": "Codex", "at": now(),
                          "source_output_hash": fingerprint({"input": row["input"], "raw": row["raw"], "reference": row["reference"]}),
                          "semantic_verdict": verdict, "notes": note,
                          "format_valid": "prediction" in row})
    write_jsonl("runs/v1-baseline-semantic-decisions.jsonl", decisions)
    write_json("runs/v1-baseline-semantic-summary.json", {
        "reviewer": "Codex, user-authorized acceptance", "count": len(decisions),
        "evaluation_hash": fingerprint(rows), "verdicts": dict(Counter(d["semantic_verdict"] for d in decisions)),
        "scope": "100 controlled test rows from 25 held-out families; not independent long-tail accuracy",
        "format_valid": sum(d["format_valid"] for d in decisions)})


if __name__ == "__main__":
    main()
