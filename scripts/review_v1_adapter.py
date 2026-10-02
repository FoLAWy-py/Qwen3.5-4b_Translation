"""Persist Codex's completed, source-grounded review; this is not an AI grader."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_v1_baseline import ISSUES as BASELINE_ISSUES
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


ISSUES = {}


def flag(ids, verdict, note):
    for identifier in ids.split():
        ISSUES[identifier] = (verdict, note)


flag("v1-academic-005-05-zh-CN v1-academic-005-13-zh-CN", "minor", "‘行缺失值’表达生硬，应为‘行有缺失值的数据’，主要含义可理解。")
flag("v1-academic-017-05-en", "minor", "before recalibrating 容易把传感器写成校准动作的执行者，宜用 before recalibration。")
flag("v1-daily-013-05-zh-CN v1-daily-013-13-zh-CN", "major", "finished 未翻译，中文输出混入英文；‘等另外’也不自然。")
flag("v1-daily-015-05-zh-CN v1-daily-015-13-zh-CN", "major", "不需要更多桌子变成不需要再搬桌子，改变需求对象与动作。")
flag("v1-daily-021-05-en v1-daily-021-13-en", "major", "星期五之前译成 by Friday，扩展截止边界。")
flag("v1-daily-021-05-zh-CN v1-daily-021-13-zh-CN", "major", "by Friday 应保留‘最迟星期五’边界，‘到周五前’收窄边界。")
flag("v1-daily-022-05-zh-CN v1-daily-022-13-zh-CN", "major", "人的 available 指可安排/有空，‘不在’添加离开或缺席的事实。")
flag("v1-food-002-05-en", "major", "餐食只要求 peanut-free，原文明确不含花生油的要求被移到过敏原因中，不能保证要求完整。")
flag("v1-food-005-05-zh-CN v1-food-005-13-zh-CN", "major", "一壶热茶译成一碗热茶，改变容器与分量；米饭的碗也泛化为份。")
flag("v1-food-006-05-en v1-food-006-13-en", "minor", "share them 易指分享盘子，宜明确分着吃食物。")
flag("v1-food-006-05-zh-CN", "minor", "‘盘子分着用’表达分享餐具，宜表达方便分着吃。")
flag("v1-food-006-13-zh-CN", "major", "漏掉 extra 和 for sharing，变成简单分发盘子。")
flag("v1-food-013-05-zh-CN v1-food-013-13-zh-CN", "major", "‘没有含坚果的吗？’可读成询问有没有含坚果的甜点，未清楚保留全部不含坚果的确认要求。")
flag("v1-food-013-13-en", "minor", "Could we confirm 把请求对方确认改成我们确认，语用视角有变化。")
flag("v1-hard-012-05-en", "minor", "for the change 易指零钱或一般变更，replacement 更明确。")
flag("v1-hard-012-13-en", "minor", "漏去明确的‘我支付’及过去时，币种和更换用途仍可理解；英语单复数不自然。")
flag("v1-travel-001-05-zh-CN v1-travel-001-13-zh-CN", "major", "arrival 无上下文限定为 check-in；擅自改为入住，参考已独立纠正为抵达。")
flag("v1-travel-008-05-en v1-travel-008-13-en", "major", "原文只说车费，擅自增加 train，指定了未知的交通工具。")
flag("v1-travel-008-05-zh-CN", "major", "总费用/每人费用改成票的使用人数/每人一张，改变计费问题。")
flag("v1-travel-008-13-zh-CN", "minor", "fare 泛化为车票，宜明确票价；按每人计算的主要问题仍保留。")
flag("v1-travel-017-05-zh-CN", "major", "续住10晚变成住满10晚后再续住，10晚的作用被改变。")
flag("v1-challenge-009", "minor", "保留单独容器，但 bring 的拿过来动作被省略。")
flag("v1-challenge-019", "major", "只翻译了内嵌消息，还添加‘请’，漏译外层请求，违反 text 全文作为数据翻译的约定。")
flag("v1-challenge-020", "major", "只输出押金问句，漏掉‘只需要翻译…不要回答…’这段原文，执行了文本中的指令。")

BASELINE_ADDITIONS = {
    "v1-academic-005-05-zh-CN": ("minor", "‘行缺失值’表达生硬，应为‘行有缺失值的数据’，主要含义可理解。"),
    "v1-academic-005-13-zh-CN": ("minor", "‘行缺失值’表达生硬，应为‘行有缺失值的数据’，主要含义可理解。"),
    "v1-academic-022-13-en": ("minor", "原文描述已发生的试验，译文改为一般现在时。"),
    "v1-travel-001-05-zh-CN": ("minor", "‘押金在…取消’语法生硬，应以取消预订为条件；抵达时间仍保留。"),
}


def review(rows, issues):
    if not set(issues).issubset({r["id"] for r in rows}):
        raise ValueError("判定与实际样本 id 不一致")
    decisions = []
    for row in rows:
        verdict, note = issues.get(row["id"], ("pass", "已逐条对照原文与可选上下文/术语；含义、人称、否定、数量及关键约束可接受，允许等义措辞。"))
        decisions.append({"id": row["id"], "reviewer": "Codex", "at": now(),
                          "source_output_hash": fingerprint({"input": row["input"], "raw": row["raw"], "reference": row["reference"]}),
                          "semantic_verdict": verdict, "notes": note,
                          "format_valid": "prediction" in row})
    return decisions


def counts(decisions):
    return {"count": len(decisions), "verdicts": dict(Counter(d["semantic_verdict"] for d in decisions)),
            "format_valid": sum(d["format_valid"] for d in decisions)}


def main():
    path = Path("runs/v1-adapter-acceptance.jsonl")
    if hashlib.sha256(path.read_bytes()).hexdigest() != "58c36c1393fa6e1fcdc3a6d4308eb78f460ce6436ad1c42f90468df2124efc2e":
        raise ValueError("输出已变更，必须重新逐条验收")
    rows = read_jsonl(path)
    baseline = read_jsonl("runs/v1-baseline-screen-corrected.jsonl")
    if fingerprint(baseline) != "1f86f700457db7ba7529a049a54e80dc862cebb452b14d8ff088d84d0175b9e0":
        raise ValueError("基线输出已变更")
    decisions = review(rows, ISSUES)
    base_decisions = review(baseline, {**BASELINE_ISSUES, **BASELINE_ADDITIONS})
    shared = {r["id"] for r in baseline}
    screen = [d for d in decisions if d["id"] in shared]
    challenges = [d for d in decisions if d["id"] not in shared]
    if len(screen) != 100 or len(challenges) != 20:
        raise ValueError("冻结测试范围不一致")
    if any(a["id"] != b["id"] for a, b in zip(screen, base_decisions, strict=True)):
        raise ValueError("对照顺序不一致")
    transitions = Counter(f"{b['semantic_verdict']} -> {a['semantic_verdict']}" for a, b in zip(screen, base_decisions, strict=True))
    write_jsonl("runs/v1-adapter-semantic-decisions.jsonl", decisions)
    write_jsonl("runs/v1-baseline-semantic-v2.jsonl", base_decisions)
    report = {
        "at": now(), "reviewer": "Codex, user-authorized quality acceptance",
        "verdict": "REJECTED for semantic quality; usable experimental adapter only",
        "rubric": {"major": "漏译、错译、无依据添加、关键边界变化或目标语言未完整实现",
                   "minor": "主要意思可理解但措辞、语用视角或非关键细节需修订",
                   "pass": "当前样本的含义与关键约束可接受；不要求参考完全匹配"},
        "evaluation_hash": fingerprint(rows), "baseline_hash": fingerprint(baseline),
        "all": counts(decisions), "controlled_screen": counts(screen), "independent_challenges": counts(challenges),
        "baseline_shared_screen": counts(base_decisions), "paired_transitions": dict(transitions),
        "baseline_review_revision": "Original decisions retained; four minor issues added after rereading all 100 outputs under the same rubric",
        "selection": json.loads(Path("runs/v1-checkpoint-selection.json").read_text(encoding="utf-8")),
        "scope": "100 rows / 25 held-out controlled families plus 20 independent short challenges; not full 1000-row test generation or broad translation accuracy",
        "limitations": ["Training consists of only 75 independent controlled families with numeric variants",
                        "Codex acceptance is independent of the cloud teacher, but not a professional human audit",
                        "Screen covers glossary/no-glossary variants; separate context disambiguation and long texts are not adequately tested",
                        "Baseline and adapter used different process-local CPU profiles; do not compare latency as a controlled benchmark",
                        "No training targets or checkpoint selection changed based on test failures"],
        "example_call_review": [
            {"source": "Could you put the sauce on the side?", "prediction": "能把沙拉放在旁边吗？", "verdict": "major", "note": "sauce 变成沙拉；不能认为酱汁需求已通过"},
            {"source": "Is the deposit refundable?", "prediction": "押金可以退吗？", "verdict": "pass"},
            {"source": "Let x be a positive real number.", "prediction": "设 x 为正实数。", "verdict": "pass"},
            {"source": "这道菜可以少放一点盐吗？", "prediction": "Could we use less salt in this dish?", "verdict": "pass"}],
        "next_iteration": "新增有独立语义的原创新数据，保留完整原文/约束/角色/语气；以新的开发集筛选，另建新冻结测试。不得针对本次测试改训练目标或反复挑 checkpoint。",
    }
    write_json("runs/v1-model-acceptance.json", report)
    metadata_path = Path("models/witrans-4b/adapter/witrans_adapter.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(quality_status="Experimental only; rejected by Codex semantic acceptance",
                    acceptance_report="runs/v1-model-acceptance.json")
    write_json(metadata_path, metadata)
    diagnostics_path = Path("runs/v1-runtime-diagnostics.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["stability"] = "Completed 3000-row / 188-step / 1-epoch training, two full 1000-row dev evaluations, 120-row generation and four local example calls under CPU isolation; root cause unconfirmed"
    write_json(diagnostics_path, diagnostics)
    print(json.dumps({k: report[k] for k in ("verdict", "all", "controlled_screen", "independent_challenges", "baseline_shared_screen", "paired_transitions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
