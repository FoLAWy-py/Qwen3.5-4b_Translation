"""Publish source-bound experimental acceptance without changing default weights."""
import hashlib
import json
from pathlib import Path

from archive.qwen3.scripts.review_v2 import SHEETS, persist
from witrans_tools.common import now, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    names = list(SHEETS)
    summaries = {name: persist(name) for name in names}
    comparisons = {}
    for precision, baseline, adapter in (("nf4", "baseline-test", "adapter-test"),
        ("int8", "int8-baseline-precision", "int8-adapter-precision"),
        ("bf16_cpu_vocab", "bf16-baseline-test", "bf16-adapter-test")):
        b, a = summaries[baseline], summaries[adapter]
        assert b["count"] == a["count"] == 40
        bv, av = b["verdicts"], a["verdicts"]
        comparisons[precision] = {"baseline": b, "adapter": a,
            "relative_gate": a["format_valid"] / a["count"] >= 0.95 and av.get("major", 0) <= bv.get("major", 0) and av.get("pass", 0) >= bv.get("pass", 0),
            "formal_gate": av.get("major", 0) == 0 and av.get("pass", 0) / a["count"] >= 0.95}
    selected = load("runs/v2-bf16-selection.json")["selected"]
    verdict = comparisons[selected]
    adapter_path = Path("models/witrans-4b-v2-selected/adapter")
    with (adapter_path / "adapter_model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == load("runs/v2-checkpoint-selection.json")["adapter_sha256"]
    budget = load("runs/v2-bf16-budget.json")
    receipt = {"at": now(), "reviewer": "Codex, user-authorized sample acceptance", "status": "passed" if verdict["formal_gate"] else "experimental_rejected",
        "default_promoted": False, "selected_inference": selected, "adapter_sha256": digest,
        "reason": "Selected configuration retains semantic errors; preserve evidence and historical default, do not reselect configuration using test outcomes",
        "comparisons": comparisons, "development": {k: v for k, v in summaries.items() if k.endswith("dev")},
        "data_acceptance": load("runs/v2-data-acceptance.json"), "checkpoint_selection": load("runs/v2-checkpoint-selection.json"),
        "inference_selection": load("runs/v2-bf16-selection.json"), "budget_check": budget,
        "limitations": ["Small independently authored synthetic cases, no professional human translation review", "Short texts/paragraphs, no long-document quality acceptance",
            "Training 388 rows and 192 groups, not 3000 independent new semantic sources", "INT8 baseline timing includes interruption long tails and is excluded from speed comparison"],
        "raw_output_sha256": {name: SHEETS[name][1] for name in names}}
    write_json("runs/v2-model-acceptance.json", receipt)
    metadata = load(adapter_path / "witrans_adapter.json")
    metadata.update(quality_status="Experimental; semantic acceptance rejected" if not verdict["formal_gate"] else "Passed limited frozen acceptance",
        recommended_quantization=selected, acceptance_report="runs/v2-model-acceptance.json")
    write_json(adapter_path / "witrans_adapter.json", metadata)
    lines = ["# v2 模型验收", "", "v2 已完成训练、权重选择、三种精度的开发比较和独立测试。正式语义验收未通过，候选保留用于实验，默认历史权重没有替换。", "",
        "每组测试均为 40 条，中英各 20 条；同一行的基础模型与 adapter 使用同输入、精度、提示词和贪心解码。各精度使用不同冻结测试，不能横向把通过率解释为精度收益。", "",
        "| 精度 / 独立集合 | 基础模型：通过 / 轻微 / 严重 | adapter：通过 / 轻微 / 严重 | JSON：基础 / adapter | 相对晋级门槛 |", "|---|---:|---:|---:|---|" ]
    def counts(summary):
        return " / ".join(str(summary["verdicts"].get(k, 0)) for k in ("pass", "minor", "major"))
    for precision, comparison in comparisons.items():
        b, a = comparison["baseline"], comparison["adapter"]
        lines.append(f"| {precision} | {counts(b)} | {counts(a)} | {b['format_valid']} / {a['format_valid']} | {'通过' if comparison['relative_gate'] else '未通过'} |")
    lines += ["", "相对晋级只要求不逊于基础模型；正式通过还要求零严重错误且至少 95% 通过。所有已测配置均未达到正式要求。NF4 相对门槛通过，不据此在看到测试后重新选择配置。", "",
        "训练使用 238 条新验收样本和 150 条旧训练重放，学习率 3e-5、dropout 0.05、rank16、长度1024；25 步约7.2分钟。按新开发集最低 loss 选择最终第25步权重（0.87276），选择发生在测试前。", "",
        "Codex 阅读 244 条 API 候选，214 条原样接受、24 条具体修订、2 条拒绝；另四条原始错误语言方向被排除并重新标注。正式训练保存实际加载数据快照，指纹复核一致。两个中止训练以及原始数据仍保存，更正记录见 [来源更正](v2-source-corrections.json) 和 [数据核验](v2-data-verification.json)。", "",
        "在同一个开发集上，INT8 与 BF16 都是30条通过、5条轻微、5条严重；平均耗时6.40与4.87秒，约减少24%。BF16共享词表在CPU执行，其余层在GPU，开发实测保留显存7.05 GiB。此速度对比只说明该环境的短句完整生成耗时，不能当作普遍性能结论。", "",
        f"1024-token预算压力测试：实际提示词{budget['prompt_tokens']} tokens，生成预算{budget['max_new_tokens']}，状态 {budget['status']}，峰值保留 {budget['peak_reserved_gib']:.2f} GiB。重复背景只测试显存和设备放置，不能证明长文本翻译质量或更大预算。", "",
        "BF16 独立测试中的实质错误包括未锁门误译为门开着、周一之前改成不晚于周一、餐盘改为碗，以及明确修理费用语境下把 charge 译成刑事指控。JSON 合法不代表语义正确。详细判定逐条绑定输入、原始输出和参考指纹，允许等义不同表达。", "",
        "- [全部原始生成指纹与结果](v2-model-acceptance.json)", "- [BF16 基础模型逐条判定](v2-bf16-baseline-test-semantic.jsonl)", "- [BF16 adapter 逐条判定](v2-bf16-adapter-test-semantic.jsonl)", "- [开发集配置选择](v2-bf16-selection.json)", "- [预算压力记录](v2-bf16-budget.json)", "",
        "后续应扩充独立语义、自然会话、上下文消歧、容器和条件关系，再从原始基础模型训练；本轮所有冻结验收集保持测试专用。当前样本量、来源多样性和长文覆盖不足，不能以增加 epoch 或复训测试样本替代。", ""]
    Path("runs/v2-model-acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "default_promoted": False, "selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
