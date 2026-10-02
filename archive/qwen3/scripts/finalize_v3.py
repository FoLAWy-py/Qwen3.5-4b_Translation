"""Summarize the complete NF4 preference feasibility trial without promotion."""
import json
from pathlib import Path

from archive.qwen3.scripts.review_v3 import SHEETS, persist
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    reviews = {name: persist(name) for name in SHEETS}
    selection = load("runs/v3-selection.json")
    for name, candidate in selection["candidates"].items():
        directory = Path(candidate["directory"])
        config = load(directory / "run_config.json")
        assert fingerprint(read_jsonl(directory / "train_snapshot.jsonl")) == config["train_hash"]
        assert fingerprint(read_jsonl(directory / "dev_snapshot.jsonl")) == config["dev_hash"]
    start, selected = reviews["start-test"], reviews["selected-test"]
    verdicts = selected["verdicts"]
    formal = verdicts.get("major", 0) == 0 and verdicts.get("pass", 0) / selected["count"] >= .95
    relative = selected["format_valid"] / selected["count"] >= .95 and verdicts.get("major", 0) <= start["verdicts"].get("major", 0) and verdicts.get("pass", 0) >= start["verdicts"].get("pass", 0)
    record = {"at": now(), "status": "experimental_rejected" if not formal else "limited_frozen_test_passed", "default_promoted": False,
        "selection": selection, "reviews": reviews, "formal_gate": formal, "relative_gate": relative,
        "data": load("runs/v3-data-acceptance.json"), "performance": {name: load(f"runs/v3-{name}.summary.json")["buckets"]["all"] for name in SHEETS},
        "inference_memory": {name: {key: load(f"runs/v3-{name}.summary.json")[key] for key in ("peak_allocated_gib", "peak_reserved_gib")} for name in SHEETS},
        "inference_token_measurement": "Generated token IDs counted directly, JSON and EOS included; full generation elapsed time, not pure decode throughput",
        "conclusion": "CPO and SFT control did not improve new-development semantic counts versus starting adapter; no algorithmic improvement established",
        "limitations": ["60 training preference pairs/30 source groups, dev16/test16; synthetic short texts", "One seed and fixed8updates; not a full-sized replication of CPO",
            "SFT control uses two positive forwards per pair, equal updates but not strictly equal FLOPs", "Candidate preference accuracy is not greedy translation quality",
            "All frozen old/new test sources stay out of training; no broad translation acceptance"],
        "snapshot_integrity": "Both actual loaded snapshots verified against immutable run configuration"}
    write_json("runs/v3-model-acceptance.json", record)
    for candidate in selection["candidates"].values():
        path = Path(candidate["directory"]) / "witrans_adapter.json"
        metadata = load(path)
        metadata.update(quality_status="Experimental; no semantic improvement established", acceptance_report="runs/v3-model-acceptance.json")
        write_json(path, metadata)
    def counts(summary):
        return " / ".join(str(summary["verdicts"].get(k, 0)) for k in ("pass", "minor", "major"))
    lines = ["# v3 NF4偏好训练可行性试验", "", "已完成新样本验收、GPU反向检查、CPO/SFT对照、开发集译文审核与所选候选的冻结测试。当前没有建立语义改善证据，默认历史权重未替换。", "",
        "64条32B候选逐条阅读，修订10条、接受50条；4条因负例不自然而排除偏好训练。最终60个偏好对、30个训练来源组；开发/测试各16条、8个来源组。规模仅用于可行性试验，尚未完成3000–5000条扩充。", "",
        "两组均从同一个v2权重开始，NF4、LoRA r16、LR1e-5、beta0.1、seed42，8次更新、每次累积8对。CPO使用正确/错误答案，SFT对照每对使用正确答案两次。各访问64对，共128次前向，控制了更新次数和前向次数，但答案长度/反向开销不同，不称严格同FLOPs。", "",
        "| 项目 | CPO | SFT对照 |", "|---|---:|---:|"]
    c, s = selection["candidates"]["cpo"]["metrics"], selection["candidates"]["sft-control"]["metrics"]
    lines += [f"| 训练耗时（秒） | {c['train_seconds']:.1f} | {s['train_seconds']:.1f} |",
        f"| 峰值已分配显存（GiB） | {c['peak_allocated_gib']:.2f} | {s['peak_allocated_gib']:.2f} |",
        f"| 峰值保留显存（GiB） | {c['peak_reserved_gib']:.2f} | {s['peak_reserved_gib']:.2f} |",
        f"| 共同开发CPO目标 | {c['eval_cpo_loss']:.4f} | {s['eval_cpo_loss']:.4f} |", "",
        "按共同开发目标选出SFT对照，选择发生在测试生成之前。三组开发语义判定均为13通过、2轻微、1严重；JSON均16/16。蛋糕糖衣glaze仍译成陶瓷釉料，速率仍使用泛指rate，偏好训练未纠正这些错误。", "",
        "| 新冻结测试 | 通过 / 轻微 / 严重 | JSON合法 |", "|---|---:|---:|",
        f"| 起点v2 adapter | {counts(start)} | {start['format_valid']}/{start['count']} |",
        f"| 所选SFT对照 | {counts(selected)} | {selected['format_valid']}/{selected['count']} |", "",
        f"相对门槛：{'通过' if relative else '未通过'}；有限集合正式门槛（零严重且至少95%通过）：{'通过' if formal else '未通过'}。冻结测试不用于重新选择CPO或改训练目标。", "",
        "原始生成现在直接记录实际生成token数、输入token数及端到端输出token/s。计入JSON/EOS，不计模型加载与预热，不能当作纯解码速度。不同运行的轻微速度差异不构成稳定提速结论。", "",
        "本轮验证了8GB显卡可运行这条CPO链路，不能证明CPO普遍无效。数据少、负例明显、更新少、词义覆盖不足均是待验证因素，尚未做因果消融。下一步应在新的训练候选池挖掘模型真实错误，增加自然语义覆盖，并在固定数据预算下比较负例选择策略。", "",
        "- [机器可读验收与性能记录](v3-model-acceptance.json)", "- [数据逐条验收](../data/generated/v3_decisions.jsonl)", "- [测试前选择记录](v3-selection.json)", "- [所选候选测试判定](v3-selected-test-semantic.jsonl)", ""]
    perf = record["performance"]["selected-test"]
    mem = record["inference_memory"]["selected-test"]
    lines += [f"所选NF4候选本次16条短句测试：平均{perf['mean_seconds']:.2f}秒，{perf['output_tokens_per_second']:.2f}输出token/s；峰值已分配{mem['peak_allocated_gib']:.2f} GiB、保留{mem['peak_reserved_gib']:.2f} GiB。这些数值对应本集合与当前运行配置。", ""]
    Path("runs/v3-model-acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print({"status": record["status"], "relative_gate": relative, "formal_gate": formal, "default_promoted": False})


if __name__ == "__main__":
    main()
