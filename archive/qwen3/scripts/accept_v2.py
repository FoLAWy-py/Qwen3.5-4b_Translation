"""Persist Codex's completed review of all 244 observed API candidates."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record, check_constraints

FIXES = {
    "007-zh-CN": "‘改期会议’不自然，修订词序。",
    "013-en": "提议语气变成确定的未来安排。",
    "027-en": "would rather 后语法不自然。",
    "028-en": "not gave 语法错误。",
    "031-zh-CN": "platform 在此应为站台，且目标为简体中文。",
    "031-en": "无依据添加 bus，原文只说快车。",
    "037-zh-CN": "shuttle 在航站楼语境是接驳车，不是航天飞机。",
    "040-en": "坐轮椅的人能否通行，不应要求其 walk。",
    "043-en": "原文未明确 AM/PM，保留原时间表达。",
    "045-zh-CN": "after 路线顺序应译过了药店之后，不能仅说物理后面。",
    "051-en": "肉汁应为 gravy，避免 meat sauce 的菜品含义。",
    "053-en": "当前点餐安排被改成过去时。",
    "054-en": "把询问厨师能否不放油改成自己制作，角色改变。",
    "056-zh-CN": "serving spoon 应为公勺，区别个人餐勺。",
    "056-en": "public spoon 不是公勺的自然译法，且叉子复数数量含义丢失。",
    "058-zh-CN": "shellfish 涵盖甲壳类及贝类，原输出只保留甲壳类。",
    "061-en": "split the bill 可能表示平摊，原文是各自支付自己的餐费。",
    "062-zh-CN": "vegetables are fine 不必额外断定煮得刚刚好。",
    "083-zh-CN": "speed 应为速率，速度不变与加速度的表述应避免矢量含义冲突。",
    "094-en": "未检测到的过去结果改成一般现在时。",
    "105-en": "已完成的数据处理流程被改成一般现在时。",
    "107-zh-CN": "suggests 被强化为表明，宜保留提示的证据程度。",
    "114-zh-CN": "执行内嵌请求，漏掉外层原文。",
    "119-zh-CN": "代码赋值表达和反引号未完整保留。",
}


def main():
    if Path("data/prepared/v2/train.jsonl").exists():
        raise ValueError("最终训练数据已冻结，不能重复验收写入；新验收请使用新版本路径")
    files = {"data/generated/v2_candidates.jsonl": "e366f39708e89ad511bbe18fa136cc7afa62ae0f6e0b882d8b6465d128e88c2e",
             "data/generated/v2_sourcefix_candidates.jsonl": "c80974f8535f505662c86865177987e3b9403ace03263d9ccf939d8e30df4c9b"}
    candidates = []
    for path, digest in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("候选输出发生变化，必须重新逐条验收")
        candidates.extend(read_jsonl(path))
    lookup = {r["id"]: r for r in candidates}
    references = read_jsonl("data/v2_train_references_corrected.jsonl")
    accepted, decisions = [], []
    for ref in references:
        candidate = lookup[ref["id"]]
        if candidate["input"] != ref["input"]:
            raise ValueError("标注输入与验收参考不一致")
        key = ref["id"].removeprefix("v2-train-")
        issue = FIXES.get(key)
        if key.startswith("079-"):
            decisions.append({"id": ref["id"], "action": "reject", "reviewer": "Codex",
                              "content_hash": fingerprint({"input": candidate["input"], "output": candidate["output"]}),
                              "notes": "该原创置信水平表述过于刻意且不自然，移出训练而非把候选参考匹配当正确性证明。"})
            continue
        final = {**candidate, "output": ref["output"] if issue else candidate["output"]}
        reviewed(final, "Codex individually read API candidate against source; retain legitimate alternative wording")
        final["review"]["notes"] = issue or "含义、角色、语气、否定、关键数量和术语可接受；保留 API 等义措辞。"
        validate_record(final, True, True)
        decisions.append({"id": ref["id"], "action": "correct" if issue else "accept_teacher", "reviewer": "Codex",
                          "at": now(), "content_hash": fingerprint({"input": candidate["input"], "output": candidate["output"]}),
                          "final_hash": fingerprint({"input": final["input"], "output": final["output"]}),
                          "notes": final["review"]["notes"]})
        accepted.append(final)
    replay = read_jsonl("data/prepared/v2/replay.jsonl")
    train = accepted + replay
    random.Random(20260930).shuffle(train)
    write_jsonl("data/prepared/v2/train.jsonl", train)
    write_jsonl("data/generated/v2_decisions.jsonl", decisions)
    write_json("runs/v2-data-acceptance.json", {
        "at": now(), "reviewer": "Codex, user-authorized acceptance", "candidate_rows_read": len(candidates),
        "accepted_new": len(accepted), "replay_rows": len(replay), "training_rows": len(train),
        "training_groups": len({r["group_id"] for r in train}), "train_hash": fingerprint(train),
        "actions": dict(Counter(d["action"] for d in decisions)),
        "authoring_source_corrections": "runs/v2-source-corrections.json; original four bad-direction candidates excluded",
        "protected_warnings_to_review": {r["id"]: check_constraints(r["input"]["text"], r["output"]["translation"]) for r in train if check_constraints(r["input"]["text"], r["output"]["translation"])},
        "method": "All 244 API candidates read by Codex; 236 original plus four corrected-source candidates considered, two rejected, equal-meaning alternatives retained",
        "usage": {key: sum(r["label_metadata"]["usage"].get(key, 0) for r in candidates) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost")},
        "limitations": "388 rows is a focused optimization trial, not 388 independent semantic sources; 192 source groups including 75 replay groups"})
    print({"new": len(accepted), "train": len(train), "actions": dict(Counter(d["action"] for d in decisions))})


if __name__ == "__main__":
    main()
