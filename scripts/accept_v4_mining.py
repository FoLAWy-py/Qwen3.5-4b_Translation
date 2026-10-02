"""Persist root's completed reading of all 48 real model generations."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = "245ef68ca4fb4f49480a1bb478415ce726dd732d0d6f3d5e4a40e21755bba83d"
ISSUES = {
    "002-zh-CN": ("major", "while paint dries晾干期间变成油漆干透时，开窗动作时间改变。"),
    "003-zh-CN": ("minor", "suitcase泛化为箱子；推着走符合wheel，未强制与参考拉着走一致。"),
    "003-en": ("major", "自己拉着走变成carry提着走，原文两种搬运行李方式的对比消失。"),
    "004-zh-CN": ("minor", "replacement无依据具体化为部件，且送到的状态没有明确保留。"),
    "004-en": ("minor", "替换品无依据具体化为replacement part。"),
    "007-zh-CN": ("minor", "checked through直挂泛化为托运到目的地，中转连续托运信息不明确。"),
    "008-zh-CN": ("major", "hostel旅舍误译为宿舍，旅行住宿设施类别改变。"),
    "009-zh-CN": ("major", "slotted spoon漏勺译成带槽勺子，未保留沥水孔结构；微沸也泛化成沸腾。"),
    "009-en": ("major", "漏勺变成colander滤篮，拿餐具请求变成使用滤篮，工具与动作改变。"),
    "010-zh-CN": ("minor", "破坏空气表达生硬，但轻拌保留空气的方向仍在；正确表达为避免消泡。"),
    "010-en": ("major", "避免消泡译成avoid foaming避免起泡，烹饪目的反转。"),
    "011-zh-CN": ("minor", "松散酱汁表述不自然，调稀意图不明确。"),
    "012-zh-CN": ("major", "anchovies凤尾鱼译成鱼子酱，食材类别改变。"),
    "017-zh-CN": ("major", "已给溪流侵蚀语境，bank仍译成银行而非河岸。"),
    "018-zh-CN": ("minor", "人员短缺语境下reinforcement只译加强，增援人手信息不明确。"),
    "019-zh-CN": ("major", "检查秤变检查量程；记录结果前变成称量前，对象及时间关系改变。"),
    "024-zh-CN": ("minor", "螺距太高搭配不自然，应为螺距太大；机械词义已识别。"),
}

def main():
    path = Path("runs/v4-mining-start.jsonl")
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError("Outputs changed; individual reading must be repeated")
    summary = json.loads(Path("runs/v4-mining-start.summary.json").read_text(encoding="utf-8"))
    if summary['adapter_sha256'] != "a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68" or summary['quantization'] != 'nf4':
        raise ValueError("Mining starting weights or quantization changed")
    destination = Path("data/prepared/v4-mining/preferences.jsonl")
    if destination.exists():
        raise ValueError("Accepted mining records already frozen")
    outputs = read_jsonl(path)
    refs = {r['id']: r for r in read_jsonl("data/prepared/v4-mining/pool.jsonl")}
    assert len(outputs) == len(refs) == 48
    assert {r['id'] for r in outputs} == set(refs)
    decisions, preferences = [], []
    for raw in outputs:
        ref = refs[raw['id']]
        assert raw['input'] == ref['input'] and raw['reference'] == ref['output']
        key = raw['id'].removeprefix("v4-mining-")
        verdict, note = ISSUES.get(key, ("pass", "逐条核对全文与语境，接受等义不同措辞；未发现需修订问题。"))
        decisions.append({"id": raw['id'], "reviewer": "Codex", "at": now(), "verdict": verdict, "note": note,
            "output_hash": fingerprint({k: raw[k] for k in ('input', 'raw', 'reference')}), "purpose": "training-only error mining, not release evaluation"})
        if verdict == 'major':
            row = {**ref, "rejected": raw['prediction'], "preference_issue": note,
                "negative_provenance": {"generation_sha256": RAW_SHA, "generation_id": raw['id'], "quantization": "nf4",
                    "starting_adapter_sha256": "a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68"}}
            reviewed(row, "Correct reference individually checked against actual model error; not a constructed role reversal")
            row['preference_review'] = {"reviewer": "Codex", "at": now(), "hash": fingerprint({k: row[k] for k in ('input', 'output', 'rejected', 'preference_issue')})}
            validate_record(row, True, True)
            preferences.append(row)
    assert set(ISSUES).issubset({r['id'].removeprefix('v4-mining-') for r in outputs})
    write_jsonl(destination, preferences)
    write_jsonl("runs/v4-mining-semantic.jsonl", decisions)
    report = {"at": now(), "reviewed_generations": 48, "verdicts": dict(Counter(r['verdict'] for r in decisions)),
        "real_error_preferences": len(preferences), "negative_source_groups": len({r['group_id'] for r in preferences}),
        "raw_sha256": RAW_SHA, "preferences_hash": fingerprint(preferences), "training_positive_rows": 48,
        "scope": "All inputs training-only; observed counts are mining yield, not unbiased accuracy or release results",
        "limitations": "Small original synthetic pool; natural model negatives may also differ in fluency; no CPO efficacy claim; no new adapter trained this round"}
    write_json("runs/v4-mining-acceptance.json", report)
    Path("runs/v4-mining-acceptance.md").write_text(
        "# v4真实错误挖掘\n\n48条新训练来源候选、20个来源组，NF4 v2起点生成，Codex逐条阅读全部输出。\n\n"
        f"挖掘结果：{report['verdicts']}。形成{len(preferences)}个真实严重错误偏好对；全部48条参考可作SFT正例。\n\n"
        "典型错误：拉行李变提行李、消泡变避免起泡、凤尾鱼变鱼子酱、溪流bank变银行。机械pitch与电荷charge语境则已正确识别，保留成功样本帮助控制回归。\n\n"
        "这批数据只用于训练池，不能作新模型测试。未用旧开发/测试原文构造新训练行，双向与同原文的不同语境同组。没有新增模型权重或质量提升结论。下一次训练需先另建独立开发来源并冻结配置；正式发布仍须600条新测试。\n\n"
        "验收标准见 [发布政策](../data/release_acceptance_policy.md)。真实负例也可能混有流畅度差异，后续须做负例选择消融。\n", encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
