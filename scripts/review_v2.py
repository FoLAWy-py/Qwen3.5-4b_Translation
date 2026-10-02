"""Source-grounded Codex review sheets for already inspected v2 outputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

BASE_DEV = {
    "001-zh-CN": ("major", "中文混入 please/anything，目标语言没有完整实现。"),
    "002-en": ("minor", "The light is good 不如灯能正常工作的表达明确自然。"),
    "003-zh-CN": ("major", "把明天退回改成明天如果不取才退回，改变条件时间的作用。"),
    "004-zh-CN": ("minor", "tools 泛化为东西，具体对象丢失。"),
    "006-en": ("major", "这里关门后锁起来变成自行车被锁起来，it 指代产生对象变化。"),
    "007-zh-CN": ("major", "只能上车变成允许上下车，改变限制。"),
    "008-zh-CN": ("major", "未翻译，整段保留英文。"),
    "008-en": ("minor", "姐姐的 older 信息省略，主要预订关系保留。"),
    "009-zh-CN": ("major", "mine 的所属关系丢失，并添加这位女士的称呼。"),
    "010-en": ("minor", "marinade 泛化为 condiment，腌料类别不够明确。"),
    "011-zh-CN": ("major", "carafe 和 glass 都译成杯，容器对比丢失。"),
    "014-en": ("minor", "总体方差应明确为 population variance，overall variance 术语不精确。"),
}

ADAPTER_DEV = {key: value for key, value in BASE_DEV.items() if key not in ("001-zh-CN", "008-zh-CN")}
ADAPTER_DEV["002-zh-CN"] = ("major", "lamp works 指功能正常，译成灯开着，添加当前开关状态。")
ADAPTER_DEV["009-zh-CN"] = ("major", "未明确 mine 的所属关系，‘保留她的’还可指奶酪以外的对象。")

BASE_TEST = {
    "004-zh-CN": ("minor", "‘发给你的和’句法生硬，宜删除多余的‘的’。"),
    "004-en": ("minor", "同行者宜用 companion/accompanying guest，fellow traveler 增加旅行含义。"),
    "006-zh-CN": ("major", "无依据指定登机/机票，并把 where 询问地点改成 how 询问方法。"),
    "007-zh-CN": ("major", "整句未翻译，目标语言未实现。"),
    "009-zh-CN": ("minor", "wedges 是角形块，片不够准确，但主要分开放置要求保留。"),
    "009-en": ("major", "柠檬角译成 lemon cups，食物对象改变。"),
    "011-zh-CN": ("major", "per glass 的单独收费变成按每人收费，计费范围改变。"),
    "017-zh-CN": ("major", "叙述性 account 译成财务账目，语境词义错误。"),
    "019-zh-CN": ("major", "明确鸟类上下文下 crane 仍译为起重机。"),
    "020-zh-CN": ("minor", "Say nothing 宜为‘什么也别说’，‘不要说’作为引文表达不完整自然。"),
}
ADAPTER_TEST = {key: value for key, value in BASE_TEST.items() if key != "017-zh-CN"}
ADAPTER_TEST["002-en"] = ("major", "‘我问他’变成 He asked，提问者身份改变。")
ADAPTER_TEST["006-zh-CN"] = ("major", "原文未限定交通工具，擅自增加登机。")
ADAPTER_TEST["007-zh-CN"] = ("major", "桥关闭维修的关闭限制被省略，只说正在维修。")
INT8_DEV = {key: value for key, value in ADAPTER_DEV.items() if key not in ("009-zh-CN", "014-en")}
INT8_DEV["009-en"] = ("minor", "点餐请求变成 Can I put，容易理解为自己操作，语用角色不够自然。")

PRECISION_BASE = {
    "001-zh-CN": ("minor", "等她和经理说话的句法生硬，并行时间关系不够自然明确。"),
    "005-en": ("minor", "不用着急在此指不必赶时间，worry 弱化了时间紧迫含义。"),
    "006-zh-CN": ("minor", "条件从句置于末尾的中文表达生硬。"),
    "008-zh-CN": ("major", "final flight 是最后一段楼梯，译成最后几层楼，增加未给出的楼层数量。"),
    "009-zh-CN": ("major", "拿容器的请求丢失，并把 cup 改成茶，容器与内容混淆。"),
    "009-en": ("major", "茶碟译成 tea tray，改变容器类型。"),
    "010-en": ("minor", "酥皮泛化为 outer crust，糕点外皮类型不明确。"),
    "011-en": ("minor", "那罐未开封的泛化为 unopened one，罐子对象没有译出。"),
    "016-en": ("major", "储存事实改成 should be stored 的操作要求，改变语气和时间。"),
    "017-en": ("major", "政策不能解释此前回答改成回答无法被解释，因果主语及解释范围改变。"),
}
PRECISION_ADAPTER = {k: v for k, v in PRECISION_BASE.items() if k not in ("001-zh-CN", "017-en")}
PRECISION_ADAPTER.update({
    "002-zh-CN": ("minor", "需要它回来句法生硬，今晚还伞的含义仍可理解。"),
    "017-en": ("minor", "收集回答的调查宜用 survey，investigation 的调查类型不够准确自然。"),
    "018-zh-CN": ("major", "按年报价是利率的表达口径，按年计息增加原文没有的计息方式。"),
})
BF16_DEV = {k: v for k, v in INT8_DEV.items() if k != "010-en"}
BF16_DEV["001-zh-CN"] = ("minor", "拿定主意译成下定论，从做决定偏向得出结论，表达不够准确自然。")
BF16_BASE_TEST = {
    "002-zh-CN": ("major", "unlocked 是未上锁，门开着增加门是否敞开的状态。"),
    "004-en": ("major", "周一之前译成 by Monday，截止边界从周一前变成不晚于周一。"),
    "005-en": ("minor", "过去转述中 have left 的时态搭配生硬，已经离开的事实保留。"),
    "006-zh-CN": ("minor", "waited 额外加入站着的姿势，原文并未说明。"),
    "006-en": ("minor", "We neither of us went in 存在多余主语 We，语法生硬。"),
    "009-en": ("minor", "等车限定为 bus，原文没有明确交通工具类型。"),
    "010-zh-CN": ("minor", "我来自己倒的词序生硬，宜为我自己来倒。"),
    "010-en": ("minor", "a empty 应为 an empty，冠词错误。"),
    "012-zh-CN": ("minor", "一盘面条泛化成一份面，盘的容器单位丢失。"),
    "019-zh-CN": ("major", "明确修理费用语境下仍把 charge 译成刑事指控。"),
    "020-zh-CN": ("minor", "译文实际字符串含多余反斜杠，不只是 JSON 必要的引号转义。"),
    "020-en": ("minor", "译文实际字符串含多余反斜杠，引用标点也不自然。"),
}
BF16_ADAPTER_TEST = {k: v for k, v in BF16_BASE_TEST.items() if k not in ("005-en", "020-zh-CN", "020-en")}
BF16_ADAPTER_TEST["012-zh-CN"] = ("major", "one plate of noodles 译成一碗面，明确改变容器与上餐单位。")

SHEETS = {
    "baseline-dev": ("runs/v2-baseline-dev-corrected.jsonl", "9d8c26d005d2664307b9fe4ba66023b7edc0d5e84c8ed54c962724757e633d68", BASE_DEV, "v2-dev-"),
    "adapter-dev": ("runs/v2-adapter-dev.jsonl", "f7f2cbfede768581f932f9707973f18c97dafa0ede4fa45302e3df4a09a8bc67", ADAPTER_DEV, "v2-dev-"),
    "baseline-test": ("runs/v2-baseline-test.jsonl", "077eab182e0f973631861ad0c2840309a649a4919892b4e1b71a1f207be8333d", BASE_TEST, "v2-test-"),
    "adapter-test": ("runs/v2-adapter-test.jsonl", "e1e4fcc1daa729345b447e7baeb8dbcd5f73801376eab028f6ec67eab43b596b", ADAPTER_TEST, "v2-test-"),
    "int8-dev": ("runs/v2-int8-dev.jsonl", "f1db65fb31485bb3e85537157473dd78cc2559d108d05ee8144624c372abb9c2", INT8_DEV, "v2-dev-"),
    "int8-baseline-precision": ("runs/v2-int8-baseline-precision.jsonl", "acfbefc6898e279bcdac29d2a5b0c6b79981652e3e863f94ae689b3ecc4405db", PRECISION_BASE, "v2-precision-"),
    "int8-adapter-precision": ("runs/v2-int8-adapter-precision.jsonl", "29c71d2a31b17432cc1f1100c0e8bc34def6d73b9bfdea6f2cffe69f2be81d4e", PRECISION_ADAPTER, "v2-precision-"),
    "bf16-dev": ("runs/v2-bf16-dev.jsonl", "46091f8e866fcb9e3851e38e37e62386d17a7053458b007de200b0001033faec", BF16_DEV, "v2-dev-"),
    "bf16-baseline-test": ("runs/v2-bf16-baseline-test.jsonl", "47629d60b41e5684c8085ea664a7af3a1dc94b7f02caeca495d092223c17b1d7", BF16_BASE_TEST, "v2-bf16-"),
    "bf16-adapter-test": ("runs/v2-bf16-adapter-test.jsonl", "e3fda6b4986d84b5ee489e3b1edd72f911ade7b70f29c7ad4efc4adf52770357", BF16_ADAPTER_TEST, "v2-bf16-"),
}


def persist(name):
    path, digest, issues, prefix = SHEETS[name]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("原始输出改变，必须重新逐条阅读")
    rows = read_jsonl(path)
    decisions = []
    for row in rows:
        key = row["id"].removeprefix(prefix)
        verdict, note = issues.get(key, ("pass", "已逐条对照当前原文、语境与关键约束；允许等义不同表达，未发现影响含义的问题。"))
        decisions.append({"id": row["id"], "reviewer": "Codex", "at": now(), "semantic_verdict": verdict,
                          "notes": note, "format_valid": "prediction" in row,
                          "source_output_hash": fingerprint({"input": row["input"], "raw": row["raw"], "reference": row["reference"]})})
    if not set(issues).issubset({r["id"].removeprefix(prefix) for r in rows}):
        raise ValueError("判定 id 不在当前集合")
    summary = {"at": now(), "reviewer": "Codex, user-authorized acceptance", "count": len(rows),
               "verdicts": dict(Counter(d["semantic_verdict"] for d in decisions)),
               "format_valid": sum(d["format_valid"] for d in decisions), "evaluation_hash": fingerprint(rows)}
    write_jsonl(f"runs/v2-{name}-semantic.jsonl", decisions)
    write_json(f"runs/v2-{name}-semantic.summary.json", summary)
    return summary


def main():
    print(json.dumps({name: persist(name) for name in SHEETS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
