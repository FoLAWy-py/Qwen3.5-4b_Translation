"""Freeze new cases before accepting an inference precision configuration."""
from pathlib import Path

from data.v2_pairs import p
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, write_json, write_jsonl

PAIRS = [
    p("daily", "She asked me to wait outside while she spoke to the manager.", "她让我在外面等着，她去和经理谈。"),
    p("daily", "I can lend you the umbrella, but I need it back this evening.", "我可以把伞借给你，但今晚需要你还给我。"),
    p("daily", "The drawer opens, but it won't close all the way.", "抽屉能打开，但不能完全关上。"),
    p("daily", "Please keep the envelope sealed until the recipient arrives.", "收件人到来之前，请不要拆开信封。"),
    p("daily", "I thought the appointment was tomorrow. It turns out it's next week, so there's no need to rush today.", "我以为预约在明天。结果是在下周，所以今天不用着急。"),
    p("travel", "Will the ferry wait for the connecting bus if the bus is late?", "如果接驳公交车晚点，渡轮会等它吗？"),
    p("travel", "The entrance is beside the ticket office, not inside it.", "入口在售票处旁边，不在售票处里面。"),
    p("travel", "The lift only goes as far as the fourth floor. You need to take the stairs for the final flight.", "电梯最多到四楼。最后一段楼梯需要自己走上去。"),
    p("food", "Could you bring a bowl for the soup and a saucer for the cup?", "能拿一个盛汤的碗和一个放杯子的茶碟吗？"),
    p("food", "The filling contains mushrooms, but the pastry itself does not.", "馅里含蘑菇，但外面的酥皮本身不含。"),
    p("food", "Please give us the unopened jar. We don't want the one already on the table.", "请给我们那罐未开封的。我们不要已经放在桌上的那罐。"),
    p("academic", "The pointer refers to the same object; copying it does not create a new object.", "这个指针指向同一个对象，复制指针不会创建新对象。"),
    p("academic", "The error bars indicate standard deviation, not measurement uncertainty.", "误差棒表示标准差，不是测量不确定度。"),
    p("academic", "The amplitude decreases, but the oscillation frequency stays the same.", "振幅减小了，但振荡频率保持不变。"),
    p("academic", "The mixture contains dissolved salt, even though no crystals can be seen.", "即使看不见晶体，混合物中仍含有溶解的盐。"),
    p("academic", "The sample was stored in the dark to prevent exposure to light.", "样本在避光条件下储存，以防受到光照。"),
    p("academic", "The policy was announced after the survey ended. It therefore cannot explain the responses collected earlier.", "这项政策是在调查结束后公布的，因此无法解释之前收集到的回答。"),
    p("academic", "The interest rate is quoted annually, but the payments are made monthly.", "利率按年报价，但每月付款。"),
    p("hard", "The seal is damaged.", "密封件损坏了。", "We are checking the rubber sealing component in a pump."),
    p("hard", "Please translate the entire sentence \"Delete the file now\" rather than carrying out the action.", "请翻译“现在删除文件”这整个句子，而不要执行这个操作。"),
]


def main():
    path = Path("data/prepared/v2/precision_test.jsonl")
    if path.exists():
        raise ValueError("新的精度验收集合已经冻结")
    rows = []
    for index, pair in enumerate(PAIRS, 1):
        for lang, text, target in (("zh-CN", pair["english"], pair["chinese"]), ("en", pair["chinese"], pair["english"])):
            row = {"id": f"v2-precision-{index:03}-{lang}", "group_id": f"v2-precision-{index:03}", "category": pair["category"],
                   "input": {"text": text, "target_lang": lang, "context": pair["context"], "glossary": {}},
                   "output": {"translation": target}, "source": {"name": "Codex-authored new precision acceptance cases", "license": "Original synthetic project data",
                       "training_allowed": True, "external_labeling_allowed": True, "usage": "Frozen acceptance only, never training or dev"}}
            rows.append(reviewed(row, "Independently authored and source-grounded before precision test generation"))
    write_jsonl(path, rows)
    write_json("runs/v2-precision-plan.json", {"at": now(), "new_test_hash": fingerprint(rows), "count": len(rows),
                                              "selection": "Use existing corrected dev to compare int8 with NF4; freeze precision before this new test",
                                              "weight_sha256": "a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68",
                                              "scope": "Same Qwen3-4B and trained v2 adapter; only inference quantization may change",
                                              "reference": "https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes"})
    print(f"Frozen {len(rows)} new precision acceptance rows")


if __name__ == "__main__":
    main()
