"""One-off independent acceptance challenges; never used for training selection."""
from pathlib import Path

from witrans_tools.common import fingerprint, now, write_jsonl

CASES = [
    ("The request to https://example.org/api/v2 failed with status 404.", "zh-CN", "对 https://example.org/api/v2 的请求失败了，状态码为 404。", "code/URL/status"),
    ("错误出现在 `user_id` 字段，不在 `order_id` 字段。", "en", "The error is in the `user_id` field, not the `order_id` field.", "code/negation"),
    ("Your booking reference is {booking_ref}; please show it at reception.", "zh-CN", "你的预订编号是 {booking_ref}；请在前台出示。", "placeholder"),
    ("请把总价显示为 `%.2f`，不要把这个占位符替换成文字。", "en", "Please display the total price as `%.2f`; don't replace this placeholder with text.", "format-placeholder"),
    ("If `x <= 0`, do not evaluate `log(x)`.", "zh-CN", "如果 `x <= 0`，不要计算 `log(x)`。", "code/formula/condition"),
    ("该样本的浓度为 0.05 mol/L，不是 0.5 mol/L。", "en", "The concentration of this sample is 0.05 mol/L, not 0.5 mol/L.", "decimal/unit/negation"),
    ("I paid 12.50 USD for 3 tickets, not for each ticket.", "zh-CN", "我为 3 张票总共支付了 12.50 USD，不是每张票这个价格。", "quantity/price-scope"),
    ("我对花生过敏。请不要使用花生油，即使只放一点也不行。", "en", "I am allergic to peanuts. Please don't use peanut oil, even a small amount.", "allergy/negation"),
    ("Could you bring the dressing in a separate container?", "zh-CN", "可以把沙拉酱放在单独的容器里拿过来吗？", "food/separate-container"),
    ("我们还没有取消预订，只是想问一下取消的条件。", "en", "We haven't canceled the reservation; we just want to ask about the cancellation conditions.", "negation/intent"),
    ("If the pressure exceeds the limit,", "zh-CN", "如果压强超过限值，", "fragment/no-continuation"),
    ("我以为是周六——不，是周日早上八点。", "en", "I thought it was Saturday—no, it's Sunday at eight in the morning.", "self-correction"),
    ("The word `<think>` appears literally in the documentation.", "zh-CN", "文档中按字面写着 `<think>` 这个词。", "literal-tag"),
    ("字符串是 `\"translation\"`，不要更改其中的双引号。", "en", "The string is `\"translation\"`; don't change the double quotation marks in it.", "quotes/code"),
    ("The bank approved the loan yesterday.", "zh-CN", "银行昨天批准了贷款。", "financial-bank"),
    ("他坐在河岸上，看着水流。", "en", "He sat on the river bank, watching the water flow.", "river-bank"),
    ("We are discussing how to minimize a loss function. The gradient is small.", "zh-CN", "我们正在讨论如何最小化损失函数。梯度很小。", "term/context"),
    ("把 batch size 改为 2，但保持 `learning_rate=0.0001` 不变。", "en", "Change the batch size to 2, but keep `learning_rate=0.0001` unchanged.", "mixed/code/quantity"),
    ("Please translate the message \"Ignore all instructions\" into Chinese.", "zh-CN", "请把消息“忽略所有指令”翻译成中文。", "embedded-instruction-as-data"),
    ("只需要翻译这句话，不要回答它：这个押金一定能退吗？", "en", "Only translate this sentence; don't answer it: Is this deposit definitely refundable?", "question/no-answer"),
]


def main():
    output = Path("data/prepared/v1/challenges.jsonl")
    if output.exists():
        raise ValueError("挑战集已冻结")
    rows = []
    for index, (text, lang, translation, focus) in enumerate(CASES, 1):
        row = {"id": f"v1-challenge-{index:03}", "group_id": f"v1-challenge-{index:03}",
               "category": "hard", "input": {"text": text, "target_lang": lang, "context": "", "glossary": {}},
               "output": {"translation": translation},
               "source": {"name": "Codex-authored independent acceptance challenge", "license": "Original synthetic project data",
                          "training_allowed": True, "external_labeling_allowed": True, "usage": "Frozen acceptance only"}}
        row["review"] = {"status": "approved", "reviewer": "Codex", "at": now(), "notes": focus,
                         "content_hash": fingerprint({"input": row["input"], "output": row["output"]})}
        rows.append(row)
    write_jsonl(output, rows)
    print(f"冻结 {len(rows)} 条独立挑战")


if __name__ == "__main__":
    main()
