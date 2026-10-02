"""Original pilot inputs, authored for this project; no scraped training corpus."""
from .common import write_jsonl

EXAMPLES = {
    "daily": [
        ("Could you give me a hand with this box?", "zh-CN"),
        ("I didn't mean to interrupt you.", "zh-CN"),
        ("Would tomorrow afternoon work for you?", "zh-CN"),
        ("I haven't received the confirmation email yet.", "zh-CN"),
        ("Let's split the bill evenly.", "zh-CN"),
        ("我可能会晚到十分钟。", "en"),
        ("能帮我把这个文件发给同事吗？", "en"),
        ("我不是不想去，只是今天实在没有时间。", "en"),
        ("如果你已经提交了申请，就不用再填一次。", "en"),
        ("这件外套还有小一号的吗？", "en"),
    ],
    "travel": [
        ("Is the deposit refundable if I cancel before Friday?", "zh-CN"),
        ("My checked bag didn't arrive on the same flight.", "zh-CN"),
        ("Does this ticket include a transfer to the airport bus?", "zh-CN"),
        ("请问可以提前办理入住吗？", "en"),
        ("去火车站应该在哪一站换乘？", "en"),
        ("我的航班延误了，但还没有取消。", "en"),
    ],
    "food": [
        ("Could you put the sauce on the side?", "zh-CN"),
        ("I am allergic to peanuts, not just trying to avoid them.", "zh-CN"),
        ("Does the soup contain dairy products?", "zh-CN"),
        ("这道菜可以少放一点盐吗？", "en"),
        ("请不要加花生，花生油也不可以。", "en"),
        ("我们想要两份米饭和一壶热茶。", "en"),
    ],
    "academic": [
        ("The learning rate controls the size of each parameter update.", "zh-CN"),
        ("Let x be a positive real number, and assume x < 1.", "zh-CN"),
        ("A statistically significant result does not necessarily imply a large effect.", "zh-CN"),
        ("The voltage across the resistor is proportional to the current through it.", "zh-CN"),
        ("The catalyst increases the reaction rate without changing the equilibrium constant.", "zh-CN"),
        ("During mitosis, replicated chromosomes separate into two daughter cells.", "zh-CN"),
        ("Please distinguish the author's claim from the evidence used to support it.", "zh-CN"),
        ("如果验证集损失持续上升，模型可能已经过拟合。", "en"),
        ("这个矩阵不可逆，因为它的行列式等于零。", "en"),
        ("我们讨论的是损失函数的梯度，而不是道路的坡度。", "en"),
        ("在温度不变的条件下，气体压强与体积成反比。", "en"),
        ("抗体可以识别特定抗原，但并非所有抗体都能中和病原体。", "en"),
        ("机会成本是选择某个方案时放弃的最佳替代方案的价值。", "en"),
        ("请先说明论点，再解释这个例子如何支持它。", "en"),
    ],
    "hard": [
        ("Set `batch_size=1`, then visit https://example.org/docs for the next step.", "zh-CN"),
        ("Send {user_name} a reminder at 09:30, not at 19:30.", "zh-CN"),
        ("把 learning rate 降到 0.0001，其他参数保持不变。", "en"),
        ("我原本以为可以退——等等，只有周五之前取消才可以。", "en"),
    ],
}


def create_seed(path="data/pilot_inputs.jsonl"):
    from pathlib import Path
    if Path(path).exists():
        raise ValueError("种子文件已存在")
    records = []
    for category, examples in EXAMPLES.items():
        for number, (text, language) in enumerate(examples, 1):
            ident = f"pilot-{category}-{number:03}"
            records.append({"id": ident, "group_id": ident, "category": category,
                            "input": {"text": text, "target_lang": language, "context": "", "glossary": {}},
                            "source": {"name": "Original project pilot inputs authored by Codex",
                                       "license": "Project-authored synthetic text for this training task",
                                       "training_allowed": True, "external_labeling_allowed": True}})
    write_jsonl(path, records)
    print(f"保存 {len(records)} 条原创输入，比例 25/15/15/35/10，中英各半")


if __name__ == "__main__":
    create_seed()
