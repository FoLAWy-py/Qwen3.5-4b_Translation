"""Record this session's explicit AI review decisions; not a human approval tool."""
from witrans_tools.common import fingerprint, read_jsonl, write_jsonl

EDITS = {
    "pilot-daily-001": ("能帮我搬一下这个箱子吗？", "修订生硬的‘帮我一下这个箱子’，保留请求帮忙的语气。"),
    "pilot-daily-003": ("你明天下午方便吗？", "修订冗余表达；work for you 在此指时间是否合适。"),
    "pilot-daily-008": ("It's not that I don't want to go; I just really don't have time today.", "保留‘不是不想’的转折语气，并修订逗号连接完整句的问题。"),
    "pilot-travel-002": ("我的托运行李没有随同一班航班到达。", "修订‘没有和我乘坐’的不自然表达，不补充行李去向。"),
    "pilot-travel-003": ("这张票包含换乘机场巴士吗？", "transfer 在票务语境译为换乘；原文未承诺具体接驳服务。"),
    "pilot-food-001": ("可以把酱汁单独放吗？", "on the side 在点餐语境是酱汁与菜分开放，不能只理解为位置在旁边。"),
    "pilot-food-002": ("我对花生过敏，不只是想少吃或不吃花生。", "保留明确的过敏与个人饮食选择的区别，不增添保证。"),
    "pilot-food-004": ("Could you use a little less salt in this dish?", "修订为自然的点餐请求，保留减少盐量。"),
    "pilot-food-005": ("Please don't add peanuts, and don't use peanut oil either.", "保留花生及花生油两个禁止项，并修订英文连接表达。"),
    "pilot-academic-002": ("设 x 为正实数，并假设 x < 1。", "保留变量和不等式，修订中文数学排版。"),
    "pilot-hard-004": ("I originally thought I could get a refund—wait, that's only possible if I cancel before Friday.", "候选把‘退’误译为取消且漏掉自我修正；结合后文取消条件修订为退款，保留周五之前。"),
}

NOTES = {
    "pilot-daily-002": "核对无意打断的否定与人称。",
    "pilot-daily-004": "核对尚未收到邮件的时间状态与否定。",
    "pilot-daily-005": "核对平分账单，不引入金额。",
    "pilot-daily-006": "核对可能性、迟到与十分钟。",
    "pilot-daily-007": "核对请求发送文件给同事，不执行请求。",
    "pilot-daily-009": "核对已提交这一条件及不必重复填写。",
    "pilot-daily-010": "核对外套与更小尺码的疑问。",
    "pilot-travel-001": "核对押金、退款与星期五之前的取消条件。",
    "pilot-travel-004": "核对提前入住请求，不回答能否入住。",
    "pilot-travel-005": "核对换乘车站与前往火车站的目标。",
    "pilot-travel-006": "核对延误与尚未取消的区别。",
    "pilot-food-003": "核对汤、乳制品与疑问语气。",
    "pilot-food-006": "核对两份米饭、一壶热茶。",
    "pilot-academic-001": "核对学习率与参数更新大小，不加入优化建议。",
    "pilot-academic-003": "核对统计显著不必然意味着大效应。",
    "pilot-academic-004": "核对电阻两端电压与电流的正比关系。",
    "pilot-academic-005": "核对催化剂提高速率但不改变平衡常数。",
    "pilot-academic-006": "核对有丝分裂、复制的染色体、两个子细胞。",
    "pilot-academic-007": "核对主张与证据之间的区分要求。",
    "pilot-academic-008": "核对验证损失持续上升与可能过拟合的模态。",
    "pilot-academic-009": "核对矩阵不可逆与行列式为零的因果。",
    "pilot-academic-010": "核对损失函数梯度与道路坡度的词义区分。",
    "pilot-academic-011": "核对温度恒定条件及压强/体积反比。",
    "pilot-academic-012": "核对特定抗原及并非所有抗体都中和的否定。",
    "pilot-academic-013": "核对机会成本定义中的最佳被放弃替代方案。",
    "pilot-academic-014": "核对先陈述论点、后解释例子的顺序。",
    "pilot-hard-001": "核对代码与 URL 完整保留，只翻译自然语言。",
    "pilot-hard-002": "核对占位符和 09:30 / 19:30 及否定；数字警报来自中文词边界而非实际丢失。",
    "pilot-hard-003": "核对 0.0001、learning rate 与其余参数不变。",
}


def main():
    rows = read_jsonl("data/generated/candidates.jsonl")
    if fingerprint(rows) != "9a3f56ab06aece4541e5d71514a45f7018d1e122a7b4ded91dcaf8218992da9b":
        raise ValueError("候选内容有变更；此逐条 AI 复核仅绑定本次候选，必须重新审阅")
    decisions = []
    for row in rows:
        ident = row["id"]
        note = EDITS[ident][1] if ident in EDITS else NOTES[ident]
        decision = {"id": ident, "status": "approved",
                    "reviewer": "Codex independent AI bilingual review (not human acceptance)",
                    "notes": note, "human_review": "pending",
                    "content_hash": fingerprint({"input": row["input"], "output": row["output"]})}
        if ident in EDITS:
            decision["translation"] = EDITS[ident][0]
        decisions.append(decision)
    write_jsonl("data/pilot_decisions.jsonl", decisions)


if __name__ == "__main__":
    main()
