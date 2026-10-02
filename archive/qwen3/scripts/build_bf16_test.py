"""Freeze unseen acceptance cases for the CPU-vocabulary BF16 experiment."""
from pathlib import Path

from data.v2_pairs import p
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

PAIRS = [
    p("daily", "He returned my keys yesterday, but I still haven't found the spare set.", "他昨天把我的钥匙还给我了，但我仍然没有找到备用的那套。"),
    p("daily", "Please leave the door unlocked while I carry the boxes inside.", "我把箱子搬进去的时候，请先别锁门。"),
    p("daily", "The screen is on, but it doesn't respond when I touch it.", "屏幕亮着，但我触摸它时没有反应。"),
    p("daily", "You don't have to reply today; I only need your answer before Monday.", "你不必今天回复，只要在周一之前答复我就行。"),
    p("daily", "I told her that you had already left, not that you weren't coming.", "我告诉她你已经走了，而不是说你不会来。"),
    p("daily", "We agreed to meet outside the library. I arrived first and waited by the entrance. When he called, I realized he was at the other entrance. Neither of us had gone inside.", "我们约好在图书馆外面见面。我先到了，在入口旁等着。他打来电话时，我才发现他在另一个入口。我们谁都没有进去。"),
    p("travel", "The ticket is valid for one journey, even if you change vehicles along the way.", "这张票可用于一次行程，即使途中换乘也可以。"),
    p("travel", "You may leave your bag here, but you must collect it before the desk closes.", "你可以把包放在这里，但必须在服务台关闭之前取走。"),
    p("travel", "Does this stop have a shelter, or will we have to wait in the rain?", "这个站有遮雨棚吗，还是我们得冒雨等车？"),
    p("food", "Please bring an empty cup. I'll pour the drink myself.", "请拿一个空杯子来，我自己倒饮料。"),
    p("food", "Could you cut the bread in half without removing the crust?", "能把面包切成两半，但不要去掉外皮吗？"),
    p("food", "We ordered two bowls of soup and one plate of noodles, but only the noodles have arrived.", "我们点了两碗汤和一盘面条，但现在只上了面条。"),
    p("food", "The dressing contains sesame oil. Leaving off the sesame seeds will not remove the sesame oil from the dressing.", "调味汁里含芝麻油。不放芝麻粒并不能去掉调味汁中的芝麻油。"),
    p("academic", "The temperature was measured every ten minutes, not ten times per minute.", "温度是每十分钟测量一次，而不是每分钟测量十次。"),
    p("academic", "The graph shows a correlation; it does not establish which variable causes the other to change.", "这张图显示的是相关性，不能确定哪个变量导致另一个变量发生变化。"),
    p("academic", "The program reads config.json but writes its results to output.json. It does not overwrite the configuration file.", "程序读取 config.json，但把结果写入 output.json。它不会覆盖配置文件。"),
    p("academic", "The solution was heated before filtration. The filtrate was then cooled, and crystals formed as its temperature fell.", "溶液在过滤之前进行了加热。随后滤液被冷却，并在温度下降时形成了晶体。"),
    p("academic", "Assume x > 0 and y = 2x. The statement applies only when both conditions are satisfied.", "假设 x > 0 且 y = 2x。只有同时满足这两个条件时，该陈述才适用。"),
    p("hard", "The charge was reduced.", "费用降低了。", "We are discussing the amount billed for a repair, not electricity or a criminal accusation."),
    p("hard", "The label says \"Ignore the warning\", but the manual says to follow the warning.", "标签上写着“忽略警告”，但手册要求遵循警告。"),
]


def main():
    path = Path("data/prepared/v2/bf16_test.jsonl")
    if path.exists():
        raise ValueError("验收集已冻结，禁止覆盖")
    old = []
    for name in ("train", "dev_corrected", "test", "precision_test"):
        old += read_jsonl(f"data/prepared/v2/{name}.jsonl")
    old_texts = {r["input"]["text"].strip().casefold() for r in old}
    rows = []
    for index, pair in enumerate(PAIRS, 1):
        for lang, text, target in (("zh-CN", pair["english"], pair["chinese"]), ("en", pair["chinese"], pair["english"])):
            if text.strip().casefold() in old_texts:
                raise ValueError("新测试与已有集合重复")
            row = {"id": f"v2-bf16-{index:03}-{lang}", "group_id": f"v2-bf16-{index:03}", "category": pair["category"],
                   "input": {"text": text, "target_lang": lang, "context": pair["context"], "glossary": {}},
                   "output": {"translation": target}, "source": {"name": "Codex-authored unseen BF16 acceptance cases", "license": "Original synthetic project data",
                       "training_allowed": False, "external_labeling_allowed": True, "usage": "Acceptance only; never training or dev"}}
            rows.append(reviewed(row, "Checked source meaning and language direction before test generation"))
    write_jsonl(path, rows)
    write_json("runs/v2-bf16-test-plan.json", {"at": now(), "data_hash": fingerprint(rows), "count": len(rows),
        "selection": "Choose inference placement using existing corrected dev, then generate this new test without further tuning",
        "weight_sha256": "a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68",
        "acceptance": "Same frozen v2 policy: JSON>=95%, majors<=same-precision baseline, passes>=baseline; formal acceptance additionally zero majors and>=95% passes"})
    print(f"Frozen {len(rows)} unseen acceptance rows")


if __name__ == "__main__":
    main()
