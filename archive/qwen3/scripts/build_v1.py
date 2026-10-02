"""Build frozen, family-disjoint controlled corpora from Codex-reviewed pairs."""
import json
import random
from collections import Counter
from pathlib import Path

from data.v1_families import FAMILIES, NOTES
from witrans_tools.common import fingerprint, now, write_json, write_jsonl

GROUP_COUNTS = {
    "daily": (19, 6, 6), "travel": (11, 4, 4), "food": (11, 4, 4),
    "academic": (26, 9, 9), "hard": (8, 2, 2),
}
TERMS = {
    "daily": {"confirmation email": "确认邮件", "attachment": "附件"},
    "travel": {"deposit": "押金", "checked bags": "托运行李", "reservation": "预订"},
    "food": {"peanuts": "花生", "peanut oil": "花生油", "dairy products": "乳制品"},
    "academic": {"learning rate": "学习率", "gradient": "梯度", "determinant": "行列式",
                 "validation loss": "验证损失", "overfitting": "过拟合", "buffers": "缓冲区",
                 "confidence interval": "置信区间", "equilibrium constant": "平衡常数",
                 "antibodies": "抗体", "opportunity cost": "机会成本", "marginal costs": "边际成本"},
    "hard": {"learning rate": "学习率", "river bank": "河岸"},
}
CONTEXTS = {
    "daily": ("This is a conversation about everyday arrangements.", "这是关于日常安排的对话。"),
    "travel": ("This is a conversation about travel arrangements.", "这是关于旅行安排的对话。"),
    "food": ("This is a conversation about ordering or serving food.", "这是关于点餐或上菜的对话。"),
    "academic": ("This sentence is part of a classroom discussion.", "这句话来自课堂讨论。"),
    "hard": ("Translate the current sentence, preserving literal text when indicated.", "请翻译当前句子，并保留明确标出的字面文本。"),
}


def build():
    destination = Path("data/prepared/v1")
    if destination.exists():
        raise ValueError("v1 数据目录已经冻结，不能覆盖")
    rng = random.Random(20260930)
    corpus, references, families = [], [], []
    splits = {"train": [], "dev": [], "test": []}
    for category, pairs in FAMILIES.items():
        counts = GROUP_COUNTS[category]
        if len(pairs) != sum(counts):
            raise ValueError(f"{category} 家族数量不符: {len(pairs)}")
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        assignments = {}
        offset = 0
        for split, count in zip(splits, counts):
            for index in indices[offset:offset+count]:
                assignments[index] = split
            offset += count
        for index, (english, chinese) in enumerate(pairs):
            group = f"v1-{category}-{index+1:03}"
            family = {"group_id": group, "english_template": english, "chinese_template": chinese,
                      "category": category, "split": assignments[index], "reviewer": "Codex",
                      "review_basis": "Independent bilingual authoring and semantic family acceptance",
                      "notes": NOTES[category], "accepted_at": now()}
            family["content_hash"] = fingerprint({"english": english, "chinese": chinese})
            families.append(family)
            for variant, n in enumerate(range(2, 42, 2), 1):
                en, zh = english.format(n=n), chinese.format(n=n)
                for target, text, reference in (("zh-CN", en, zh), ("en", zh, en)):
                    terms = {k: v for k, v in TERMS[category].items() if k.lower() in en.lower() and v in zh}
                    context = ""
                    glossary = {}
                    if variant <= 4:
                        context = CONTEXTS[category][0 if target == "zh-CN" else 1]
                    elif variant <= 8:
                        glossary = terms if target == "zh-CN" else {v: k for k, v in terms.items()}
                    record = {"id": f"{group}-{variant:02}-{target}", "group_id": group, "category": category,
                              "input": {"text": text, "target_lang": target, "context": context, "glossary": glossary},
                              "source": {"name": "Codex-authored controlled bilingual v1 corpus",
                                         "license": "Original synthetic project data authored for this task",
                                         "training_allowed": True, "external_labeling_allowed": True,
                                         "family_hash": family["content_hash"], "construction": "20 even-number substitutions per family; both directions"}}
                    ref = {**record, "output": {"translation": reference}}
                    ref["review"] = {"status": "approved", "reviewer": "Codex (user-authorized AI acceptance)",
                                     "method": "Reviewed bilingual family plus deterministic slot substitution",
                                     "notes": NOTES[category], "at": now(), "family_hash": family["content_hash"],
                                     "content_hash": fingerprint({"input": ref["input"], "output": ref["output"]})}
                    corpus.append(record)
                    references.append(ref)
                    splits[assignments[index]].append(ref)
    texts = [r["input"]["text"].casefold().strip() for r in corpus]
    if len(set(texts)) != len(texts):
        raise ValueError("存在重复原文，需先统一来源组")
    for split, rows in splits.items():
        rng.shuffle(rows)
        # Training targets will be installed after teacher annotation and acceptance.
        if split != "train":
            write_jsonl(destination / f"{split}.jsonl", rows)
    write_jsonl("data/v1_inputs.jsonl", corpus)
    write_jsonl("data/v1_references.jsonl", references)
    write_jsonl("data/v1_family_acceptance.jsonl", families)
    manifest = {"created_at": now(), "method": "Frozen stratified source-family split before labeling",
                "source_families": len(families), "total_records": len(corpus),
                "independent_family_split": True,
                "limitations": "Controlled templates, 20 numeric variants; row count is not independent semantic diversity",
                "splits": {name: {"count": len(rows), "family_count": len({r['group_id'] for r in rows}),
                                  "categories": dict(Counter(r['category'] for r in rows)),
                                  "directions": dict(Counter(r['input']['target_lang'] for r in rows)),
                                  "sha256": fingerprint(rows)} for name, rows in splits.items()},
                "family_hash": fingerprint(families), "references_hash": fingerprint(references)}
    write_json(destination / "split_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
