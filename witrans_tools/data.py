from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict

from witrans import make_messages, parse_translation
from .common import fingerprint, read_jsonl, write_json, write_jsonl

CATEGORIES = {"daily", "travel", "food", "academic", "hard"}


def validate_record(record, require_output=False, require_review=False, *, purpose="training"):
    for name in ("id", "group_id", "category"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"缺少非空字段 {name}")
    if record["category"] not in CATEGORIES:
        raise ValueError("未知 category")
    source = record.get("source", {})
    if purpose not in ("training", "evaluation"):
        raise ValueError("未知数据用途")
    if any(type(source.get(flag)) is not bool for flag in ("training_allowed", "external_labeling_allowed")):
        raise ValueError("source 必须明确记录授权状态")
    if purpose == "training" and (source["training_allowed"] is not True or source["external_labeling_allowed"] is not True):
        raise ValueError("source 必须记录训练和外发标注授权")
    if not source.get("name") or not source.get("license"):
        raise ValueError("source 缺少名称和许可")
    make_messages(**record["input"])
    if not record["input"]["text"].strip():
        raise ValueError("训练原文不能为空")
    if require_output:
        parse_translation(json.dumps(record["output"], ensure_ascii=False))
        if not record["output"]["translation"].strip():
            raise ValueError("训练译文不能为空")
    if require_review:
        review = record.get("review", {})
        if (review.get("status") != "approved" or not review.get("reviewer")
                or review.get("content_hash") != fingerprint({"input": record["input"], "output": record["output"]})):
            raise ValueError("样本未审核或审核后内容有变更")


def protected_items(text):
    # Conservative diagnostic only: a translated number can be semantically correct.
    pattern = r"https?://[^\s<>\"，。]+|\{[A-Za-z_][\w.]*\}|%\([\w]+\)s|%[sd]|`[^`]+`|(?<![\d.])\d+(?:\.\d+)?(?!\d)"
    return Counter(re.findall(pattern, text))


def check_constraints(text, translation):
    expected, actual = protected_items(text), protected_items(translation)
    return [f"需核对保留内容: {item}" for item, count in expected.items()
            if actual[item] < count]


def encode_example(tokenizer, record, max_length=2048):
    prompt_ids = tokenizer.apply_chat_template(
        make_messages(**record["input"]), tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    # Transformers5 defaults to BatchEncoding; older releases return a list.
    # Explicitly request IDs, and validate rather than treating mapping keys
    # as two prompt tokens and silently corrupting supervision/budgets.
    if isinstance(prompt_ids, dict) or hasattr(prompt_ids, 'keys'):
        prompt_ids = prompt_ids['input_ids']
    if not isinstance(prompt_ids, (list, tuple)) or any(type(v) is not int for v in prompt_ids):
        raise ValueError('Chat template must return flat integer prompt token IDs')
    answer = json.dumps(record["output"], ensure_ascii=False, separators=(",", ":"))
    if tokenizer.eos_token_id is None:
        raise ValueError("Missing EOS token")
    target_ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = list(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        raise ValueError(f"{record['id']}: {len(input_ids)} tokens 超过 {max_length}；重新切分，不能截断答案")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids}


def split_records(records, seed=42, dev_fraction=0.1, test_fraction=0.1):
    if not 0 < dev_fraction < 1 or not 0 < test_fraction < 1 or dev_fraction + test_fraction >= 1:
        raise ValueError("划分比例无效")
    groups = defaultdict(list)
    seen_ids, seen_inputs = set(), {}
    for record in records:
        validate_record(record, require_output=True, require_review=True)
        if record["id"] in seen_ids:
            raise ValueError("重复 id")
        seen_ids.add(record["id"])
        # Same source text with differing contexts may still leak across splits.
        key = " ".join(record["input"]["text"].split()).casefold()
        previous_group = seen_inputs.get(key)
        if previous_group is not None and previous_group != record["group_id"]:
            raise ValueError("相同原文跨 group；请统一来源分组后再划分")
        seen_inputs[key] = record["group_id"]
        groups[record["group_id"]].append(record)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("至少需要三个独立来源组")
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_dev = max(1, round(len(keys) * dev_fraction))
    n_test = max(1, round(len(keys) * test_fraction))
    if n_dev + n_test >= len(keys):
        raise ValueError("来源组不足以划分")
    assignments = {"dev": keys[:n_dev], "test": keys[n_dev:n_dev+n_test], "train": keys[n_dev+n_test:]}
    result = {}
    for split, selected in assignments.items():
        result[split] = [record for key in selected for record in groups[key]]
        rng.shuffle(result[split])
    return result


def prepare(args):
    from pathlib import Path
    result = split_records(read_jsonl(args.input), args.seed, args.dev_fraction, args.test_fraction)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("输出目录必须为空；保护已冻结的划分")
    summary = {}
    for split, records in result.items():
        write_jsonl(destination / f"{split}.jsonl", records)
        summary[split] = {"count": len(records), "sha256": fingerprint(records),
                          "categories": dict(Counter(r["category"] for r in records)),
                          "directions": dict(Counter(r["input"]["target_lang"] for r in records))}
    write_json(destination / "split_manifest.json", {"seed": args.seed, "splits": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
