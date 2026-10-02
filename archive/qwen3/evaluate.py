from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .runtime import WiTrans, SYSTEM_PROMPT, _LocalTranslator, parse_translation
from witrans_tools.common import append_jsonl, base_manifest, fingerprint, now, read_jsonl, write_json
from witrans_tools.data import check_constraints, validate_record


def evaluate(args):
    import torch

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("评测集为空")
    for row in rows:
        validate_record(row, require_output=True, require_review=True, purpose="evaluation")
    if Path(args.output).exists():
        raise ValueError("报告已存在；使用新输出路径")
    manifest = base_manifest(args.base_dir)
    start = time.perf_counter()
    quantization = getattr(args, "quantization", "nf4")
    model = _LocalTranslator(args.base_dir, None, args.max_length, quantization=quantization) if args.baseline else WiTrans(
        args.base_dir, args.adapter_dir, args.max_length, quantization=quantization)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    # Warm-up must not require a model to already obey the JSON contract.
    model.generate_raw("Hello.", target_lang="zh-CN", max_new_tokens=64)
    buckets = defaultdict(lambda: {"count": 0, "valid_json": 0, "constraint_warnings": 0, "generated_tokens": 0, "seconds": []})
    for record in rows:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = {"id": record["id"], "category": record["category"], "input": record["input"],
                  "reference": record["output"], "semantic_review": "pending"}
        valid = False
        warnings = []
        try:
            raw, ended = model.generate_raw(**record["input"], max_new_tokens=args.max_new_tokens)
            result.update(raw=raw, ended=ended)
            result.update(model.last_generation_stats or {})
            if not ended:
                raise ValueError("输出截断")
            translation = parse_translation(raw)
            valid = True
            result["prediction"] = translation
            warnings = check_constraints(record["input"]["text"], translation["translation"])
            result["constraint_warnings"] = warnings
        except (RuntimeError, ValueError) as exc:
            result["error"] = str(exc)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result["seconds"] = elapsed
        append_jsonl(args.output, result)
        for key in ("all", f"{record['category']}/{record['input']['target_lang']}"):
            bucket = buckets[key]
            bucket["count"] += 1
            bucket["valid_json"] += int(valid)
            bucket["constraint_warnings"] += int(bool(warnings))
            bucket["generated_tokens"] += result.get("generated_tokens_including_eos", 0)
            bucket["seconds"].append(elapsed)
        print(f"{record['id']}: JSON={valid}, {elapsed:.2f}s", flush=True)
    for bucket in buckets.values():
        seconds = sorted(bucket.pop("seconds"))
        bucket["mean_seconds"] = sum(seconds) / len(seconds)
        bucket["output_tokens_per_second"] = bucket["generated_tokens"] / sum(seconds)
        bucket["p95_seconds"] = seconds[min(len(seconds)-1, int(len(seconds) * 0.95))]
    write_json(Path(args.output).with_suffix(".summary.json"), {
        "at": now(), "model": "Qwen3-4B baseline" if args.baseline else "witrans-4b",
        "base": manifest, "data_hash": fingerprint(rows), "load_seconds": load_seconds,
        "decoding": {"do_sample": False, "max_length": args.max_length, "max_new_tokens": args.max_new_tokens},
        "quantization": quantization,
        "prompt_hash": fingerprint(SYSTEM_PROMPT),
        "adapter_sha256": model.adapter_sha256,
        "parameter_placement": {device: sum(p.numel() for p in model.model.parameters() if p.device.type == device)
                                for device in ("cpu", "cuda")},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "physical_vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "adapter_dir": None if args.baseline else str(Path(args.adapter_dir).resolve()),
        "buckets": dict(buckets), "semantic_quality": "Pending Codex acceptance authorized by user; format is not semantic quality"})
