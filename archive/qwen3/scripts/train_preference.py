"""Small auditable NF4 CPO/control trial, fixed updates from one starting adapter."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from archive.qwen3.runtime import SYSTEM_PROMPT, WiTrans
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record
from witrans_tools.preference import answer_scores, cpo_loss
from witrans_tools.critical_spans import encode_critical_spans, validated_annotations


def main():
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpo", "sft_control"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument('--data-dir', default='data/prepared/v3')
    parser.add_argument('--starting-adapter', default='models/witrans-4b-v2-selected/adapter')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--critical-spans')
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("训练目录已存在")
    if args.steps < 1 or args.accumulation < 1:
        raise ValueError('Updates and accumulation must be positive')
    rows = read_jsonl(Path(args.data_dir) / 'train.jsonl')
    dev = read_jsonl(Path(args.data_dir) / 'dev.jsonl')
    if not rows or not dev:
        raise ValueError('Empty training or development set')
    span_report, annotations = None, {}
    if args.critical_spans:
        if args.mode != 'cpo' or args.smoke:
            raise ValueError('Frozen critical-span trial requires complete CPO mode')
        span_report = json.loads(Path(args.critical_spans).read_text(encoding='utf-8'))
        annotations = validated_annotations(rows, span_report)
    manifest_path = Path(args.data_dir) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    protocol = manifest.get('protocol', {}) if manifest else {}
    if protocol:
        if (manifest['train']['hash'] != fingerprint(rows) or manifest['dev']['hash'] != fingerprint(dev)
                or args.mode not in protocol['modes'] or args.steps != protocol['steps']
                or args.accumulation != protocol['accumulation']
                or Path(args.starting_adapter).resolve() != Path(protocol['starting_adapter']).resolve()):
            raise ValueError('Frozen trial configuration or data changed')
    for row in rows + dev:
        validate_record(row, True, True)
        validate_record({**row, "output": row["rejected"]}, True)
        if row["output"] == row["rejected"]:
            raise ValueError("正负译文相同")
        if row["preference_review"]["hash"] != fingerprint({k: row[k] for k in ("input", "output", "rejected", "preference_issue")}):
            raise ValueError("偏好对未审核或内容变化")
    if {r["group_id"] for r in rows} & {r["group_id"] for r in dev}:
        raise ValueError("训练开发来源重叠")
    if {r['input']['text'].strip().casefold() for r in rows} & {r['input']['text'].strip().casefold() for r in dev}:
        raise ValueError('训练开发原文重叠')
    set_seed(42)
    # WiTrans verifies starting revision, prompt, and actual adapter weight hash.
    translator = WiTrans("models/Qwen3-4B", args.starting_adapter, max_length=1024, quantization="nf4")
    if protocol and translator.adapter_sha256 != protocol['starting_adapter_sha256']:
        raise ValueError('Frozen starting adapter changed')
    base = translator.model.unload()
    if any("lora_" in name for name, _ in base.named_parameters()):
        raise ValueError("卸载后仍有旧LoRA，禁止重复注入")
    if hasattr(base, "peft_config"):
        delattr(base, "peft_config")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, args.starting_adapter, is_trainable=True, local_files_only=True)
    model.config.use_cache = False
    tokenizer = translator.tokenizer
    def encode(row, rejected=False):
        example = {**row, "output": row["rejected"] if rejected else row["output"]}
        if not rejected and row['id'] in annotations:
            annotation = annotations[row['id']]
            return encode_critical_spans(tokenizer, example, annotation['critical_spans'],
                1024, annotation['multiplier'])
        return encode_example(tokenizer, example, 1024)
    encoded = [(encode(r), encode(r, True)) for r in rows]
    dev_encoded = [(encode(r), encode(r, True)) for r in dev]
    def score(example):
        tensors = {k: torch.tensor([v], device="cuda:0") for k,v in example.items()}
        labels = tensors.pop("labels")
        weights = tensors.pop('token_weights', None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits, labels, weights)
    steps, accumulation = (1, 1) if args.smoke else (args.steps, args.accumulation)
    config = {"at": now(), "mode": args.mode, "steps": steps, "accumulation": accumulation,
        "learning_rate": 1e-5, "beta": 0.1, "seed": 42, "max_length": 1024, "quantization": "nf4",
        "starting_adapter_sha256": translator.adapter_sha256, "train_hash": fingerprint(rows), "dev_hash": fingerprint(dev),
        "control": "Both modes use two forwards per pair and equal updates; SFT control uses preferred answer twice; not a strictly equal FLOP comparison",
        'starting_adapter_dir':str(Path(args.starting_adapter).resolve()),
        'data_dir':str(Path(args.data_dir).resolve()),
        "selection": "Fixed update budget, common known-dev CPO objective auxiliary only; semantic development comparison required before release testing"}
    if span_report:
        config.update(method='critical_span_cpo', annotation_hash=span_report['annotation_hash'],
            annotated_rows=len(annotations), weighted_component='Chosen normalized NLL only; summed logp margin unchanged')
        write_json(destination / 'critical_spans_snapshot.json', span_report)
    write_json(destination / 'run_config.json', config)
    if manifest:
        write_json(destination / 'data_manifest_snapshot.json', manifest)
    write_jsonl(destination / "train_snapshot.jsonl", rows)
    write_jsonl(destination / "dev_snapshot.jsonl", dev)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters, lr=1e-5)
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0
        for micro in range(accumulation):
            chosen, rejected = encoded[order[(step * accumulation + micro) % len(order)]]
            lp, nll = score(chosen)
            if args.mode == "cpo":
                rejected_lp, _ = score(rejected)
                loss = cpo_loss(lp, rejected_lp, nll)
            else:
                _, second_nll = score(chosen)
                loss = (nll + second_nll) / 2
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append({"step": step + 1, "loss": total})
        print(history[-1], flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    model.eval()
    losses, margins = [], []
    with torch.inference_mode():
        for chosen, rejected in dev_encoded:
            lp, nll = score(chosen)
            rp, _ = score(rejected)
            losses.append(float(cpo_loss(lp, rp, nll)))
            margins.append(float(lp - rp))
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    with (destination / "adapter_model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    original = json.loads((Path(args.starting_adapter) / 'witrans_adapter.json').read_text(encoding="utf-8"))
    for key in ("selected_by", "selected_checkpoint", "selected_dev_loss", "acceptance_report"):
        original.pop(key, None)
    original.update(smoke_only=args.smoke, adapter_sha256=digest, prompt_hash=fingerprint(SYSTEM_PROMPT),
        created_at=now(), recommended_quantization="nf4", quality_status="Experimental, semantic acceptance pending", training_method='critical_span_cpo' if annotations else args.mode,
        parent_adapter_sha256=translator.adapter_sha256)
    write_json(destination / "witrans_adapter.json", original)
    write_json(destination / "metrics.json", {"at": now(), "train_seconds": train_seconds, "history": history,
        "eval_cpo_loss": sum(losses) / len(losses), "dev_preference_accuracy": sum(m > 0 for m in margins) / len(margins),
        "dev_margins": margins, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "physical_vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "trainable_parameters": sum(p.numel() for p in parameters)})
    print({"dev_cpo_loss": sum(losses) / len(losses), "train_seconds": train_seconds}, flush=True)


if __name__ == "__main__":
    main()
