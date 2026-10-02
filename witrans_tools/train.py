from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .common import base_manifest, fingerprint, now, read_jsonl, write_json
from .data import encode_example, validate_record


class EncodedDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AnswerCollator:
    """Right padding without recomputing or replacing the answer-only labels."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        batch = {}
        for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows], dtype=torch.long)
        return batch


def load_training_rows(path, tokenizer, max_length, allow_unreviewed=False):
    records = read_jsonl(path)
    if not records:
        raise ValueError("数据为空")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("训练/评估数据 id 重复")
    for r in records:
        validate_record(r, require_output=True, require_review=not allow_unreviewed)
    return records, [encode_example(tokenizer, r, max_length) for r in records]


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 NVIDIA CUDA")
    if (not 256 <= args.max_length <= 2048 or args.rank < 1 or args.epochs <= 0
            or args.max_steps == 0 or args.max_steps < -1):
        raise ValueError("检查 max_length / rank / epochs")
    if args.smoke and args.resume:
        raise ValueError("smoke 不允许恢复正式训练")
    if not 0 < args.learning_rate < 1 or not 0 <= args.lora_dropout < 1 or args.eval_steps < 1 or args.save_total_limit < 1:
        raise ValueError("检查 learning-rate / lora-dropout / eval-steps / save-total-limit")
    destination = Path(args.output)
    if args.smoke and destination.resolve() == (Path("models/witrans-4b/adapter")).resolve():
        raise ValueError("smoke 产物必须保存在 runs/，不能冒充正式权重")
    if destination.exists() and any(destination.iterdir()) and not args.resume:
        raise ValueError("输出目录已有产物；使用新目录或 --resume checkpoint")
    manifest = base_manifest(args.base_dir)
    if manifest.get("status") != "complete":
        raise ValueError("基础模型下载尚未完成")
    tokenizer = AutoTokenizer.from_pretrained(args.base_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records, encoded = load_training_rows(args.input, tokenizer, args.max_length, args.smoke)
    dev_records = []
    dev_encoded = []
    if args.dev:
        dev_records, dev_encoded = load_training_rows(args.dev, tokenizer, args.max_length, args.smoke)
        if {r["group_id"] for r in records} & {r["group_id"] for r in dev_records}:
            raise ValueError("训练/开发集来源组重叠")
        if {r["input"]["text"].strip().casefold() for r in records} & {r["input"]["text"].strip().casefold() for r in dev_records}:
            raise ValueError("训练/开发集原文重叠")
    if not args.smoke and not dev_encoded:
        raise ValueError("正式训练必须提供 --dev")
    if args.smoke:
        # Force a representative longest record into the backward/eval/save test.
        encoded = [max(encoded, key=lambda row: len(row["input_ids"]))]
        dev_encoded = encoded
    valid_labels = [v for v in encoded[0]["labels"] if v != -100]
    if valid_labels[-1] != tokenizer.eos_token_id:
        raise ValueError("EOS loss mask 错误")
    decoded = tokenizer.decode(valid_labels[:-1], skip_special_tokens=False)
    from witrans import parse_translation
    parse_translation(decoded)
    print(f"有效 labels 解码检查通过；最长样本 {max(map(lambda r: len(r['input_ids']), encoded))} tokens", flush=True)
    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    model.print_trainable_parameters()
    configuration = {"at": now(), "base": manifest, "smoke_only": args.smoke,
                     "max_length": args.max_length, "rank": args.rank,
                     "learning_rate": args.learning_rate, "lora_dropout": args.lora_dropout,
                     "epochs": args.epochs, "max_steps": args.max_steps,
                     "eval_steps": args.eval_steps, "save_total_limit": args.save_total_limit,
                     "train_count": len(encoded), "train_hash": fingerprint(records),
                     "dev_hash": fingerprint(dev_records), "seed": args.seed,
                     "prompt_hash": fingerprint(__import__('witrans').SYSTEM_PROMPT),
                     "versions": {n: importlib.metadata.version(n) for n in
                                  ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}
    if args.resume:
        previous = json.loads((destination / "run_config.json").read_text(encoding="utf-8"))
        for key in ("base", "train_hash", "dev_hash", "max_length", "rank", "prompt_hash"):
            if previous[key] != configuration[key]:
                raise ValueError(f"恢复训练配置不一致: {key}")
        defaults = {"learning_rate": 1e-4, "lora_dropout": 0, "epochs": 1, "max_steps": -1, "seed": 42}
        for key, default in defaults.items():
            if previous.get(key, default) != configuration[key]:
                raise ValueError(f"恢复训练配置不一致: {key}")
    write_json(destination / "run_config.json", configuration)
    # Preserve exactly the reviewed records loaded by this run, including audit fields.
    from .common import write_jsonl
    write_jsonl(destination / "train_snapshot.jsonl", records)
    write_jsonl(destination / "dev_snapshot.jsonl", dev_records)
    training_args = TrainingArguments(
        output_dir=str(destination / "checkpoints"), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=1 if args.smoke else 16,
        num_train_epochs=args.epochs, max_steps=1 if args.smoke else args.max_steps,
        learning_rate=args.learning_rate, optim="adamw_bnb_8bit", bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else min(10, args.eval_steps), save_steps=args.eval_steps, save_total_limit=args.save_total_limit,
        eval_strategy="no" if args.smoke else "steps", eval_steps=args.eval_steps,
        prediction_loss_only=True, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False,
        warmup_ratio=0 if args.smoke else 0.03, seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=EncodedDataset(encoded),
                      eval_dataset=EncodedDataset(dev_encoded), data_collator=AnswerCollator(tokenizer.pad_token_id))
    try:
        result = trainer.train(resume_from_checkpoint=args.resume)
        evaluation = trainer.evaluate()
        trainer.save_model(str(destination))
        tokenizer.save_pretrained(destination)
        write_json(destination / "witrans_adapter.json", {
            "base_model_id": manifest["model_id"], "base_revision": manifest["revision"],
            "smoke_only": args.smoke, "prompt_hash": configuration["prompt_hash"], "created_at": now()})
        metrics = {**result.metrics, **evaluation,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                   "physical_vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3}
        metrics["allocator_exceeded_physical_vram"] = metrics["peak_reserved_gib"] > metrics["physical_vram_gib"]
        write_json(destination / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except torch.cuda.OutOfMemoryError:
        write_json(destination / "failure.json", {"at": now(), "reason": "CUDA OOM", "max_length": args.max_length})
        raise RuntimeError("CUDA OOM：换新输出目录，将 max-length 降到 1536/1024，重新切分超长样本；必要时 rank=8") from None
