from __future__ import annotations

import argparse

from witrans_tools.common import BASE_DIR


def main():
    parser = argparse.ArgumentParser(description="witrans-4b 数据、训练与本地评测（uv run python -m witrans_tools）")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查 CUDA、NF4 反向传播与 8-bit 优化器")
    doctor.add_argument("--output", default="runs/doctor.json")
    download = commands.add_parser("download", help="下载固定 revision 的原始 Qwen3-4B")
    download.add_argument("--base-dir", default=str(BASE_DIR))
    download.add_argument("--revision")
    label = commands.add_parser("label", help="调用 DeepInfra Qwen3-32B 生成待审核候选译文")
    label.add_argument("--input", required=True)
    label.add_argument("--output", default="data/generated/candidates.jsonl")
    label.add_argument("--limit", type=int, default=20)
    label.add_argument("--retries", type=int, default=2)
    label.add_argument("--max-tokens", type=int, default=512)
    label.add_argument("--workers", type=int, default=1, help="1–16 路并行 API 请求，单线程按序落盘")
    review = commands.add_parser("review", help="应用逐条审核决策")
    review.add_argument("--input", required=True)
    review.add_argument("--decisions", required=True)
    review.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="按来源组划分已审核样本，冻结测试集")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", default="data/prepared/v1")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dev-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    train = commands.add_parser("train", help="4-bit NF4 QLoRA SFT")
    train.add_argument("--base-dir", default=str(BASE_DIR))
    train.add_argument("--input", required=True)
    train.add_argument("--dev")
    train.add_argument("--output", default="models/witrans-4b/adapter")
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--lora-dropout", type=float, default=0)
    train.add_argument("--eval-steps", type=int, default=100)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--epochs", type=float, default=1)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume", help="完整 checkpoint 目录")
    train.add_argument("--smoke", action="store_true", help="仅一个最长样本反向/评估/保存测试；产物不可正式调用")
    evaluate = commands.add_parser("evaluate", help="同提示词/量化/贪心解码本地评测")
    evaluate.add_argument("--base-dir", default=str(BASE_DIR))
    evaluate.add_argument("--adapter-dir", default="models/witrans-4b/adapter")
    evaluate.add_argument("--baseline", action="store_true")
    evaluate.add_argument("--quantization", choices=("nf4", "int8", "bf16_cpu_vocab"), default="nf4")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--max-length", type=int, default=2048)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.command == "doctor":
        from .doctor import doctor as execute
    elif args.command == "download":
        from .download import download as execute
    elif args.command == "label":
        from .label import label as execute
    elif args.command == "review":
        from .review import review as execute
    elif args.command == "prepare":
        from witrans_tools.data import prepare as execute
    elif args.command == "train":
        from .train import train as execute
    else:
        from .evaluate import evaluate as execute
    execute(args)


if __name__ == "__main__":
    main()
