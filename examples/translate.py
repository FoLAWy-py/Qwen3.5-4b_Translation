"""Local experimental WiTrans invocation; see acceptance status in README."""
from pathlib import Path
import argparse

from witrans import WiTrans


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default=str(root / "models/witrans-4b/adapter"))
    parser.add_argument("--quantization", choices=("nf4", "int8", "bf16_cpu_vocab"), default=None)
    args = parser.parse_args()
    model = WiTrans(str(root / "models/Qwen3-4B"), args.adapter_dir, max_length=1024, quantization=args.quantization)
    for text, target in [
        ("Could you put the sauce on the side?", "zh-CN"),
        ("Is the deposit refundable?", "zh-CN"),
        ("Let x be a positive real number.", "zh-CN"),
        ("这道菜可以少放一点盐吗？", "en"),
    ]:
        print(text, "->", model.translate(text, target_lang=target, max_new_tokens=256)["translation"])


if __name__ == "__main__":
    main()
