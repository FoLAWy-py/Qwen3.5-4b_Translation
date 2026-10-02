"""Load the experimental pilot adapter once, then translate locally."""
from pathlib import Path

from witrans import WiTrans


def main():
    root = Path(__file__).resolve().parent.parent
    model = WiTrans(str(root / "models/Qwen3-4B"),
                    str(root / "models/witrans-4b-pilot/adapter"), max_length=1024)
    print("试运行模型；尚未通过正式语义质量验收。")
    for text, language in [
        ("Could you put the sauce on the side?", "zh-CN"),
        ("抗体可以识别特定抗原，但并非所有抗体都能中和病原体。", "en"),
    ]:
        print(model.translate(text, target_lang=language, max_new_tokens=256))


if __name__ == "__main__":
    main()
