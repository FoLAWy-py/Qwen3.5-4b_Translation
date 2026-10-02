"""Run the retained Qwen3.5 v2 adapter in its isolated environment."""
from witrans_tools.qwen35 import Qwen35Translator


def main():
    translator = Qwen35Translator(
        base_dir="models/Qwen3.5-4B",
        adapter_dir="models/witrans-qwen35-v2-critical-cpo",
        max_length=1024,
    )
    print(translator.translate(
        "The cast has been removed.",
        "zh-CN",
        context="We are discussing a plaster cast on a healed wrist.",
        max_new_tokens=256,
    ))


if __name__ == "__main__":
    main()

