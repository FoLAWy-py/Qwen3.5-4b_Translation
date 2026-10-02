"""Full 1024-token budget stress check; no semantic-quality claim."""
from pathlib import Path

from archive.qwen3.runtime import WiTrans, make_messages
from witrans_tools.common import now, write_json


def main():
    import torch
    output = Path("runs/v2-bf16-budget.json")
    if output.exists():
        raise ValueError("压力测试报告已存在")
    model = WiTrans("models/Qwen3-4B", "models/witrans-4b-v2-selected/adapter", max_length=1024,
                    quantization="bf16_cpu_vocab")
    def size(context):
        return len(model.tokenizer.apply_chat_template(make_messages("Hello.", "zh-CN", context),
            tokenize=True, add_generation_prompt=True, enable_thinking=False))
    context = "Additional background:"
    while size(context + " background") <= 1024 - 32:
        context += " background"
    receipt = {"at": now(), "quantization": model.quantization, "adapter_sha256": model.adapter_sha256,
               "max_length": 1024, "prompt_tokens": size(context), "max_new_tokens": 32,
               "scope": "Synthetic repeated context tests device placement and memory only, not long-text translation quality"}
    try:
        raw, ended = model.generate_raw("Hello.", "zh-CN", context=context, max_new_tokens=32)
        torch.cuda.synchronize()
        receipt.update(status="completed", ended=ended, raw=raw,
            cpu_vocab=model.model.get_input_embeddings().weight.device.type == "cpu" and model.model.get_output_embeddings().weight.device.type == "cpu")
    except torch.cuda.OutOfMemoryError as exc:
        receipt.update(status="oom", error=str(exc))
    receipt.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                   physical_vram_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3)
    write_json(output, receipt)
    print(receipt)


if __name__ == "__main__":
    main()
