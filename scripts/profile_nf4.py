"""Small instrumented NF4 diagnostic, never a semantic acceptance benchmark."""
import json
import time
from collections import Counter
from pathlib import Path

from witrans import WiTrans, make_messages
from witrans_tools.common import now, read_jsonl, write_json


class TokenClock:
    def __init__(self):
        self.times = []
        self.prompt_seen = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.times.append(time.perf_counter())

    def end(self):
        pass


def main():
    import torch
    from transformers import GenerationConfig
    output = Path("runs/nf4-diagnostic.json")
    if output.exists():
        raise ValueError("诊断记录已经存在")
    translator = WiTrans("models/Qwen3-4B", "models/witrans-4b-v2-selected/adapter", max_length=1024, quantization="nf4")
    rows = read_jsonl("data/prepared/v2/dev_corrected.jsonl")
    selected = [rows[i] for i in (0, 7, 14, 21, 28, 35)]
    lora_parameters = [(name, parameter) for name, parameter in translator.model.named_parameters() if "lora_" in name]
    original_dtypes = dict(Counter(str(p.dtype) for _, p in lora_parameters))
    report = {"at": now(), "quantization": "nf4", "adapter_sha256": translator.adapter_sha256,
        "scope": "Six development inputs; diagnostic only; streamer adds CPU synchronization; no checkpoint change",
        "cpu_threads": torch.get_num_threads(), "initial_lora_dtypes": original_dtypes, "experiments": {}}
    # In-memory conversion only; original adapter file remains unchanged.
    for experiment in ("original_lora", "bf16_lora"):
        if experiment == "bf16_lora":
            for _, parameter in lora_parameters:
                parameter.data = parameter.data.to(torch.bfloat16)
        translator.generate_raw("Hello.", "zh-CN", max_new_tokens=64)
        torch.cuda.reset_peak_memory_stats()
        results = []
        for row in selected:
            torch.cuda.synchronize()
            start = time.perf_counter()
            prompt = translator.tokenizer.apply_chat_template(make_messages(**row["input"]), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = translator.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            clock = TokenClock()
            config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=128,
                eos_token_id=translator.tokenizer.eos_token_id, pad_token_id=translator.tokenizer.eos_token_id, use_cache=True)
            with torch.inference_mode():
                generated = translator.model.generate(**inputs, generation_config=config, use_model_defaults=False, streamer=clock)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
            assert len(ids) == len(clock.times)
            results.append({"id": row["id"], "seconds": elapsed, "generated_tokens_including_eos": len(ids),
                "ttft_seconds": clock.times[0] - start, "decode_seconds_after_first": clock.times[-1] - clock.times[0],
                "raw": translator.tokenizer.decode(ids[:-1] if ids[-1] == translator.tokenizer.eos_token_id else ids, skip_special_tokens=False)})
        report["experiments"][experiment] = {"rows": results, "mean_seconds": sum(r["seconds"] for r in results) / len(results),
            "mean_ttft_seconds": sum(r["ttft_seconds"] for r in results) / len(results),
            "output_tokens_per_second": sum(r["generated_tokens_including_eos"] for r in results) / sum(r["seconds"] for r in results),
            "decode_tokens_per_second": sum(r["generated_tokens_including_eos"] - 1 for r in results) / sum(r["decode_seconds_after_first"] for r in results),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        print(experiment, {k: v for k, v in report["experiments"][experiment].items() if k != "rows"}, flush=True)
    report["identical_raw_outputs"] = sum(a["raw"] == b["raw"] for a, b in zip(report["experiments"]["original_lora"]["rows"], report["experiments"]["bf16_lora"]["rows"]))
    write_json(output, report)


if __name__ == "__main__":
    main()
