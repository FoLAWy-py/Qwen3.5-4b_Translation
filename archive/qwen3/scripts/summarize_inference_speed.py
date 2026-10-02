"""Compare same-dev runs; token rates reconstructed from frozen raw outputs."""
import json
from pathlib import Path

from transformers import AutoTokenizer

from witrans_tools.common import now, read_jsonl, write_json


def main():
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B", local_files_only=True)
    runs = {}
    ids = None
    data_hash = None
    for precision, stem in (("NF4", "v2-adapter-dev"), ("INT8", "v2-int8-dev"), ("BF16_CPU_VOCAB", "v2-bf16-dev")):
        rows = read_jsonl(f"runs/{stem}.jsonl")
        summary = json.loads(Path(f"runs/{stem}.summary.json").read_text(encoding="utf-8"))
        current_ids = [r["id"] for r in rows]
        if ids is not None:
            assert current_ids == ids and data_hash == summary["data_hash"]
        ids, data_hash = current_ids, summary["data_hash"]
        tokens = sum(len(tokenizer.encode(row["raw"], add_special_tokens=False)) + int(row["ended"]) for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        runs[precision] = {"rows": len(rows), "mean_seconds": seconds / len(rows),
            "reconstructed_output_tokens": tokens, "mean_output_tokens": tokens / len(rows),
            "estimated_tokens_per_second": tokens / seconds,
            "peak_allocated_gib": summary["peak_allocated_gib"], "peak_reserved_gib": summary.get("peak_reserved_gib"),
            "raw_outputs": f"runs/{stem}.jsonl"}
    result = {"at": now(), "data_hash": data_hash,
        "method": "Output-token rate is estimated by retokenizing saved raw JSON plus EOS, not directly recorded generated IDs. Total output tokens / total timed generation seconds. Includes prompt tokenization/prefill/decode, excludes model loading and warmup.",
        "memory": "PyTorch process peak allocation/reservation, not whole-device nvidia-smi usage. NF4 reservation was not recorded.",
        "runs": runs}
    write_json("runs/v2-inference-speed.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
