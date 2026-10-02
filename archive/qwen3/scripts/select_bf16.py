"""Freeze a development-only inference choice before unseen acceptance."""
import json
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    output = Path("runs/v2-bf16-selection.json")
    if output.exists():
        raise ValueError("推理配置选择已冻结")
    candidates = {}
    for name, key in (("nf4", "adapter-dev"), ("int8", "int8-dev"), ("bf16_cpu_vocab", "bf16-dev")):
        semantic = json.loads(Path(f"runs/v2-{key}-semantic.summary.json").read_text(encoding="utf-8"))
        runtime_key = "adapter-dev" if name == "nf4" else "int8-dev" if name == "int8" else "bf16-dev"
        runtime = json.loads(Path(f"runs/v2-{runtime_key}.summary.json").read_text(encoding="utf-8"))
        candidates[name] = {"semantic": semantic["verdicts"], "json_valid": semantic["format_valid"],
            "mean_seconds": runtime["buckets"]["all"]["mean_seconds"], "reserved_gib": runtime.get("peak_reserved_gib"),
            "dev_hash": runtime["data_hash"], "evaluation_hash": semantic["evaluation_hash"]}
    assert len({c["dev_hash"] for c in candidates.values()}) == 1
    selected = min(candidates, key=lambda n: (candidates[n]["semantic"].get("major", 0),
        -candidates[n]["semantic"].get("pass", 0), candidates[n]["mean_seconds"]))
    write_json(output, {"at": now(), "selected": selected, "criterion": "Major errors ascending, passes descending, mean latency ascending; dev only",
        "candidates": candidates, "new_test_hash": fingerprint(read_jsonl("data/prepared/v2/bf16_test.jsonl")),
        "status": "Inference experiment only; never a semantic acceptance or adapter promotion"})
    print(selected)


if __name__ == "__main__":
    main()
