"""Freeze inference precision from reviewed development results, never test."""
import json
from pathlib import Path

from witrans_tools.common import now, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if Path("runs/v2-precision-selection.json").exists():
        raise ValueError("推理精度选择已冻结")
    options = []
    for mode, name in (("nf4", "adapter-dev"), ("int8", "int8-dev")):
        review = read(f"runs/v2-{name}-semantic.summary.json")
        raw = read("runs/v2-adapter-dev.summary.json" if mode == "nf4" else "runs/v2-int8-dev.summary.json")
        options.append({"mode": mode, "major": review["verdicts"].get("major", 0), "pass": review["verdicts"].get("pass", 0),
                        "json_valid": review["format_valid"], "count": review["count"], "dev_review_hash": review["evaluation_hash"],
                        "mean_seconds": raw["buckets"]["all"]["mean_seconds"], "peak_allocated_gib": raw["peak_allocated_gib"]})
    chosen = min(options, key=lambda r: (r["major"], -r["pass"], r["mean_seconds"]))
    receipt = {"at": now(), "criterion": "Development semantic major errors first, acceptable count second, latency breaks ties; no new test results used",
               "selected_quantization": chosen["mode"], "options": options,
               "weight_sha256": "a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68"}
    write_json("runs/v2-precision-selection.json", receipt)
    metadata_path = Path("models/witrans-4b-v2-selected/adapter/witrans_adapter.json")
    metadata = read(metadata_path)
    metadata.update(recommended_quantization=chosen["mode"], quality_status="Experimental; final new precision acceptance pending")
    write_json(metadata_path, metadata)
    print(receipt)


if __name__ == "__main__":
    main()
