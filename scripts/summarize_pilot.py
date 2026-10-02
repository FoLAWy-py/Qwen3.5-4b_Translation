"""Save measured pilot provenance and usage without exposing credentials."""
import json
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    candidates = read_jsonl("data/generated/candidates.jsonl")
    reviewed = read_jsonl("data/generated/pilot_reviewed.jsonl")
    baseline = json.loads(Path("runs/pilot-baseline-greedy.summary.json").read_text(encoding="utf-8"))
    adapter = json.loads(Path("runs/pilot-adapter-greedy.summary.json").read_text(encoding="utf-8"))
    if baseline["data_hash"] != adapter["data_hash"] or baseline["decoding"] != adapter["decoding"]:
        raise ValueError("比较的数据/解码配置不一致")
    write_json("runs/pilot-status.json", {
        "at": now(), "stage": "pilot completed; full training and human acceptance pending",
        "candidate_count": len(candidates), "reviewed_count": len(reviewed),
        "review_type": "independent AI bilingual review, not human acceptance",
        "edited_count": sum(a["output"] != b["output"] for a, b in zip(candidates, reviewed)),
        "usage": {key: sum(row["label_metadata"]["usage"].get(key, 0) for row in candidates)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost")},
        "candidate_hash": fingerprint(candidates),
        "training": json.loads(Path("models/witrans-4b-pilot/adapter/metrics.json").read_text(encoding="utf-8")),
        "comparison": {"baseline": baseline["buckets"], "pilot_adapter": adapter["buckets"]},
        "limitations": ["Only 32 train / 4 dev / 4 test samples", "Human semantic acceptance pending",
                        "2048 token stress exceeded physical VRAM; use 1024 initial configuration",
                        "Full 3000-5000 sample first stage remains to be prepared and trained"]})
    # Earlier development result used inherited model defaults and is excluded.
    invalid_path = Path("runs/pilot-baseline.summary.json")
    if invalid_path.exists():
        invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
        invalid["excluded_from_comparison"] = "Inherited model defaults; superseded by pilot-baseline-greedy"
        write_json(invalid_path, invalid)
    print(Path("runs/pilot-status.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
