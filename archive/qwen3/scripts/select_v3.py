"""Freeze the equal-update trial choice using its common development objective."""
import hashlib
import json
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    path = Path("runs/v3-selection.json")
    if path.exists():
        raise ValueError("选择已经冻结")
    candidates = {}
    for name in ("cpo", "sft-control"):
        directory = Path(f"models/witrans-4b-v3-{name}/adapter")
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        with (directory / "adapter_model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        candidates[name] = {"config": config, "metrics": metrics, "adapter_sha256": digest, "directory": str(directory.resolve())}
    assert len({v["config"]["train_hash"] for v in candidates.values()}) == 1
    assert len({v["config"]["starting_adapter_sha256"] for v in candidates.values()}) == 1
    selected = min(candidates, key=lambda name: candidates[name]["metrics"]["eval_cpo_loss"])
    write_json(path, {"at": now(), "criterion": "Common new-development CPO objective at fixed8updates; never test results",
        "selected": selected, "candidates": candidates, "test_hash": fingerprint(read_jsonl("data/prepared/v3/test.jsonl")),
        "quality_status": "Pending source-grounded semantic acceptance; objective and preference accuracy are not translation quality"})
    print({"selected": selected, "losses": {name: v["metrics"]["eval_cpo_loss"] for name,v in candidates.items()}})


if __name__ == "__main__":
    main()
