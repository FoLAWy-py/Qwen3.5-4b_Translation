"""Verify final frozen data and the exact snapshots used by the clean run."""
import json
from pathlib import Path

from witrans_tools.common import fingerprint, read_jsonl, write_json, now
from witrans_tools.data import encode_example, validate_record


def main():
    from transformers import AutoTokenizer
    paths = {"train": "data/prepared/v2/train.jsonl", "dev": "data/prepared/v2/dev_corrected.jsonl", "test": "data/prepared/v2/test.jsonl"}
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-4B", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    report = {"at": now(), "splits": {}}
    for name, path in paths.items():
        rows = read_jsonl(path)
        for row in rows:
            validate_record(row, True, True)
        encoded = [encode_example(tokenizer, row, 1024) for row in rows]
        report["splits"][name] = {"rows": len(rows), "groups": len({r["group_id"] for r in rows}), "hash": fingerprint(rows),
                                   "min_tokens": min(len(r["input_ids"]) for r in encoded), "max_tokens": max(len(r["input_ids"]) for r in encoded)}
    directory = Path("models/witrans-4b-v2-clean/adapter")
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    for name in ("train", "dev"):
        snapshot = read_jsonl(directory / f"{name}_snapshot.jsonl")
        if fingerprint(snapshot) != config[f"{name}_hash"] or config[f"{name}_hash"] != report["splits"][name]["hash"]:
            raise ValueError(f"{name} 的冻结文件、实际读取快照、训练记录指纹不一致")
    report["snapshot_integrity"] = "Verified both snapshots against frozen data and run configuration"
    write_json("runs/v2-data-verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
