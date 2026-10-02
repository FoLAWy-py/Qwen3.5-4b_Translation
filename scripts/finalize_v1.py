"""Check artifact provenance and record the completed v1 milestone."""
import hashlib
import json
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config = read("models/witrans-4b/adapter/run_config.json")
    datasets = {name: read_jsonl(f"data/prepared/v1/{name}.jsonl") for name in ("train", "dev", "test")}
    if fingerprint(datasets["train"]) != config["train_hash"] or fingerprint(datasets["dev"]) != config["dev_hash"]:
        raise ValueError("训练/开发数据与实际训练记录不一致")
    groups = {name: {row["group_id"] for row in rows} for name, rows in datasets.items()}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[left] & groups[right]:
            raise ValueError("发现来源组泄漏")
    acceptance = read("runs/v1-model-acceptance.json")
    outputs = read_jsonl("runs/v1-adapter-acceptance.jsonl")
    decisions = read_jsonl("runs/v1-adapter-semantic-decisions.jsonl")
    inputs = read_jsonl("data/prepared/v1/acceptance_120.jsonl")
    if fingerprint(outputs) != acceptance["evaluation_hash"]:
        raise ValueError("验收不再绑定当前模型输出")
    for source, output, decision in zip(inputs, outputs, decisions, strict=True):
        if source["id"] != output["id"] or output["id"] != decision["id"] or source["input"] != output["input"]:
            raise ValueError("冻结输入、模型输出、判定不对应")
        bound = fingerprint({"input": output["input"], "raw": output["raw"], "reference": output["reference"]})
        if bound != decision["source_output_hash"]:
            raise ValueError("逐条判定指纹失效")
    metadata = read("models/witrans-4b/adapter/witrans_adapter.json")
    with Path("models/witrans-4b/adapter/adapter_model.safetensors").open("rb") as stream:
        weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if weights_hash != metadata["adapter_sha256"] or weights_hash != acceptance["selection"]["adapter_sha256"]:
        raise ValueError("所验收权重与安装权重不一致")
    evaluation = read("runs/v1-adapter-acceptance.summary.json")
    evaluation.update(semantic_quality=acceptance["verdict"], semantic_acceptance_report="runs/v1-model-acceptance.json")
    write_json("runs/v1-adapter-acceptance.summary.json", evaluation)
    status = {
        "at": now(), "stage": "v1 training and Codex acceptance completed",
        "release_quality": "NOT ACCEPTED; experimental candidate retained",
        "data": {name: {"count": len(rows), "groups": len(groups[name]), "hash": fingerprint(rows)} for name, rows in datasets.items()},
        "training": read("models/witrans-4b-v1-isolated/adapter/metrics.json"),
        "selected_adapter": metadata, "evaluation": evaluation["buckets"]["all"],
        "semantic_acceptance": {key: acceptance[key] for key in ("all", "controlled_screen", "independent_challenges", "baseline_shared_screen")},
        "integrity_checks": "Training/dev hashes, three-way family isolation, all 120 frozen inputs and decision bindings, installed weight hash verified",
        "runtime_diagnostics": "runs/v1-runtime-diagnostics.json", "acceptance_report": "runs/v1-model-acceptance.json",
        "next_stage": "More independent semantic training sources and new dev/test sets; no tuning against this test",
    }
    write_json("runs/v1-status.json", status)
    print(status["integrity_checks"])


if __name__ == "__main__":
    main()
