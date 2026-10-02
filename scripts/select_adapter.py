"""Install a dev-loss-selected experimental checkpoint into a new directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from witrans_tools.common import ROOT, now, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not all(p.is_relative_to(ROOT.resolve()) for p in (source, output)):
        raise ValueError("模型目录必须在工作区内")
    if output.exists() and any(output.iterdir()):
        raise ValueError("目标目录非空；不能覆盖现有权重")
    metadata = read(source / "witrans_adapter.json")
    if metadata.get("smoke_only"):
        raise ValueError("不能选择 smoke 权重")
    options = [(float(read(source / "metrics.json")["eval_loss"]), "final", source)]
    for checkpoint in (source / "checkpoints").glob("checkpoint-*"):
        state = read(checkpoint / "trainer_state.json")
        current = [r["eval_loss"] for r in state["log_history"] if "eval_loss" in r and r.get("step") == state["global_step"]]
        if current:
            options.append((float(current[-1]), checkpoint.name, checkpoint))
    loss, name, selected = min(options, key=lambda value: value[0])
    output.mkdir(parents=True)
    for artifact in source.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, output / artifact.name)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(selected / filename, output / filename)
    with (output / "adapter_model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata.update(selected_by="minimum new-development loss before test", selected_checkpoint=name,
                    selected_dev_loss=loss, adapter_sha256=digest, quality_status="Experimental, new semantic acceptance pending")
    write_json(output / "witrans_adapter.json", metadata)
    write_json(args.report, {"at": now(), "criterion": metadata["selected_by"], "selected_checkpoint": name,
                            "selected_dev_loss": loss, "adapter_sha256": digest, "installed_adapter": str(output),
                            "source": str(selected), "candidates": [{"loss": v[0], "checkpoint": v[1]} for v in options]})
    print({"checkpoint": name, "loss": loss, "output": str(output)})


if __name__ == "__main__":
    main()
