"""Choose a checkpoint by development loss before test-screen evaluation."""
import json
import shutil
import hashlib
from pathlib import Path

from witrans_tools.common import ROOT, now, write_json


def main():
    source = ROOT / "models/witrans-4b-v1-isolated/adapter"
    destination = ROOT / "models/witrans-4b/adapter"
    final_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    options = [(float(final_metrics["eval_loss"]), "final", source)]
    for checkpoint in (source / "checkpoints").glob("checkpoint-*"):
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        evaluations = [r for r in state["log_history"] if "eval_loss" in r and r.get("step") == state["global_step"]]
        if evaluations:
            options.append((float(evaluations[-1]["eval_loss"]), checkpoint.name, checkpoint))
    loss, name, selected = min(options, key=lambda value: value[0])
    for path in (source, destination, selected):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("模型路径必须位于工作区")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "adapter_model.safetensors").exists():
        raise ValueError("发布目录已有真实权重，不得静默覆盖")
    # Preserve the failed startup attempt's config before installing artifacts.
    if (destination / "run_config.json").exists():
        shutil.copy2(destination / "run_config.json", ROOT / "runs/v1-failed-startup-run-config.json")
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(selected / filename, destination / filename)
    source_hash = hashlib.file_digest((selected / "adapter_model.safetensors").open("rb"), "sha256").hexdigest()
    installed_hash = hashlib.file_digest((destination / "adapter_model.safetensors").open("rb"), "sha256").hexdigest()
    if source_hash != installed_hash:
        raise ValueError("安装后的 adapter 权重校验不一致")
    metadata = json.loads((source / "witrans_adapter.json").read_text(encoding="utf-8"))
    metadata.update(selected_by="minimum development loss", selected_checkpoint=name,
                    selected_dev_loss=loss, installed_at=now(),
                    adapter_sha256=installed_hash,
                    data_scope="Controlled synthetic v1, 75 training families, 3000 rows",
                    quality_status="Test-screen acceptance pending")
    write_json(destination / "witrans_adapter.json", metadata)
    write_json(ROOT / "runs/v1-checkpoint-selection.json", {
        "at": now(), "criterion": "Development loss only; no test results used",
        "candidates": [{"dev_loss": item[0], "checkpoint": item[1]} for item in options],
        "selected_checkpoint": name, "selected_dev_loss": loss,
        "adapter_sha256": installed_hash,
        "source": str(selected), "installed_adapter": str(destination)})
    print({"checkpoint": name, "dev_loss": loss, "adapter": str(destination)})


if __name__ == "__main__":
    main()
