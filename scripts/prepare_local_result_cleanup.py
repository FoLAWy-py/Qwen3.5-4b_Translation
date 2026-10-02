"""Build a bounded, read-only plan for user-authorized old-result removal."""
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
BEST = "models/witrans-qwen35-v2-critical-cpo"
EXPECTED = "fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27"
PROTECTED_MODELS = {BEST, "models/Qwen3.5-4B", "models/Qwen3-4B"}
KEEP_EXACT = {
    "qwen35-acquisition.json", "qwen35-v3-verified-start-audit.json",
    "qwen35-v2-model-comparison.json", "qwen35-v2-model-comparison.md",
    "qwen35-v2-experiment-audit.json",
    "qwen35-v3-confirmation-protocol.json",
    "qwen35-v3-candidate-timing-input-preflight.json",
    "qwen35-v3-lexical-negative-correction.json",
    "qwen35-v3-sweetener-negative-correction.json",
    "qwen35-v3-supplement-reference-age-precision-audit.json",
    "qwen35-v3-context-supplement-final-audit.json",
    "qwen35-v3-lexical-correction-binding-audit.json",
    "qwen35-v3-reviewed23-gradient-diagnosis.json",
    "qwen35-v3-train-discovery-recovery1.jsonl",
    "qwen35-v3-train-discovery-recovery1.summary.json",
    "qwen35-v3-train-discovery-focused-corrected-manual.jsonl",
    "qwen35-v3-train-discovery-lexical-corrected-manual.jsonl",
    "qwen35-v3-train-discovery-sweetener-corrected-manual.jsonl",
    "qwen35-v3-focused32-negative-final-audit.json",
    "qwen35-v3-focused32-negative-effective-readings.jsonl",
    "qwen35-v3-focused32-negative-quality-rereview.jsonl",
}
PERFORMANCE = {"plan.json", "performance-inputs.jsonl", "performance.jsonl", "performance-summary.json"}
MODEL_KEYS = ("scripts.train_", "scripts.evaluate_qwen35", "scripts.run_v",
              "scripts.decode_qwen35", "scripts.benchmark_qwen35", "scripts.probe_qwen35")


def keep_run(path):
    rel = path.relative_to(ROOT).as_posix()
    if path.name in KEEP_EXACT:
        return True
    if rel.startswith("runs/qwen35-v3-diagnostic-recovery1/"):
        return path.name in PERFORMANCE
    return any(path.name == stem + suffix
        for stem in ("qwen35-finetuned-known", "qwen35-finetuned-public")
        for suffix in (".jsonl", ".summary.json", "-manual.jsonl", "-semantic.jsonl"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path):
    assert path.resolve().is_relative_to(ROOT)
    assert not path.is_symlink()
    return dict(path=path.relative_to(ROOT).as_posix(),
                bytes=path.stat().st_size, sha256=digest(path))


def assert_idle():
    for proc in psutil.process_iter(["name"]):
        if (proc.info.get("name") or "").lower() not in {"python.exe", "uv.exe"}:
            continue
        try:
            cmd = " ".join(proc.cmdline())
            if proc.pid != os.getpid() and any(key in cmd for key in MODEL_KEYS):
                raise RuntimeError(f"Active model process {proc.pid}; do not delete")
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError(f"Cannot inspect Python/uv process {proc.pid}") from exc


def main():
    dest = ROOT / "publication/local-cleanup-plan.json"
    if dest.exists():
        raise FileExistsError("Cleanup plan already exists")
    assert_idle()
    assert digest(ROOT / BEST / "adapter_model.safetensors") == EXPECTED
    model_dirs = [p for p in sorted((ROOT / "models").iterdir())
                  if p.is_dir() and p.relative_to(ROOT).as_posix() not in PROTECTED_MODELS]
    dir_records = []
    for directory in model_dirs:
        assert not directory.is_symlink()
        paths = sorted(p for p in directory.rglob("*") if p.is_file())
        dir_records.append(dict(path=directory.relative_to(ROOT).as_posix(),
                                files=[file_record(p) for p in paths]))
    delete_files, keep_files = [], []
    for p in sorted((ROOT / "runs").rglob("*")):
        if p.is_file():
            (keep_files if keep_run(p) else delete_files).append(file_record(p))
    extra = ["v6-cpo-semantic.jsonl", "v6-cpo-semantic.summary.json", "v6-cpo-unmatched.jsonl",
             "docs/history/README-before-qwen35-20261002.md"]
    for rel in extra:
        p = ROOT / rel
        if p.exists():
            delete_files.append(file_record(p))
    metrics = json.loads((ROOT / "models/witrans-qwen35-v3-reviewed23-sft/metrics.json").read_text(encoding="utf-8"))
    readings = [json.loads(line) for line in (ROOT / "runs/qwen35-v3-reviewed23-sft-recall-manual.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    history = dict(candidate=metrics["adapter_sha256"], training_seconds=metrics["seconds"],
        peak_reserved_gib=metrics["peak_reserved_gib"], reviewed=len(readings),
        counts=dict(Counter(r["verdict"] for r in readings)),
        status="TRAIN gate failed; no DEV or formal candidate timing",
        original_errors={"pass":4,"minor":5,"major":14},
        preservation={"pass":38,"minor":2,"major":0})
    all_deletes = delete_files + [f for d in dir_records for f in d["files"]]
    plan = dict(at=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
        user_authorization="不上传过去训练结果了，会占用大量git LFS存储，没有实质意义，本地也删除",
        archive_old_results=False, best_adapter=BEST, best_sha256=EXPECTED,
        protected_model_directories=sorted(PROTECTED_MODELS),
        remove_model_directories=dir_records, remove_files=delete_files,
        retained_run_files=keep_files, latest_failed_trial_summary=history,
        delete_file_count=len(all_deletes), delete_bytes=sum(r["bytes"] for r in all_deletes),
        best_file=file_record(ROOT / BEST / "adapter_model.safetensors"),
        deletion_executed=False)
    dest.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(model_directories=len(dir_records), delete_files=plan["delete_file_count"],
        delete_bytes=plan["delete_bytes"], retained_run_files=len(keep_files),
        best_sha256=EXPECTED, model_processes=[]),ensure_ascii=False))


if __name__ == "__main__":
    main()

