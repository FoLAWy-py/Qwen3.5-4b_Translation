"""Inventory only the retained project for publication; never uploads or deletes."""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.prepare_local_result_cleanup import BEST, EXPECTED, keep_run

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {".git", ".cache", "__pycache__", ".pytest_cache", ".venv", ".venv-qwen35",
          ".publication-staging", ".aws", ".codex", ".agents", "history"}
ROOT_FILES = {".gitignore", ".gitattributes", ".python-version", "README.md",
              "requirements-qwen35.txt", "pyproject.toml", "uv.lock", "witrans.py",
              "witrans-4b-model-spec-v1.3.md", "env_example.md"}
BASE = ROOT / "models/Qwen3.5-4B"


def ignored(path):
    return (any(p in IGNORE for p in path.relative_to(ROOT).parts)
        or path.is_symlink() or path.name == ".env" or path.name.startswith(".env.")
        or path.suffix.lower() in {".pyc", ".pem", ".key", ".part", ".download", ".chunk"})


def main():
    candidates = [ROOT / name for name in sorted(ROOT_FILES) if (ROOT / name).is_file()]
    for folder in ("data", "docs", "examples", "scripts", "tests", "witrans_tools", BEST):
        candidates.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    candidates.extend(p for p in BASE.iterdir()
        if p.is_file() and p.suffix.lower() in {".json", ".jinja", ".txt", ".md"})
    candidates.extend(p for p in (ROOT / "runs").rglob("*") if p.is_file() and keep_run(p))
    rows = []
    totals = Counter()
    for path in sorted(set(candidates)):
        if ignored(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        size = path.stat().st_size
        if size > 100 * 1024**2 and rel != BEST + "/adapter_model.safetensors":
            raise ValueError(f"Unexpected large publication file: {rel}")
        transport = "git-lfs" if rel == BEST + "/adapter_model.safetensors" else "git"
        rows.append(dict(path=rel, bytes=size, transport=transport))
        totals["files"] += 1
        totals["bytes"] += size
        totals[transport + "_bytes"] += size
    with (ROOT / BEST / "adapter_model.safetensors").open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert sha == EXPECTED
    links = []
    for filename in ("README.md", "docs/experiment-report.md", "docs/publication-plan.md"):
        doc = ROOT / filename
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8-sig")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (doc.parent / target.split("#")[0]).resolve()
            assert resolved.is_relative_to(ROOT)
            assert resolved.exists(), f"Broken link: {filename} -> {target}"
            links.append(dict(source=filename, target=target, exists=True))
    receipt = ROOT / "publication/local-cleanup-receipt.json"
    cleanup = json.loads(receipt.read_text(encoding="utf-8-sig")) if receipt.exists() else None
    report = dict(at=datetime.now(timezone.utc).isoformat(),
        target_repository="https://github.com/FoLAWy-py/Qwen3.5-4b_Translation.git",
        commit_name="FoLAWy-py", commit_email="ljp2219819716@gmail.com",
        status="awaiting_readme_review", upload_executed=False,
        archive_old_results=False, best_adapter_sha256=sha,
        cleanup_status=cleanup["status"] if cleanup else "not_executed",
        totals=dict(totals), files=rows, local_links=links,
        excluded="Historical weights/checkpoints/logs; secrets; virtual environments; official base binaries; caches; local cleanup manifests.",
        hash_scope="Best adapter checked. Refresh all publication file hashes before actual upload.")
    (ROOT / "publication/upload-inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(totals=dict(totals), links=len(links),
        cleanup_status=report["cleanup_status"], upload_executed=False),ensure_ascii=False))


if __name__ == "__main__":
    main()

