from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "models" / "Qwen3-4B"
MODEL_ID = "Qwen/Qwen3-4B"
TEACHER_ID = "Qwen/Qwen3-32B"


def load_env():
    load_dotenv(ROOT / ".env", override=False)


def secret(name):
    load_env()
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"请在 .env 配置 {name}")
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: 非法 JSON") from exc
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def base_manifest(base_dir):
    path = Path(base_dir) / "witrans_base.json"
    if not path.is_file():
        raise ValueError("基础模型缺少 revision 记录；先执行 download")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or not manifest.get("revision"):
        raise ValueError("基础模型必须是固定 revision 的原始 Qwen/Qwen3-4B")
    return manifest
