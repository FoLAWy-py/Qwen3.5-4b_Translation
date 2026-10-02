from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from witrans_tools.common import MODEL_ID, now, secret, write_json


def download(args):
    token = secret("HF_Access_Token")
    destination = Path(args.base_dir)
    lock_path = destination / "witrans_base.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["model_id"] != MODEL_ID:
            raise ValueError("已存在其他基础模型")
        revision = lock["revision"]
        if args.revision and args.revision != revision:
            raise ValueError("目录已锁定 revision；不同 revision 请使用新目录")
    else:
        revision = HfApi(token=token).model_info(MODEL_ID, revision=args.revision or "main").sha
        write_json(lock_path, {"model_id": MODEL_ID, "revision": revision,
                              "created_at": now(), "status": "downloading"})
    print(f"下载 {MODEL_ID}@{revision} -> {destination}", flush=True)
    snapshot_download(MODEL_ID, revision=revision, token=token, local_dir=destination,
                      max_workers=2, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.md", "LICENSE*"])
    write_json(lock_path, {"model_id": MODEL_ID, "revision": revision,
                          "completed_at": now(), "status": "complete"})
    print("基础模型下载完成")
