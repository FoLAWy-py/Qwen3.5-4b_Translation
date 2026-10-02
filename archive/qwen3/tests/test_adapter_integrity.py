import json

import pytest

from archive.qwen3.runtime import SYSTEM_PROMPT, WiTrans, _LocalTranslator
from witrans_tools.common import fingerprint


def test_modified_selected_adapter_is_rejected_before_gpu_load(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    for path in (base / "config.json", adapter / "adapter_config.json"):
        path.write_text("{}", encoding="utf-8")
    (base / "witrans_base.json").write_text(json.dumps({"model_id": "Qwen/Qwen3-4B", "revision": "test"}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"changed weight contents")
    (adapter / "witrans_adapter.json").write_text(json.dumps({
        "base_model_id": "Qwen/Qwen3-4B", "base_revision": "test", "prompt_hash": fingerprint(SYSTEM_PROMPT),
        "adapter_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="权重指纹"):
        WiTrans(str(base), str(adapter))


def test_bf16_rejects_unverified_budget_before_loading_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="1024 tokens"):
        _LocalTranslator(str(tmp_path), None, max_length=2048, quantization="bf16_cpu_vocab")
