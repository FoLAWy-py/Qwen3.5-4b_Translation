"""witrans-4b 的 Python 本地推理入口（不含模型权重）。

需要训练得到的 adapter，以及与训练时相同的 Qwen3-4B 本地目录。
使用 CUDA、Transformers、bitsandbytes 与 PEFT；不会启动服务或调用云模型。
训练与评测进度参见 README.md 和 runs/ 中的实测报告。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from witrans_tools.protocol import SYSTEM_PROMPT, make_messages, parse_translation

from witrans_tools.runtime import GenerationRuntime


class _LocalTranslator(GenerationRuntime):
    """共享本地生成实现；无 adapter 的实例仅用于明确标识的基线评测。"""

    def __init__(self, base_dir: str, adapter_dir: str | None,
                 max_length: int = 2048, *, quantization: str | None = None) -> None:
        base = Path(base_dir)
        adapter = Path(adapter_dir) if adapter_dir is not None else None
        self.adapter_sha256 = None
        if not (base / "config.json").is_file():
            raise FileNotFoundError("base_dir 必须指向已下载的基础模型目录")
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError("需要实际训练得到的 adapter，不能只用基础模型冒充")
        if adapter is not None and not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("缺少 adapter_model.safetensors")
        if adapter is not None:
            import hashlib
            metadata_path = adapter / "witrans_adapter.json"
            if not metadata_path.is_file() or not (base / "witrans_base.json").is_file():
                raise ValueError("缺少基础模型或 adapter revision 记录")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            base_metadata = json.loads((base / "witrans_base.json").read_text(encoding="utf-8"))
            if metadata.get("smoke_only"):
                raise ValueError("此 adapter 仅用于 smoke 测试，不能作为 witrans-4b 使用")
            if (metadata.get("base_model_id") != "Qwen/Qwen3-4B"
                    or metadata.get("base_revision") != base_metadata.get("revision")
                    or base_metadata.get("model_id") != "Qwen/Qwen3-4B"):
                raise ValueError("基础模型与 adapter 的 revision 不一致")
            prompt_hash = hashlib.sha256(json.dumps(SYSTEM_PROMPT, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if metadata.get("prompt_hash") != prompt_hash:
                raise ValueError("提示词与训练时不一致")
            with (adapter / "adapter_model.safetensors").open("rb") as stream:
                self.adapter_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if metadata.get("adapter_sha256") is not None and metadata["adapter_sha256"] != self.adapter_sha256:
                raise ValueError("adapter 权重指纹与选择记录不一致")
        if type(max_length) is not int or max_length < 256:
            raise ValueError("max_length 必须是至少 256 的整数")
        if quantization is None:
            # User-selected deployment policy. Historical experiment metadata
            # must not silently switch the runtime to INT8 or BF16.
            quantization = "nf4"
        if quantization not in ("nf4", "int8", "bf16_cpu_vocab"):
            raise ValueError("quantization 必须为 nf4、int8 或 bf16_cpu_vocab")
        if quantization == "bf16_cpu_vocab" and max_length > 1024:
            raise ValueError("bf16_cpu_vocab 当前仅验证至 1024 tokens；请显式设置 max_length=1024 或选择 int8/nf4")
        self.quantization = quantization

        import torch
        from peft import PeftModel
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, GenerationConfig)

        if not torch.cuda.is_available():
            raise RuntimeError("此参考入口需要可用的 NVIDIA CUDA 显卡")
        self._torch = torch
        self._generation_config = GenerationConfig
        self.max_length = max_length
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(base), local_files_only=True, trust_remote_code=False)
        if quantization == "bf16_cpu_vocab" and dtype != torch.bfloat16:
            raise RuntimeError("bf16_cpu_vocab 需要 BF16 CUDA 支持")
        quant_config = None if quantization == "bf16_cpu_vocab" else (
            BitsAndBytesConfig(load_in_8bit=True) if quantization == "int8" else BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=dtype,
            device_map=None if quant_config is None else {"": 0},
            attn_implementation="sdpa",
            quantization_config=quant_config,
        )
        self.model = PeftModel.from_pretrained(
            base_model, str(adapter), is_trainable=False,
            local_files_only=True).eval() if adapter is not None else base_model.eval()
        if quantization == "bf16_cpu_vocab":
            _place_bf16_cpu_vocab(base_model)
        self.model.config.use_cache = True




def _place_bf16_cpu_vocab(base_model):
    """Keep tied vocabulary weights on CPU, including computation, not offload."""
    import torch
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    if base_model.config.model_type != "qwen3":
        raise ValueError("bf16_cpu_vocab 只支持本项目固定的 Qwen3 结构")
    for name, module in base_model.model.named_children():
        if name != "embed_tokens":
            module.to("cuda:0")
    for module in (base_model.get_input_embeddings(), base_model.get_output_embeddings()):
        add_hook_to_module(module, AlignDevicesHook(
            execution_device=torch.device("cpu"), io_same_device=True))


class WiTrans(_LocalTranslator):
    """一个真实训练的 adapter，统一翻译所有场景。"""

    def __init__(self, base_dir: str, adapter_dir: str, max_length: int = 2048, *, quantization: str | None = None):
        if adapter_dir is None or not str(adapter_dir).strip():
            raise ValueError("WiTrans 必须加载真实 adapter")
        super().__init__(base_dir, adapter_dir, max_length, quantization=quantization)
