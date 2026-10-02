from __future__ import annotations

import importlib.metadata
import json
import platform

from .common import load_env, now, write_json


def doctor(args):
    import os
    import torch
    import bitsandbytes as bnb

    load_env()
    report = {"at": now(), "python": platform.python_version(), "platform": platform.platform(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface-hub")},
              "credentials_present": {name: bool(os.getenv(name)) for name in
                                      ("DEEP_INFRA_APIKEY", "HF_Access_Token")},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        write_json(args.output, report)
        raise RuntimeError("CUDA 不可用，报告已保存")
    report["gpu"] = torch.cuda.get_device_name(0)
    report["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    report["bf16_supported"] = torch.cuda.is_bf16_supported()
    layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16,
                             quant_type="nf4", compress_statistics=True).to("cuda")
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("NF4 反向传播测试失败")
    parameter = torch.nn.Parameter(torch.randn(8192, device="cuda"))
    optimizer = bnb.optim.AdamW8bit([parameter], lr=1e-4)
    parameter.square().mean().backward()
    optimizer.step()
    report["nf4_forward_backward"] = "passed"
    report["adamw8bit_step"] = "passed"
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
