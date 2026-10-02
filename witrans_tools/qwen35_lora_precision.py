"""Explicit experimental inference-only LoRA cast; never used by default loaders.

This does not save/merge/requantize weights or establish output equivalence.
Changed inference precision requires an explicit runtime fingerprint, own-weight
timing and full semantic regression before a final candidate can be frozen.
"""
import re

from .common import fingerprint


_LORA_WEIGHT = re.compile(r'(?:^|\.)lora_[AB]\.default\.weight$')


def _cast_inference_lora(model, *, expected_device, expected_tensors, expected_parameters):
    """Internal shared implementation; CPU use is solely for small unit fixtures."""
    import torch
    parameters = list(model.named_parameters())
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Inference precision cast requires eval mode throughout')
    if any(parameter.requires_grad for _, parameter in parameters):
        raise ValueError('Inference precision cast must not touch a trainable model')
    if not parameters or any(parameter.device.type != expected_device for _, parameter in parameters):
        raise ValueError('Parameter offload or unexpected parameter device')
    lora = [(name, parameter) for name, parameter in parameters if _LORA_WEIGHT.search(name)]
    other_lora = [name for name, _ in parameters if 'lora_' in name and not _LORA_WEIGHT.search(name)]
    if other_lora:
        raise ValueError('Unexpected LoRA parameter format or additional adapter')
    if len(lora) != expected_tensors or sum(p.numel() for _, p in lora) != expected_parameters:
        raise ValueError('LoRA inventory differs from frozen adapter architecture')
    if any(p.dtype != torch.float32 for _, p in lora):
        raise ValueError('Expected the original FP32 LoRA inference parameters')
    # Validate the complete inventory before mutating any parameter. Preserve
    # base tensors, including NF4 subclasses and quantization metadata, in place.
    chosen = {name for name, _ in lora}
    base = [(name, p) for name, p in parameters if name not in chosen]
    def identity(p):
        return (id(p), p.data_ptr(), str(p.dtype), tuple(p.shape), str(p.device), p.requires_grad)
    base_before = {name: identity(p) for name, p in base}
    before = [dict(name=name, dtype=str(p.dtype), shape=list(p.shape), elements=p.numel()) for name, p in lora]
    with torch.no_grad():
        for _, p in lora:
            p.data = p.data.to(dtype=torch.bfloat16)
    if {name: identity(p) for name, p in base} != base_before:
        raise RuntimeError('Base parameter identity changed during LoRA-only conversion')
    if any(p.dtype != torch.bfloat16 or p.requires_grad or p.device.type != expected_device for _, p in lora):
        raise RuntimeError('LoRA conversion postcondition failed')
    after = [dict(name=name, dtype=str(p.dtype), shape=list(p.shape), elements=p.numel()) for name, p in lora]
    return dict(operation='inference_only_fp32_lora_to_bfloat16', tensors=len(lora),
        parameters=sum(p.numel() for _, p in lora), before_hash=fingerprint(before), after_hash=fingerprint(after),
        base_parameter_identity_preserved=True, saved_weights_modified=False,
        gpu_executed=expected_device == 'cuda', quality_or_speed_proven=False,
        requires_full_runtime_quality_and_own_weight_timing=True)


def cast_qwen35_lora_to_bfloat16(model):
    """Opt-in cast for the frozen Qwen35 rank16 architecture, on CUDA only."""
    return _cast_inference_lora(model, expected_device='cuda',
        expected_tensors=496, expected_parameters=32464896)
