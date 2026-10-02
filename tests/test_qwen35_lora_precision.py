"""CPU fixture checks only; never claimed as CUDA/output/latency validation."""
import pytest
import torch

from witrans_tools.qwen35_lora_precision import _cast_inference_lora, cast_qwen35_lora_to_bfloat16


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4, bias=False)
        self.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(4, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 4, bias=False)})
        self.requires_grad_(False)
        self.eval()


def cpu_cast(model, **kwargs):
    return _cast_inference_lora(model, expected_device='cpu',
        expected_tensors=kwargs.get('tensors', 2), expected_parameters=16)


def snapshot(model):
    return {name: (p.data_ptr(), p.dtype, p.detach().clone()) for name, p in model.named_parameters()}


def assert_untouched(model, saved):
    for name, p in model.named_parameters():
        pointer, dtype, value = saved[name]
        assert p.data_ptr() == pointer and p.dtype == dtype and torch.equal(p, value)


def test_cast_only_adapter_and_preserve_base_storage_and_values():
    model = Fixture()
    saved = snapshot(model)
    report = cpu_cast(model)
    assert model.base.weight.data_ptr() == saved['base.weight'][0]
    assert torch.equal(model.base.weight, saved['base.weight'][2])
    assert model.base.weight.dtype == torch.float32
    for name in ('lora_A.default.weight', 'lora_B.default.weight'):
        p = dict(model.named_parameters())[name]
        assert p.dtype == torch.bfloat16
        assert torch.equal(p, saved[name][2].to(torch.bfloat16)) and not p.requires_grad
    assert report['base_parameter_identity_preserved'] and not report['gpu_executed']
    assert report['quality_or_speed_proven'] is False and not report['saved_weights_modified']


@pytest.mark.parametrize('kind', ('training', 'trainable', 'inventory', 'dtype', 'other_adapter'))
def test_unsafe_models_rejected_before_any_partial_mutation(kind):
    model = Fixture()
    if kind == 'training':
        model.lora_B.train()
    elif kind == 'trainable':
        model.lora_A.default.weight.requires_grad_(True)
    elif kind == 'dtype':
        model.lora_B.default.weight.data = model.lora_B.default.weight.data.to(torch.bfloat16)
    elif kind == 'other_adapter':
        model.lora_A['extra'] = torch.nn.Linear(4, 2, bias=False).requires_grad_(False).eval()
    saved = snapshot(model)
    with pytest.raises(ValueError):
        cpu_cast(model, tensors=3 if kind == 'inventory' else 2)
    assert_untouched(model, saved)


def test_public_cuda_only_entry_rejects_cpu_without_modification():
    model = Fixture()
    saved = snapshot(model)
    with pytest.raises(ValueError, match='device'):
        cast_qwen35_lora_to_bfloat16(model)
    assert_untouched(model, saved)
