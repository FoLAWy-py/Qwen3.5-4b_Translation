"""CPU-only LoRA roundoff feasibility; does not change any runtime or saved weights."""
import hashlib
import math
from pathlib import Path

from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, write_json


def main():
    dest=Path('runs/qwen35-v3-lora-bf16-cpu-diagnosis.json')
    assert not dest.exists()
    adapter=Path('models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors')
    expected='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
    with adapter.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    import torch
    from safetensors.torch import load_file
    tensors=load_file(adapter,device='cpu')
    assert len(tensors)==496 and all(t.dtype==torch.float32 for t in tensors.values())
    rows=[]
    total_norm_sq=total_error_sq=0.
    for name,tensor in tensors.items():
        assert 'lora_A' in name or 'lora_B' in name
        assert tensor.device.type=='cpu' and torch.isfinite(tensor).all()
        rounded=tensor.to(torch.bfloat16).to(torch.float32)
        norm_sq=float(tensor.double().square().sum())
        error_sq=float((rounded.double()-tensor.double()).square().sum())
        total_norm_sq+=norm_sq;total_error_sq+=error_sq
        rows.append(dict(name=name,elements=tensor.numel(),original_dtype=str(tensor.dtype),
            simulated_dtype='torch.bfloat16',norm_sq=norm_sq,error_sq=error_sq,
            relative_l2=math.sqrt(error_sq/norm_sq) if norm_sq else 0.,
            maximum_absolute_error=float((rounded-tensor).abs().max())))
    package=Path('.venv-qwen35/Lib/site-packages/peft')
    bnb=package/'tuners/lora/bnb.py'
    peft_model=package/'peft_model.py'
    bnb_source=bnb.read_text(encoding='utf-8');peft_source=peft_model.read_text(encoding='utf-8')
    assert 'x = self._cast_input_dtype(x, lora_A.weight.dtype)' in bnb_source
    assert 'output = output.to(expected_dtype)' in bnb_source
    assert 'autocast_adapter_dtype: bool = True' in peft_source
    profile=load('runs/qwen35-v3-diagnostic-recovery1/operators.json')
    assert profile['lora_dtypes']=={'torch.float32':496}
    diagnosis=load('runs/qwen35-v3-diagnostic-recovery1/diagnosis.json')
    with adapter.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    write_json(dest,dict(at=now(),adapter_sha256=expected,tensor_count=len(rows),
        parameter_count=sum(r['elements'] for r in rows),relative_roundtrip_l2=math.sqrt(total_error_sq/total_norm_sq),
        maximum_tensor_relative_l2=max(r['relative_l2'] for r in rows),
        simulated_storage_bytes=dict(fp32=sum(r['elements'] for r in rows)*4,bfloat16=sum(r['elements'] for r in rows)*2),
        tensor_diagnostics=rows,profile_hash=fingerprint(profile),timing_diagnosis_hash=fingerprint(diagnosis),
        local_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (bnb,peft_model)},
        source_observation='Installed PEFT NF4 inference casts activation to LoRA weight dtype and result back when autocast is disabled; measured inherited LoRA tensors are FP32. This suggests an explicitly fingerprinted BF16 LoRA-only runtime experiment, not a demonstrated bottleneck attribution.',
        candidate_runtime_hypothesis='After an eligible candidate exists and the GPU is free, test one LoRA-only BF16 inference change against that same candidate eager FP32 runtime on the full fixed24 inputs. No base merge or requantization, no dependencies changed. Any raw-output differences require individual semantic/runtime review and full known200/public116 gates before final freeze.',
        experiment_frozen=False,final_candidate_sha_pending=True,saved_weights_unchanged=True,
        gpu_model_loaded=False,gpu_executed=False,optimizer_updates=0,
        limitations='CPU weight rounding is not output equivalence, quality, latency or candidate-ownweight resource evidence. Starting-weight diagnosis only; no runtime change made or acceptance inferred.',
        stage_goal_complete=False,release_approved=False,default_promoted=False))
    print(dict(diagnosis=str(dest),tensors=len(rows),relative_roundtrip_l2=math.sqrt(total_error_sq/total_norm_sq),gpu_executed=False),flush=True)


if __name__=='__main__':
    main()
