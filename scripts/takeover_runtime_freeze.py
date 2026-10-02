"""Freeze the chosen runtime before independent-source authoring; old gates remain pending."""
import hashlib
import json
from pathlib import Path

from witrans_tools.common import fingerprint,now,write_json
from witrans_tools.qwen35 import validate_identity,validate_runtime_files,DEFAULT_ADAPTER


def main():
    dest=Path('runs/takeover-20261002/runtime-freeze.json')
    if dest.exists():raise FileExistsError(dest)
    receipt,digest=validate_identity('models/Qwen3.5-4B',DEFAULT_ADAPTER,1024)
    validate_runtime_files('models/Qwen3.5-4B',DEFAULT_ADAPTER)
    root=Path('runs/takeover-20261002/decode-final')
    plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if summary['plan_hash']!=fingerprint(plan) or not summary['performance_passed']:
        raise ValueError('Final own-weight formal performance gate not complete')
    for path,digest_file in plan['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest_file:
            raise ValueError('Runtime source changed after final timing: '+path)
    config=dict(base_revision=receipt['revision'],adapter_sha256=digest,prompt_hash=plan['prompt_hash'],
        quantization='nf4',double_quant=True,compute_dtype='bfloat16',lora_dtype='float32',attention='sdpa',
        runtime='decode_compiled',cache='static',max_cache_len=1024,
        compile=plan['compile'],decoding=dict(do_sample=False,num_beams=1,thinking=False,max_length=1024,
            quality_max_new_tokens=256,performance_max_new_tokens=128),dependencies=plan['dependencies'])
    write_json(dest,dict(at=now(),runtime_frozen=True,configuration=config,runtime_fingerprint=fingerprint(config),
        source_hashes=plan['source_hashes'],uv_lock_sha256=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest(),
        performance_plan_hash=fingerprint(plan),performance_summary_hash=fingerprint(summary),
        old_semantic_regression='generation/review pending; independent model test entry blocked until complete',
        independent_source_authoring_allowed=True,independent_generation_allowed=False,
        source_authoring='Separate blind process may read only source/reference and old-source exclusion catalog, never candidate test outputs.',
        configuration_changes_after_test='Invalidate the independent status if prompted by new test errors; new independent sources then required.',
        release_approved=False))
    print('Frozen decode-only NF4/FP32-LoRA configuration; independent generation still gated on old semantic review.')


if __name__=='__main__':main()
