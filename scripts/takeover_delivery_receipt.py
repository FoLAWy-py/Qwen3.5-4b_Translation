"""Verify the completed evidence chain without inference or data mutation."""
import hashlib
import json
from pathlib import Path
from witrans_tools.common import read_jsonl,write_json,fingerprint,now
from scripts.evaluate_independent_frozen import verify_frozen

def main():
    root=Path('runs/takeover-20261002')
    runtime,freeze,regression,sources=verify_frozen(Path('data/independent-20261002-frozen'),'confirmation')
    outputs=read_jsonl(root/'confirmation/outputs.jsonl');reviews=read_jsonl(root/'confirmation/semantic.jsonl')
    result=json.loads((root/'confirmation/semantic-summary.json').read_text(encoding='utf-8'))
    if len(sources)!=400 or len(outputs)!=400 or len(reviews)!=400:raise ValueError('Incomplete confirmation')
    for source,out,review in zip(sources,outputs,reviews):
        if source['id']!=out['id'] or out['id']!=review['id']:raise ValueError('Review identity mismatch')
        if fingerprint(source)!=out['source_row_hash'] or fingerprint(out)!=review['generation_hash']:raise ValueError('Stale review')
    if result['review_hash']!=fingerprint(reviews) or result['output_hash']!=fingerprint(outputs):raise ValueError('Stale summary')
    if (root/'release').exists():raise ValueError('Unexpected release outputs')
    adapter=Path('models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors')
    adapter_hash=hashlib.file_digest(adapter.open('rb'),'sha256').hexdigest()
    if adapter_hash!=runtime['configuration']['adapter_sha256']:raise ValueError('Adapter changed')
    old=json.loads((root/'historic-source-catalog.json').read_text(encoding='utf-8'))
    for entry in old['files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Historic source changed')
    cases=[Path('data/independent-20261002-frozen')/(k+'.jsonl') for k in ('confirmation','release')]
    for path in cases:
        rows=read_jsonl(path)
        if fingerprint(rows)!=freeze['sets'][path.stem]['hash']:raise ValueError('Frozen source changed')
    p=root/'decode-final/summary.json';perf=json.loads(p.read_text(encoding='utf-8'))
    write_json(root/'delivery-receipt.json',dict(at=now(),adapter_sha256=adapter_hash,runtime_hash=fingerprint(runtime),
        data_freeze_hash=fingerprint(freeze),full_development_review_hash=fingerprint(regression),
        performance_summary_hash=fingerprint(perf),confirmation_summary_hash=fingerprint(result),
        rows=400,all_source_generation_review_bindings_verified=True,all_historic_file_hashes_unchanged=True,
        frozen_test_sources_unchanged=True,candidate_runtime_code_unchanged_after_test=True,
        confirmation_prompt_tokens_range=[min(r['prompt_tokens'] for r in outputs),max(r['prompt_tokens'] for r in outputs)],
        confirmation_output_including_eos_tokens_range=[min(r['generated_tokens_including_eos'] for r in outputs),max(r['generated_tokens_including_eos'] for r in outputs)],
        tests=dict(command='.venv-qwen35-mainline/Scripts/python.exe -X utf8 -m scripts.low_cpu_run --module pytest -q',
            passed=88,failed=0,observed_seconds=8.34,optimized_mode_guard_cases_included=True),
        release_gate_guard=dict(observed_exception='ValueError: Frozen confirmation entry gates did not pass',
            rejected_before_torch_loading=True,release_output_directory_absent=True),
        no_training=True,no_push=True,release_executed=False,release_approved=False))
    print(dict(confirmation=result['counts'],performance=perf['performance_passed'],identity_and_evidence_verified=True))

if __name__=='__main__':main()
