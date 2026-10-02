"""Recompute complete frozen three-round timing and enforce token eligibility."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from witrans import SYSTEM_PROMPT, make_messages, parse_translation
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.root);dest=Path(args.output)
    assert not dest.exists()
    plan=load(root/'plan.json');summary=load(root/'performance-summary.json')
    rows=read_jsonl(root/'performance.jsonl');inputs=read_jsonl(root/'performance-inputs.jsonl')
    by_id={r['id']:r for r in inputs}
    assert len(by_id)==len(inputs) and fingerprint(inputs)==plan['performance_inputs_hash']
    assert summary['plan_hash']==fingerprint(plan) and summary['adapter_sha256']==plan['adapter_sha256']
    with (Path(plan['adapter_dir'])/'adapter_model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['adapter_sha256']
    assert plan['prompt_hash']==fingerprint(SYSTEM_PROMPT)
    assert plan['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=128)
    assert plan['quantization']==dict(type='nf4',double_quant=True,compute_dtype='bfloat16')
    assert plan['warmup_rounds']>=1 and plan['measured_rounds']>=3
    assert len(rows)==summary['calls']==len(inputs)*plan['measured_rounds']
    assert {(r['id'],r['repeat']) for r in rows}=={(rid,i) for rid in by_id for i in range(plan['measured_rounds'])}
    runtime_verified=True;warmup=[];baseline=[]
    if 'runtime' in plan:
        assert summary['runtime_fingerprint']==plan['runtime_fingerprint']==fingerprint(plan['runtime'])
        assert summary['raw_measurements_hash']==fingerprint(rows)
        warmup=read_jsonl(root/'warmup.jsonl')
        assert summary['warmup_hash']==fingerprint(warmup)
        assert summary['warmup_calls']==len(warmup)==len(inputs)*plan['warmup_rounds']
        assert [(r['id'],r['repeat']) for r in warmup]==[(r['id'],-1) for r in inputs]
        assert all(r['input']==by_id[r['id']]['input'] and r['ended'] is True
            and r['valid_json'] and r['cpu_parameter_count']==0 for r in warmup)
        for row in warmup:
            assert parse_translation(row['raw'])==row['prediction']
        assert all(r['cpu_parameter_count']==0 for r in rows)
        assert summary['peak_reserved_gib']>=max(r['reserved_gib'] for r in warmup)
        if plan['runtime']['mode']=='compiled':
            baseline=read_jsonl(root/'eager-baseline.jsonl')
            assert [r['id'] for r in baseline]==[r['id'] for r in inputs]
            assert all(r['input']==by_id[r['id']]['input'] and r['ended'] is True
                and parse_translation(r['raw'])==r['prediction'] for r in baseline)
            eager={r['id']:r['raw'] for r in baseline}
            runtime_verified=all(r['raw']==eager[r['id']] for r in warmup+rows)
            assert runtime_verified==summary['compiled_matches_all_eager_baselines']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    eligible=True;all_json=True;all_eos=True
    for row in rows+warmup+baseline:
        assert row['input']==by_id[row['id']]['input'] and row['seconds']>0
        body=len(tokenizer.encode(row['input']['text'],add_special_tokens=False))
        prompt=tokenizer.apply_chat_template(make_messages(**row['input']),tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if hasattr(prompt,'keys'):prompt=prompt['input_ids']
        assert body==row['body_tokens'] and len(prompt)==row['prompt_tokens']
        eligible=eligible and 20<=body<=120 and len(prompt)+128<=1024 and row['generated_tokens_including_eos']<=128
        try: parse_translation(row['raw']);valid=True
        except ValueError: valid=False
        assert valid==row['valid_json']
        all_json=all_json and valid;all_eos=all_eos and row['ended'] is True
    times=[r['seconds'] for r in rows]
    mean=statistics.mean(times);p95=sorted(times)[math.ceil(.95*len(times))-1]
    assert abs(mean-summary['mean_seconds'])<1e-8 and abs(p95-summary['p95_seconds'])<1e-8
    assert summary['peak_reserved_gib']>=max(r['reserved_gib'] for r in rows)
    gates=dict(input_output_and_total_token_eligibility=eligible,all_json=all_json,all_eos=all_eos,
               mean_at_most4s=mean<=4,p95_at_most8s=p95<=8,
               peak_reserved_at_most6p5=summary['peak_reserved_gib']<=6.5,
               no_cpu_parameter_offload=summary['cpu_parameter_count']==0,
               full_runtime_equivalence_if_compiled=runtime_verified)
    report=dict(at=now(),root=str(root),plan_hash=fingerprint(plan),raw_measurements_hash=fingerprint(rows),
        adapter_sha256=summary['adapter_sha256'],calls=len(rows),rounds=plan['measured_rounds'],
        mean_seconds=mean,p95_seconds=p95,load_seconds=summary['load_seconds'],
        output_tokens_per_second=sum(r['generated_tokens_including_eos'] for r in rows)/sum(times),
        peak_allocated_gib=summary['peak_allocated_gib'],peak_reserved_gib=summary['peak_reserved_gib'],
        gates=gates,performance_passed=all(gates.values()),
        scope='Verified own-weight frozen Qwen3.5 performance only; language/semantic acceptance, final candidate and independent confirmation remain separate.',
        stage_goal_complete=False,release_approved=False)
    write_json(dest,report);print(report,flush=True)


if __name__=='__main__':
    main()
