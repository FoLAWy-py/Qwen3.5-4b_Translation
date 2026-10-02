"""Own-weight formal timing after semantic gates; CPU input preflight is separate."""
import argparse
import importlib.metadata
import math
import os
import statistics
import time
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load, verify_training
from witrans import SYSTEM_PROMPT, make_messages, parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json, write_jsonl


def fixed_inputs():
    prior=load('runs/qwen35-v3-diagnostic-recovery1/plan.json')
    rows=read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance-inputs.jsonl')
    assert fingerprint(rows)==prior['performance_inputs_hash'] and len(rows)==24
    assert len({r['id'] for r in rows})==len(rows)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    lengths={}
    for row in rows:
        body=len(tokenizer.encode(row['input']['text'],add_special_tokens=False))
        prompt=tokenizer.apply_chat_template(make_messages(**row['input']),tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if hasattr(prompt,'keys'):prompt=prompt['input_ids']
        assert 20<=body<=120 and len(prompt)+128<=1024
        lengths[row['id']]=dict(body_tokens=body,prompt_tokens=len(prompt))
    return rows,lengths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs-preflight',help='Fresh CPU-only receipt path; no candidate acceptance claim')
    parser.add_argument('--training-plan')
    parser.add_argument('--acceptance',help='Content-bound completed candidate DEV receipt')
    parser.add_argument('--owner-job',help='Completed serial model predecessor')
    parser.add_argument('--root',help='Fresh result directory')
    parser.add_argument('--mode',choices=('eager','compiled'),default='eager')
    args=parser.parse_args()
    rows,lengths=fixed_inputs()
    if args.inputs_preflight:
        dest=Path(args.inputs_preflight);assert not dest.exists()
        write_json(dest,dict(at=now(),performance_inputs_hash=fingerprint(rows),rows=len(rows),
            lengths=lengths,warmup_rounds=1,measured_rounds=3,decoding=dict(do_sample=False,max_length=1024,max_new_tokens=128),
            gpu_executed=False,candidate_weight_validation_pending=True,stage_goal_complete=False))
        print(dict(cpu_input_preflight=str(dest),eligible_rows=len(rows)),flush=True)
        return
    assert all((args.training_plan,args.acceptance,args.owner_job,args.root))
    root=Path(args.root);assert not root.exists(), 'Preserve previous timing attempts'
    training=load(args.training_plan);metrics,sha=verify_training(training)
    acceptance=load(args.acceptance)
    assert acceptance['optimization_screen_passed'] and acceptance['stage_development']['passed']
    assert acceptance['adapter_sha256']==sha and acceptance['training_plan_hash']==fingerprint(training)
    owner=load(args.owner_job);assert owner['status']=='finished' and owner['exit_code']==0
    own=psutil.Process();family={own.pid,*(p.pid for p in own.parents())}
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid in family or 'python' not in (process.info['name'] or '').lower():continue
        command=' '.join(process.info['cmdline'] or [])
        assert not any(term in command for term in ('scripts.low_cpu_run','scripts.train_',
            'scripts.evaluate_','scripts.probe_qwen35_','scripts.benchmark_','scripts.diagnose_qwen35_stage'))
    import torch
    toolchain=None
    if args.mode=='compiled':
        from scripts.probe_qwen35_compiled_runtime import cached_toolchain
        toolchain=cached_toolchain()
        import torch._inductor.config
        torch._inductor.config.emulate_precision_casts=True
        torch._inductor.config.emulate_divison_rounding=True
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str((root/'inductor').resolve())
        os.environ['TRITON_CACHE_DIR']=str((root/'triton').resolve())
    runtime=dict(mode=args.mode,toolchain=toolchain,
        compile=dict(backend='inductor',mode='default',fullgraph=False,dynamic=None,
            cache_implementation='static',disable_hf_automatic_compile=True,
            emulate_precision_casts=True,emulate_division_rounding=True) if toolchain else None,
        dependencies={key:importlib.metadata.version(key) for key in ('torch','transformers','peft','bitsandbytes')})
    plan=dict(at=now(),adapter_dir=training['output'],adapter_sha256=sha,
        base_revision=training['base']['revision'],prompt_hash=fingerprint(SYSTEM_PROMPT),
        quantization=dict(type='nf4',double_quant=True,compute_dtype='bfloat16'),
        decoding=dict(do_sample=False,max_length=1024,max_new_tokens=128),
        training_plan_hash=fingerprint(training),training_metrics_hash=fingerprint(metrics),
        development_acceptance_hash=fingerprint(acceptance),owner_job=args.owner_job,
        performance_inputs_hash=fingerprint(rows),performance_lengths=lengths,
        warmup_rounds=1,measured_rounds=3,runtime=runtime,runtime_fingerprint=fingerprint(runtime),
        selection='All24 pre-existing tokenizer-eligible inputs in frozen source order; no output filtering. Full runtime warmup then3 measured rounds; no reruns selected by latency.',
        scope='Own candidate weights; compiled mode additionally requires all24 eager baseline raw outputs to match every compiled call. This finite equivalence check does not replace full semantic/runtime regression.',
        stop_conditions=['InvalidJSON/EOS, CPU placement, token mismatch/truncation, memory>6.5GiB, nonfinite timing, OOM: preserve evidence and stop; no automatic retry.'],
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    root.mkdir();write_json(root/'plan.json',plan);write_jsonl(root/'performance-inputs.jsonl',rows)
    state=dict(at=now(),phase='loading',status='running',plan_hash=fingerprint(plan),
        process_pid=own.pid,process_created=own.create_time(),gpu_executed=True,stage_goal_complete=False)
    write_json(root/'progress.json',state)
    try:
        from witrans_tools.qwen35 import Qwen35Translator
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        translator=Qwen35Translator(adapter_dir=training['output']);torch.cuda.synchronize()
        load_seconds=time.perf_counter()-start;assert translator.adapter_sha256==sha
        def measured(row,repeat):
            torch.cuda.synchronize();begin=time.perf_counter()
            raw,ended=translator.generate_raw(**row['input'],max_new_tokens=128)
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            prediction=parse_translation(raw);stats=translator.last_generation_stats
            assert ended and stats['generated_tokens_including_eos']<=128
            assert stats['prompt_tokens']==lengths[row['id']]['prompt_tokens']
            assert all(p.device.type=='cuda' for p in translator.model.parameters())
            assert torch.cuda.max_memory_reserved()/1024**3<=6.5 and math.isfinite(seconds) and seconds>0
            return dict(id=row['id'],repeat=repeat,input=row['input'],raw=raw,prediction=prediction,
                ended=ended,valid_json=True,seconds=seconds,body_tokens=lengths[row['id']]['body_tokens'],**stats,
                allocated_gib=torch.cuda.memory_allocated()/1024**3,reserved_gib=torch.cuda.memory_reserved()/1024**3,
                cpu_parameter_count=0)
        baseline={}
        if args.mode=='compiled':
            state['phase']='eager_equivalence_baseline';write_json(root/'progress.json',state)
            for row in rows:
                result=measured(row,-2);baseline[row['id']]=result['raw'];append_jsonl(root/'eager-baseline.jsonl',result)
            base=translator.model.get_base_model();factory=translator._generation_config
            translator._generation_config=lambda **kw:factory(**kw,cache_implementation='static',disable_compile=True)
            base.forward=torch.compile(base.forward,backend='inductor',mode='default',fullgraph=False,dynamic=None)
        for repeat in (-1,0,1,2):
            state.update(phase='warmup' if repeat==-1 else 'formal_timing',round=repeat,updated_at=now())
            write_json(root/'progress.json',state)
            for row in rows:
                result=measured(row,repeat)
                if baseline:assert result['raw']==baseline[row['id']], 'Compiled output changed; preserve failure'
                append_jsonl(root/('warmup.jsonl' if repeat==-1 else 'performance.jsonl'),result)
                print(dict(round=repeat,id=row['id'],seconds=result['seconds']),flush=True)
        measurements=read_jsonl(root/'performance.jsonl');times=[r['seconds'] for r in measurements]
        summary=dict(at=now(),plan_hash=fingerprint(plan),adapter_sha256=sha,calls=len(times),rounds=3,
            load_seconds=load_seconds,mean_seconds=statistics.mean(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],
            generated_tokens_including_eos_max=max(r['generated_tokens_including_eos'] for r in measurements),
            valid_json=len(times),ended=len(times),raw_measurements_hash=fingerprint(measurements),
            warmup_hash=fingerprint(read_jsonl(root/'warmup.jsonl')),warmup_calls=len(rows),
            output_tokens_per_second=sum(r['generated_tokens_including_eos'] for r in measurements)/sum(times),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,cpu_parameter_count=0,
            runtime_fingerprint=fingerprint(runtime),compiled_matches_all_eager_baselines=bool(baseline) if toolchain else None,
            loading_info=translator.loading_info,stage_goal_complete=False,release_approved=False)
        write_json(root/'performance-summary.json',summary)
        state.update(status='finished',phase='formal_timing_complete',summary_hash=fingerprint(summary))
    except Exception as exc:
        state.update(status='failed',phase='failed_preserving_evidence',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        state['updated_at']=now();write_json(root/'progress.json',state)


if __name__=='__main__':main()
