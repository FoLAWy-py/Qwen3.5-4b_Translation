"""Serial frozen-candidate latency diagnosis and real TRAIN error discovery."""
import gc
import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil
from witrans import SYSTEM_PROMPT, make_messages, parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.qwen35 import Qwen35Translator


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class TokenClock:
    def __init__(self):
        self.seen_prompt = False
        self.times = []

    def put(self, value):
        if not self.seen_prompt:
            self.seen_prompt = True
        else:
            self.times.append(time.perf_counter())

    def end(self):
        pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',default='')
    args=parser.parse_args()
    assert not args.attempt or args.attempt.isalnum(), 'Simple attempt identifier required'
    root = Path('runs/qwen35-v3-diagnostic'+('-'+args.attempt if args.attempt else ''))
    assert not root.exists(), 'New attempt required; preserve partial output'
    root.mkdir()
    state_path = root/'progress.json'
    state = dict(at=now(), phase='preflight', process_pid=psutil.Process().pid,
                 process_created=psutil.Process().create_time(), command=psutil.Process().cmdline(),
                 stage_goal_complete=False, release_approved=False)
    write_json(state_path,state)
    try:
        # Read-only process inventory before loading CUDA. Ignore this process and uv parents.
        own = psutil.Process(); parents = {p.pid for p in own.parents()}
        inventory = []
        for p in psutil.process_iter(['pid','name','cmdline','create_time']):
            info = p.info
            if not info['name'] or not any(s in info['name'].lower() for s in ('python','uv')):
                continue
            inventory.append(info)
            if info['pid']!=own.pid and info['pid'] not in parents and info['cmdline']:
                command = ' '.join(info['cmdline']).lower()
                assert not any(s in command for s in ('train_', 'evaluate_', 'decode_', 'benchmark_', 'diagnose_qwen35_stage')), 'Another GPU job is active'
        write_json(root/'process-preflight.json',dict(at=now(),processes=inventory))
        audit = load('runs/qwen35-v3-verified-start-audit.json')
        assert audit['reviewed_train']['rows']==416
        adapter = 'models/witrans-qwen35-v2-critical-cpo'
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
        benchmark = read_jsonl('data/prepared/short-benchmark-v1/inputs.jsonl')
        eligible, excluded, lengths = [], [], {}
        for r in benchmark:
            body = len(tokenizer.encode(r['input']['text'],add_special_tokens=False))
            prompt = tokenizer.apply_chat_template(make_messages(**r['input']),tokenize=True,
                      add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if hasattr(prompt,'keys'): prompt=prompt['input_ids']
            lengths[r['id']] = dict(body_tokens=body,prompt_tokens=len(prompt))
            if 20<=body<=120 and len(prompt)+128<=1024:
                eligible.append(r)
            else:
                excluded.append(dict(id=r['id'],**lengths[r['id']],reason='Tokenizer eligibility only; no padding or output-driven selection'))
        assert eligible and {r['input']['target_lang'] for r in eligible}=={'en','zh-CN'}
        write_jsonl(root/'performance-inputs.jsonl',eligible)
        train = read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
        assert fingerprint(train)==audit['reviewed_train']['hash']
        cases = load('runs/qwen35-v2-short-benchmark.json')['cases']
        plan = dict(at=now(),adapter_dir=adapter,adapter_sha256=audit['verified_weights'][str(Path(adapter)/'adapter_model.safetensors')],
                    base_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',prompt_hash=fingerprint(SYSTEM_PROMPT),
                    quantization=dict(type='nf4',double_quant=True,compute_dtype='bfloat16'),
                    decoding=dict(do_sample=False,max_length=1024,max_new_tokens=128),
                    latency_hypothesis='Separate output length and time-to-first-token from decoding; profiler locates costly operators, not a causal LoRA ablation.',
                    diagnostics=cases,performance_inputs_hash=fingerprint(eligible),performance_lengths=lengths,
                    performance_excluded=excluded,warmup_rounds=1,measured_rounds=3,
                    train_path='data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl',
                    train_hash=fingerprint(train),train_rows=416,
                    train_generation_output='runs/qwen35-v3-train-discovery'+('-'+args.attempt if args.attempt else '')+'.jsonl',
                    order='Latencies and operator profile; formal timing; then all TRAIN real generation. One GPU process at a time.',
                    method_claim=False,stage_goal_complete=False,release_approved=False)
        write_json(root/'plan.json',plan)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter(); translator=Qwen35Translator(adapter_dir=adapter);torch.cuda.synchronize()
        load_seconds=time.perf_counter()-start
        assert translator.adapter_sha256==plan['adapter_sha256']
        lora = Counter(str(p.dtype) for n,p in translator.model.named_parameters() if 'lora_' in n)
        state.update(phase='latency_diagnostic',plan_hash=fingerprint(plan),load_seconds=load_seconds)
        write_json(state_path,state)
        for case in cases: translator.generate_raw(**case,max_new_tokens=128)
        for i,case in enumerate(cases):
            torch.cuda.synchronize();start=time.perf_counter()
            raw,eos=translator.generate_raw(**case,max_new_tokens=128)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            append_jsonl(root/'latency.jsonl',dict(mode='uninstrumented',case=i,input=case,raw=raw,ended=eos,
                          seconds=seconds,**translator.last_generation_stats))
            prompt=tokenizer.apply_chat_template(make_messages(**case),tokenize=False,
                           add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda:0')
            clock=TokenClock(); config=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=128,
                 eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                output=translator.model.generate(**inputs,generation_config=config,streamer=clock)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            ids=output[0,inputs['input_ids'].shape[1]:].tolist();ended=ids[-1]==tokenizer.eos_token_id
            assert len(ids)==len(clock.times)
            append_jsonl(root/'latency.jsonl',dict(mode='streamer_diagnostic',case=i,input=case,
                raw=tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False),ended=ended,
                seconds=seconds,prompt_tokens=inputs['input_ids'].shape[1],generated_tokens_including_eos=len(ids),
                ttft_seconds=clock.times[0]-start,decode_after_first_seconds=clock.times[-1]-clock.times[0],
                scope='TTFT includes prompt/setup/prefill/first decode; streamer synchronizes CPU. Not formal timing.'))
        state.update(phase='operator_profile',updated_at=now());write_json(state_path,state)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            raw,eos=translator.generate_raw(**cases[0],max_new_tokens=128)
            torch.cuda.synchronize()
        operators=[]
        for event in profile.key_averages():
            operators.append(dict(name=event.key,calls=event.count,self_cpu_us=event.self_cpu_time_total,
                self_device_us=getattr(event,'self_device_time_total',0),cpu_us=event.cpu_time_total))
        write_json(root/'operators.json',dict(at=now(),raw=raw,ended=eos,lora_dtypes=dict(lora),
            largest_cpu=sorted(operators,key=lambda r:r['self_cpu_us'],reverse=True)[:40],
            largest_device=sorted(operators,key=lambda r:r['self_device_us'],reverse=True)[:40],
            scope='Instrumented single candidate; profiling overhead excluded from acceptance timing. No base rerun or kernel/weight changes.'))
        state.update(phase='formal_timing',updated_at=now(),eligible_per_round=len(eligible));write_json(state_path,state)
        for row in eligible: translator.generate_raw(**row['input'],max_new_tokens=128)
        torch.cuda.reset_peak_memory_stats();times=[];valid=0;ended=0;max_tokens=0
        for repeat in range(3):
            for row in eligible:
                torch.cuda.synchronize();start=time.perf_counter()
                raw,eos=translator.generate_raw(**row['input'],max_new_tokens=128)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                try: parse_translation(raw);ok=True
                except ValueError: ok=False
                stats=translator.last_generation_stats
                times.append(seconds);valid+=ok;ended+=eos;max_tokens=max(max_tokens,stats['generated_tokens_including_eos'])
                append_jsonl(root/'performance.jsonl',dict(id=row['id'],repeat=repeat,input=row['input'],raw=raw,ended=eos,
                    valid_json=ok,body_tokens=lengths[row['id']]['body_tokens'],seconds=seconds,**stats,
                    allocated_gib=torch.cuda.memory_allocated()/1024**3,reserved_gib=torch.cuda.memory_reserved()/1024**3))
                state.update(round=repeat,measured=len(times),updated_at=now());write_json(state_path,state)
                print(dict(phase='formal_timing',round=repeat,id=row['id'],seconds=seconds),flush=True)
        p95=sorted(times)[math.ceil(.95*len(times))-1]
        write_json(root/'performance-summary.json',dict(at=now(),plan_hash=fingerprint(plan),adapter_sha256=translator.adapter_sha256,
            calls=len(times),rounds=3,load_seconds=load_seconds,mean_seconds=statistics.mean(times),p95_seconds=p95,
            generated_tokens_including_eos_max=max_tokens,valid_json=valid,ended=ended,
            output_tokens_per_second=sum(r['generated_tokens_including_eos'] for r in read_jsonl(root/'performance.jsonl'))/sum(times),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            cpu_parameter_count=sum(p.numel() for p in translator.model.parameters() if p.device.type=='cpu'),
            mean_at_most4s=statistics.mean(times)<=4,p95_at_most8s=p95<=8,
            scope='Frozen current Qwen3.5 candidate; future final candidate must rerun own-weight test. Semantic/direction acceptance separate.'))
        del translator;gc.collect();torch.cuda.empty_cache()
        state.update(phase='train_generation',updated_at=now());write_json(state_path,state)
        # Process exit is required to guarantee that profiler/CUDA references
        # release their physical GPU allocations. del/gc/empty_cache did not
        # do so in the recovery1 observation. Launch TRAIN as a separate owned
        # job only after this process exits; never nest another model loader.
        state.update(phase='formal_timing_complete_train_launch_pending',updated_at=now(),
            next_serial_command=[sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module','scripts.evaluate_qwen35',
                '--adapter-dir',adapter,'--input',plan['train_path'],'--output',plan['train_generation_output'],
                '--max-new-tokens','256'])
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error_type=type(exc).__name__,error=str(exc),updated_at=now())
        raise
    finally:
        write_json(state_path,state)


if __name__=='__main__':
    main()
