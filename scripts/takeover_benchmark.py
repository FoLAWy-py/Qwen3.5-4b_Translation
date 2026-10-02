"""Frozen v2 benchmark; no training, output filtering, retries or promotion."""
import argparse
import importlib.metadata
import hashlib
import math
import os
import statistics
import time
from pathlib import Path

import psutil
from witrans_tools.protocol import SYSTEM_PROMPT, make_messages, parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--mode', choices=['baseline', 'bf16', 'compiled','decode_compiled'], default='baseline')
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise FileExistsError(root)
    own = psutil.Process()
    family = {own.pid, *(p.pid for p in own.parents())}
    inventory = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        if 'python' in (p.info['name'] or '').lower():
            inventory.append(p.info)
            if p.pid not in family:
                raise RuntimeError(f'Other Python process requires serial inspection: {p.info}')
    import torch
    from witrans_tools.qwen35 import Qwen35Translator
    root.mkdir(parents=True)
    rows = read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance-inputs.jsonl')
    if len(rows) != 24 or fingerprint(rows) != 'f4057624ca5a024d3bf286b0d0acb37fcacfd6273250ae9979535d64c7c570f9':
        raise ValueError('Fixed24 identity changed')
    config = dict(at=now(), mode=args.mode, adapter_sha256='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27',
        revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', prompt_hash=fingerprint(SYSTEM_PROMPT),
        quantization='nf4', compute_dtype='bfloat16', lora_dtype='float32' if args.mode != 'bf16' else 'bfloat16',
        cache='static' if args.mode in ('compiled','decode_compiled') else 'dynamic', do_sample=False, max_length=1024, max_new_tokens=128,
        compile=dict(backend='inductor',mode='default',fullgraph=False,dynamic=False if args.mode=='decode_compiled' else None,
            scope='decode_only' if args.mode=='decode_compiled' else 'prefill_and_decode',emulate_precision_casts=True,
            max_cache_len=1024 if args.mode=='decode_compiled' else None,
            forward_signature_preserved=True if args.mode=='decode_compiled' else None,
            emulate_divison_rounding=True,disable_hf_automatic_compile=True) if args.mode in ('compiled','decode_compiled') else None,
        warmup_rounds=1, measured_rounds=3, input_hash=fingerprint(rows), processes=inventory,
        cpu_affinity=own.cpu_affinity(), threads=torch.get_num_threads(),
        env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM')},
        dependencies={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes')},
        gpu=dict(name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            capability=list(torch.cuda.get_device_capability(0))),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('witrans_tools/qwen35.py'),Path('witrans_tools/runtime.py'),Path('witrans_tools/protocol.py'),Path('witrans_tools/compile_runtime.py')]},
        release_approved=False)
    write_json(root/'plan.json', config)
    state = dict(at=now(),pid=own.pid,created=own.create_time(),phase='loading',status='running')
    write_json(root/'progress.json',state)
    try:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        translator = Qwen35Translator(adapter_dir='models/witrans-qwen35-v2-critical-cpo')
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-start
        if translator.adapter_sha256 != config['adapter_sha256']:
            raise ValueError('Frozen adapter mismatch')
        if args.mode == 'bf16':
            from witrans_tools.qwen35_lora_precision import cast_qwen35_lora_to_bfloat16
            write_json(root/'cast.json',cast_qwen35_lora_to_bfloat16(translator.model))
        if args.mode in ('compiled','decode_compiled'):
            from witrans_tools.compile_runtime import enable_compiled_runtime
            toolchain=enable_compiled_runtime(translator,root,decode_only=args.mode=='decode_compiled')
            write_json(root/'compile.json',toolchain)
            if args.mode=='decode_compiled' and not translator.model.get_base_model()._supports_logits_to_keep():
                raise RuntimeError('Decode dispatcher lost logits_to_keep capability')
        results=[]
        warmup_start=time.perf_counter()
        for repeat in (-1,0,1,2):
            state.update(phase='warmup' if repeat == -1 else 'timing',repeat=repeat,at=now())
            write_json(root/'progress.json',state)
            for row in rows:
                body=len(translator.tokenizer.encode(row['input']['text'],add_special_tokens=False))
                if not 20 <= body <= 120:
                    raise ValueError(f'Fixed timing body budget: {row["id"]} {body}')
                torch.cuda.synchronize();start=time.perf_counter()
                raw,ended=translator.generate_raw(**row['input'],max_new_tokens=128)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                result=dict(id=row['id'],repeat=repeat,input=row['input'],raw=raw,ended=ended,
                    prediction=parse_translation(raw),seconds=seconds,body_tokens=body,
                    **translator.last_generation_stats,
                    allocated_gib=torch.cuda.memory_allocated()/1024**3,reserved_gib=torch.cuda.memory_reserved()/1024**3)
                result['output_tokens_per_second']=result['generated_tokens_including_eos']/seconds
                result['input_output_tokens_per_second']=(result['prompt_tokens']+result['generated_tokens_including_eos'])/seconds
                append_jsonl(root/('warmup.jsonl' if repeat==-1 else 'performance.jsonl'),result)
                if not ended or result['generated_tokens_including_eos'] > 128:
                    raise RuntimeError('EOS/output token limit failed')
                if any(p.device.type!='cuda' for p in translator.model.parameters()):
                    raise RuntimeError('Parameter offload')
                if torch.cuda.max_memory_reserved()/1024**3 > 6.5:
                    raise RuntimeError('Memory cap exceeded')
                if repeat>=0: results.append(result)
                print(dict(repeat=repeat,id=row['id'],seconds=seconds),flush=True)
            if repeat==-1: warmup_seconds=time.perf_counter()-warmup_start
        times=[r['seconds'] for r in results]
        summary=dict(at=now(),plan_hash=fingerprint(config),calls=len(results),load_seconds=load_seconds,
            warmup_seconds=warmup_seconds,compile_cost_included_in_warmup=args.mode in ('compiled','decode_compiled'),
            mean_seconds=statistics.mean(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],
            output_tokens_per_second=sum(r['generated_tokens_including_eos'] for r in results)/sum(times),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,cpu_parameter_count=0,
            measurements_hash=fingerprint(results),all_json_valid=True,all_eos=True,
            performance_passed=statistics.mean(times)<=4 and sorted(times)[math.ceil(.95*len(times))-1]<=8,
            semantic_regression='pending' if args.mode!='baseline' else 'existing_development_results_require_binding_check',
            release_approved=False)
        write_json(root/'summary.json',summary)
        state.update(status='finished',phase='complete')
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        state['at']=now();write_json(root/'progress.json',state)


if __name__=='__main__':main()
