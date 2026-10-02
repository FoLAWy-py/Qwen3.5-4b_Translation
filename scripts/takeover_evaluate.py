"""Full old-set quality regression for a recorded inference-only runtime."""
import argparse
import importlib.metadata
import hashlib
import os
import time
from pathlib import Path
import psutil

from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json
from witrans_tools.protocol import SYSTEM_PROMPT, parse_translation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--mode',choices=['baseline','bf16','compiled','decode_compiled'],required=True)
    parser.add_argument('--cache-root')
    args=parser.parse_args()
    root=Path(args.root)
    if root.exists():raise FileExistsError(root)
    own=psutil.Process();family={own.pid,*(p.pid for p in own.parents())}
    inventory=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        if 'python' in (p.info['name'] or '').lower():
            inventory.append(p.info)
            if p.pid not in family:raise RuntimeError(f'Serial GPU prerequisite: {p.info}')
    import torch
    from witrans_tools.qwen35 import Qwen35Translator
    root.mkdir(parents=True)
    known=read_jsonl('data/prepared/v12-factorial-v2/dev.jsonl')
    public=read_jsonl('data/prepared/v12-factorial-v2/public-dev.jsonl')
    if len(known)!=200 or len(public)!=116:raise ValueError('Expected full200/116 development sets')
    plan=dict(at=now(),mode=args.mode,data_hashes={'known':fingerprint(known),'public':fingerprint(public)},
        prompt_hash=fingerprint(SYSTEM_PROMPT),processes=inventory,quantization='nf4',compute_dtype='bfloat16',
        lora_dtype='bfloat16' if args.mode=='bf16' else 'float32',cache='static' if args.mode in ('compiled','decode_compiled') else 'dynamic',
        compile=dict(backend='inductor',mode='default',fullgraph=False,dynamic=False if args.mode=='decode_compiled' else None,
            scope='decode_only' if args.mode=='decode_compiled' else 'prefill_and_decode',emulate_precision_casts=True,
            max_cache_len=1024 if args.mode=='decode_compiled' else None,
            forward_signature_preserved=True if args.mode=='decode_compiled' else None,
            emulate_divison_rounding=True,disable_hf_automatic_compile=True) if args.mode in ('compiled','decode_compiled') else None,
        max_length=1024,max_new_tokens=256,do_sample=False,thinking=False,
        cache_root=args.cache_root,
        dependencies={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes')},
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('witrans_tools/qwen35.py'),Path('witrans_tools/runtime.py'),Path('witrans_tools/protocol.py'),Path('witrans_tools/compile_runtime.py')]},
        semantic_review='pending_Codex_AI_full_source_context_output_review',release_approved=False)
    write_json(root/'plan.json',plan)
    state=dict(pid=own.pid,created=own.create_time(),status='running',phase='loading',at=now())
    write_json(root/'progress.json',state)
    try:
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        translator=Qwen35Translator();torch.cuda.synchronize();load_seconds=time.perf_counter()-begin
        if args.mode=='bf16':
            from witrans_tools.qwen35_lora_precision import cast_qwen35_lora_to_bfloat16
            write_json(root/'cast.json',cast_qwen35_lora_to_bfloat16(translator.model))
        if args.mode in ('compiled','decode_compiled'):
            from witrans_tools.compile_runtime import enable_compiled_runtime
            enable_compiled_runtime(translator,Path(args.cache_root) if args.cache_root else Path('runs/takeover-20261002')/args.mode,
                decode_only=args.mode=='decode_compiled')
        summaries={}
        for label,rows in (('known',known),('public',public)):
            completed=[]
            for ref in rows:
                row=dict(id=ref['id'],group_id=ref['group_id'],category=ref['category'],input=ref['input'],
                    reference=ref['output'],semantic_review='pending')
                torch.cuda.synchronize();start=time.perf_counter()
                raw,ended=translator.generate_raw(**ref['input'],max_new_tokens=256)
                torch.cuda.synchronize()
                row.update(raw=raw,ended=ended,seconds=time.perf_counter()-start,**translator.last_generation_stats)
                try:row['prediction']=parse_translation(raw);row['json_valid']=True
                except ValueError as exc:row.update(json_valid=False,error=str(exc))
                append_jsonl(root/(label+'.jsonl'),row);completed.append(row)
                state.update(phase=label,completed=len(completed),last_id=ref['id'],at=now())
                write_json(root/'progress.json',state)
                print(dict(set=label,id=ref['id'],json=row['json_valid'],eos=ended,seconds=row['seconds']),flush=True)
            summaries[label]=dict(rows=len(completed),json_valid=sum(r['json_valid'] for r in completed),
                ended=sum(r['ended'] for r in completed),outputs_hash=fingerprint(completed),semantic_review='pending')
        write_json(root/'generation-summary.json',dict(at=now(),plan_hash=fingerprint(plan),adapter_sha256=translator.adapter_sha256,
            load_seconds=load_seconds,sets=summaries,cpu_parameter_count=0,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,release_approved=False))
        state.update(status='finished',phase='generation_complete_review_pending')
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        state['at']=now();write_json(root/'progress.json',state)


if __name__=='__main__':main()
