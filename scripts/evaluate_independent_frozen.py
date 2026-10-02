"""Run a frozen independent stage; never optimize against its references."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import psutil
from witrans_tools.common import append_jsonl,fingerprint,now,read_jsonl,write_json
from witrans_tools.protocol import parse_translation


def verify_frozen(data_root,stage):
    runtime=json.loads(Path('runs/takeover-20261002/runtime-freeze.json').read_text(encoding='utf-8'))
    if not runtime.get('runtime_frozen'):raise ValueError('Unfrozen runtime')
    for name,expected in runtime['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen runtime code changed: '+name)
    if hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest()!=runtime['uv_lock_sha256']:raise ValueError('Frozen dependencies changed')
    regression=json.loads(Path('runs/takeover-20261002/development-final/semantic-summary.json').read_text(encoding='utf-8'))
    if not regression.get('old_set_regression_review_complete'):raise ValueError('Full old-set semantic regression required')
    freeze=json.loads((data_root/'freeze.json').read_text(encoding='utf-8'))
    if freeze.get('outputs_seen') is not False or freeze['runtime_freeze_hash']!=fingerprint(runtime):raise ValueError('Source freeze/runtime mismatch')
    protocol=json.loads(Path('data/independent-test-protocol-20261002.json').read_text(encoding='utf-8'))
    if freeze['protocol_hash']!=fingerprint(protocol):raise ValueError('Test protocol changed')
    rows=read_jsonl(data_root/(stage+'.jsonl'))
    if fingerprint(rows)!=freeze['sets'][stage]['hash']:raise ValueError('Frozen data changed')
    if stage=='release':
        result=json.loads(Path('runs/takeover-20261002/confirmation/semantic-summary.json').read_text(encoding='utf-8'))
        if result.get('all_gates_passed') is not True:raise ValueError('Frozen confirmation entry gates did not pass')
    return runtime,freeze,regression,rows


def serial_inventory():
    own=psutil.Process();family={own.pid,*(p.pid for p in own.parents())};inventory=[]
    for process in psutil.process_iter(['pid','name','cmdline','create_time']):
        if 'python' not in (process.info['name'] or '').lower():continue
        info=process.info;inventory.append(info)
        if process.pid in family:continue
        # Explicitly inspected CPU-only source authoring process; no torch imports.
        if 'scripts.independent_source_pipeline' in (info.get('cmdline') or []):continue
        raise RuntimeError('Unverified concurrent Python task: '+str(info))
    return inventory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('confirmation','release'))
    parser.add_argument('--data-root',required=True,type=Path)
    args=parser.parse_args()
    root=Path('runs/takeover-20261002')/args.stage
    if root.exists():raise FileExistsError('Do not overwrite independent outputs')
    runtime,freeze,regression,rows=verify_frozen(args.data_root,args.stage)
    inventory=serial_inventory()
    import torch
    from witrans_tools.qwen35 import Qwen35Translator
    actual_versions={k:importlib.metadata.version(k) for k in runtime['configuration']['dependencies']}
    if actual_versions!=runtime['configuration']['dependencies']:raise ValueError('Runtime dependency version mismatch')
    root.mkdir(parents=True)
    write_json(root/'plan.json',dict(at=now(),stage=args.stage,runtime_freeze_hash=fingerprint(runtime),
        data_freeze_hash=fingerprint(freeze),data_hash=fingerprint(rows),configuration=runtime['configuration'],
        source_first=True,optimization_from_new_test=False,process_inventory=inventory,
        full_old_review_hash=fingerprint(regression),reviewer='Codex AI',human_acceptance=False,release_approved=False))
    own=psutil.Process();state=dict(pid=own.pid,created=own.create_time(),status='loading',at=now(),completed=0)
    write_json(root/'progress.json',state)
    try:
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        translator=Qwen35Translator(runtime='decode_compiled',lora_precision='float32',cache_dir='runs/takeover-20261002/decode-final')
        torch.cuda.synchronize();load=time.perf_counter()-start
        for source in rows:
            torch.cuda.synchronize();start=time.perf_counter()
            raw,ended=translator.generate_raw(**source['input'],max_new_tokens=256)
            torch.cuda.synchronize()
            row=dict(id=source['id'],group_id=source['group_id'],category=source['category'],
                input=source['input'],reference=source['output'],source_row_hash=fingerprint(source),raw=raw,
                ended=ended,seconds=time.perf_counter()-start,**translator.last_generation_stats,semantic_review='pending')
            try:row.update(prediction=parse_translation(raw),json_valid=True)
            except ValueError as exc:row.update(json_valid=False,error=str(exc))
            append_jsonl(root/'outputs.jsonl',row)
            state.update(status='running',completed=state['completed']+1,last_id=source['id'],at=now())
            write_json(root/'progress.json',state)
            print(dict(id=source['id'],completed=state['completed'],eos=ended,json_valid=row['json_valid']),flush=True)
        write_json(root/'generation-summary.json',dict(at=now(),rows=len(rows),load_seconds=load,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,cpu_parameter_count=0,
            outputs_hash=fingerprint(read_jsonl(root/'outputs.jsonl')),semantic_review='pending',release_approved=False))
        state.update(status='generation_complete_review_pending')
    except BaseException as exc:
        state.update(status='failed_preserving_outputs',error=type(exc).__name__+': '+str(exc));raise
    finally:
        state['at']=now();write_json(root/'progress.json',state)

if __name__=='__main__':main()
