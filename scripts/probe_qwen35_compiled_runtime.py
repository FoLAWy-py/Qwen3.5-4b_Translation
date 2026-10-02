"""One frozen Qwen3.5 compile hypothesis, using only a previously cached toolchain."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import psutil
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cached_toolchain():
    previous=load('runs/v7-compile-eager-rounding-regression.json')['toolchain']
    package=Path(previous['package_directory'])
    assert package.name=='triton' and package.is_dir()
    # Append after the active environment. Only absent Triton is supplied by the
    # cached overlay; don't replace Qwen3.5 transformers/PEFT/torch dependencies.
    if importlib.util.find_spec('triton') is None:
        sys.path.append(str(package.parent))
    from scripts.triton_toolchain import configure_bundled_toolchain
    configured=configure_bundled_toolchain()
    assert Path(configured['package_directory']).resolve()==package.resolve()
    from triton.runtime.build import get_cc
    assert Path(get_cc()).resolve()==Path(configured['c_compiler']).resolve()
    files=[Path(configured['c_compiler']),package/'backends/nvidia/bin/ptxas.exe',package/'__init__.py']
    return dict(paths=configured,files_sha256={str(p):sha(p) for p in files},
                triton_version=importlib.metadata.version('triton-windows'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--owner-job',default='runs/qwen35-v3-train-adopted-job.json')
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists(), 'Preserve all attempts'
    state=dict(at=now(),phase='toolchain_preflight',status='running',stage_goal_complete=False,
               default_promoted=False,release_approved=False)
    write_json(dest,state)
    try:
        toolchain=cached_toolchain()
        import torch
        import torch._inductor.config
        from transformers import StaticCache, AutoConfig
        from witrans_tools.qwen35 import Qwen35Translator
        dependencies={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','bitsandbytes','triton-windows')}
        config=AutoConfig.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
        cache=StaticCache(config=config,max_cache_len=1024)
        rows=read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance-inputs.jsonl')
        prior=load('runs/qwen35-v3-diagnostic-recovery1/plan.json')
        assert fingerprint(rows)==prior['performance_inputs_hash']
        protocol=dict(base_revision=prior['base_revision'],adapter_dir=prior['adapter_dir'],
            adapter_sha256=prior['adapter_sha256'],prompt_hash=fingerprint(SYSTEM_PROMPT),
            quantization=prior['quantization'],decoding=prior['decoding'],
            performance_inputs_hash=fingerprint(rows),diagnostic_id=rows[0]['id'],
            compile=dict(backend='inductor',mode='default',fullgraph=False,dynamic=None,
              cache_implementation='static',disable_hf_automatic_compile=True,
              emulate_precision_casts=True,emulate_division_rounding=True),
            dependencies=dependencies,toolchain=toolchain,
            hypothesis='Compiling fixed candidate forward with static hybrid cache and eager rounding reduces many small CUDA launch costs without changing raw outputs.',
            selection='First frozen eligible input by stored order, one eager warmup plus one baseline; four compiled calls (first2 compilation warmups, last2 diagnostic). Stop on invalid JSON/EOS, CPU placement or memory>6.5GiB. No tuning by semantic DEV.',
            scope='Single runtime feasibility probe; full own-weight24-input/3-round timing and quality regression required before promotion.')
        state.update(toolchain=toolchain,protocol=protocol,runtime_fingerprint=fingerprint(protocol),
                     hybrid_cache_compileable=cache.is_compileable,python_prefix=sys.prefix,
                     torch_file=torch.__file__,gpu_compilation_executed=False)
        write_json(dest,state)
        assert cache.is_compileable
        if args.preflight:
            state.update(status='success',phase='cpu_preflight_complete')
        else:
            owner=load(args.owner_job)
            assert owner['status']=='finished' and owner['exit_code']==0, 'Serial predecessor must finish first'
            own=psutil.Process();parents={p.pid for p in own.parents()}
            for p in psutil.process_iter(['pid','name','cmdline','create_time']):
                if p.pid==own.pid or p.pid in parents or not p.info['name'] or 'python' not in p.info['name'].lower(): continue
                command=' '.join(p.info['cmdline'] or []).lower()
                assert not any(k in command for k in ('train_','evaluate_','diagnose_qwen35_stage','probe_qwen35_compiled_runtime','benchmark_')), 'Another GPU process is active'
            import os
            cache_root=dest.with_suffix('')
            os.environ['TORCHINDUCTOR_CACHE_DIR']=str((cache_root/'inductor').resolve())
            os.environ['TRITON_CACHE_DIR']=str((cache_root/'triton').resolve())
            torch._inductor.config.emulate_precision_casts=True
            torch._inductor.config.emulate_divison_rounding=True
            torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();translator=Qwen35Translator(adapter_dir=prior['adapter_dir']);torch.cuda.synchronize()
            state.update(load_seconds=time.perf_counter()-start,phase='eager_baseline')
            assert translator.adapter_sha256==protocol['adapter_sha256']
            write_json(dest,state)
            from witrans import parse_translation
            def measured():
                torch.cuda.synchronize();begin=time.perf_counter()
                raw,ended=translator.generate_raw(**rows[0]['input'],max_new_tokens=128)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                prediction=parse_translation(raw)
                reserved=torch.cuda.max_memory_reserved()/1024**3
                assert ended and reserved<=6.5 and all(p.device.type=='cuda' for p in translator.model.parameters())
                return dict(raw=raw,ended=ended,prediction=prediction,seconds=seconds,
                    **translator.last_generation_stats,peak_reserved_gib=reserved,
                    compiler_graph_count=int(torch._dynamo.utils.counters['stats'].get('unique_graphs',0)))
            measured();state['eager']=measured();write_json(dest,state)
            base=translator.model.get_base_model()
            factory=translator._generation_config
            translator._generation_config=lambda **kw:factory(**kw,cache_implementation='static',disable_compile=True)
            base.forward=torch.compile(base.forward,backend='inductor',mode='default',fullgraph=False,dynamic=None)
            state.update(phase='compiled_generations',gpu_compilation_executed=True,compiled=[]);write_json(dest,state)
            for index in range(4):
                result=measured();state['compiled'].append(result);state['completed_calls']=index+1;write_json(dest,state)
                print(dict(index=index,seconds=result['seconds'],exact_raw_match=result['raw']==state['eager']['raw']),flush=True)
            state.update(status='success',phase='probe_complete',
                exact_raw_match=all(r['raw']==state['eager']['raw'] for r in state['compiled']),
                repeated_mean_seconds=statistics.mean(r['seconds'] for r in state['compiled'][-2:]),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    except Exception as exc:
        state.update(status='failed',phase='failed_preserving_evidence',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        state['updated_at']=now();write_json(dest,state)
    print(dict(status=state['status'],phase=state['phase'],runtime_fingerprint=state.get('runtime_fingerprint')),flush=True)


if __name__=='__main__':
    main()
