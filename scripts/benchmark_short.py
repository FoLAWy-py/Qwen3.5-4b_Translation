"""Three warm steady-state NF4 rounds on the pre-frozen source-length benchmark."""
import argparse
import json
import math
import time
from pathlib import Path
from witrans import SYSTEM_PROMPT, WiTrans, parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compiled-forward', action='store_true')
    parser.add_argument('--emulate-eager-casts', action='store_true')
    parser.add_argument('--emulate-eager-division', action='store_true')
    args = parser.parse_args()
    if (args.emulate_eager_casts or args.emulate_eager_division) and not args.compiled_forward:
        raise ValueError('Precision cast experiment requires compiled forward')
    if args.compiled_forward:
        from scripts.triton_toolchain import configure_bundled_toolchain
        toolchain = configure_bundled_toolchain()
    import torch
    output = Path(args.output)
    if output.exists() or output.with_suffix('.summary.json').exists():
        raise ValueError('Performance evidence already exists')
    root = Path('data/prepared/short-benchmark-v1')
    rows = read_jsonl(root / 'inputs.jsonl')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if fingerprint(rows) != manifest['hash'] or manifest['repetitions'] != 3 or manifest['generation_budget'] != 128:
        raise ValueError('Frozen protocol changed')
    for row in rows:
        validate_record(row, True, True, purpose='evaluation')
    load_start = time.perf_counter()
    model = WiTrans('models/Qwen3-4B', args.adapter_dir, max_length=1024, quantization='nf4')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    for row in rows:
        length = len(model.tokenizer.encode(row['input']['text'], add_special_tokens=False))
        if not 20 <= length <= 120 or length != manifest['lengths'][row['id']]['body_tokens']:
            raise ValueError('Source token length changed')
    compile_settings = None
    if args.compiled_forward:
        import torch._inductor.config
        if args.emulate_eager_casts:
            # Preserve the BF16/FP16 intermediate casts that ordinary eager
            # operators perform; this remains an explicit process-local trial.
            torch._inductor.config.emulate_precision_casts = True
        if args.emulate_eager_division:
            torch._inductor.config.emulate_divison_rounding = True
        base = model.model.get_base_model()
        compile_settings = {'backend':'inductor', 'mode':'default', 'fullgraph':False,
            'dynamic':None, 'cache_implementation':'static', 'automatic_hf_compile_disabled':True,
            'hf_quantizer_is_compileable':base.hf_quantizer.is_compileable,
            'emulate_precision_casts':bool(torch._inductor.config.emulate_precision_casts),
            'emulate_divison_rounding':bool(torch._inductor.config.emulate_divison_rounding)}
        factory = model._generation_config
        model._generation_config = lambda **kwargs: factory(**kwargs,
            cache_implementation='static', disable_compile=True)
        base.forward = torch.compile(base.forward, backend='inductor', mode='default',
            fullgraph=False, dynamic=None)
    # Warm each frozen input once, without using latency to remove or reorder inputs.
    warm_start = time.perf_counter()
    warm_rounds = 2 if args.compiled_forward else 1
    for warm_round in range(warm_rounds):
        for row in rows:
            model.generate_raw(**row['input'], max_new_tokens=128)
            print(f"warm{warm_round+1} {row['id']} complete", flush=True)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - warm_start
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for repetition in range(1, 4):
        for row in rows:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = {'id':row['id'], 'round':repetition, 'input':row['input'], 'valid_json':False}
            try:
                raw, ended = model.generate_raw(**row['input'], max_new_tokens=128)
                result.update(raw=raw, ended=ended, **(model.last_generation_stats or {}))
                result['prediction'] = parse_translation(raw)
                result['valid_json'] = True
            except (ValueError, RuntimeError) as exc:
                result['error'] = str(exc)
            torch.cuda.synchronize()
            result['seconds'] = time.perf_counter() - start
            append_jsonl(output, result)
            measurements.append(result)
            print(f"round{repetition} {row['id']}: {result['seconds']:.3f}s JSON={result['valid_json']} EOS={result.get('ended')}", flush=True)
    def metrics(selected):
        seconds = sorted(r['seconds'] for r in selected)
        return {'count':len(selected), 'mean_seconds':sum(seconds)/len(seconds),
            'p95_seconds':seconds[math.ceil(.95*len(seconds))-1],
            'output_tokens_per_second':sum(r.get('generated_tokens_including_eos',0) for r in selected)/sum(seconds),
            'valid_json':sum(r['valid_json'] for r in selected), 'ended':sum(r.get('ended',False) for r in selected),
            'errors':sum('error' in r for r in selected)}
    overall = metrics(measurements)
    reserved = torch.cuda.max_memory_reserved()/1024**3
    cpu = sum(p.numel() for p in model.model.parameters() if p.device.type == 'cpu')
    write_json(output.with_suffix('.summary.json'), {'at':now(), 'input_hash':manifest['hash'],
        'adapter_sha256':model.adapter_sha256, 'prompt_hash':fingerprint(SYSTEM_PROMPT),
        'quantization':'nf4', 'max_length':1024, 'max_new_tokens':128,
        'load_seconds':load_seconds, 'warm_seconds':warm_seconds,
        'warmup':f'{warm_rounds} full input rounds discarded', 'repetitions':3, 'overall':overall,
        'compile_settings':compile_settings,
        'toolchain':toolchain if args.compiled_forward else None,
        'compiler_counters':{name:dict(values) for name,values in torch._dynamo.utils.counters.items()}
            if args.compiled_forward else None,
        'rounds':{str(i):metrics([r for r in measurements if r['round']==i]) for i in range(1,4)},
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3, 'peak_reserved_gib':reserved,
        'cpu_parameter_count':cpu,
        'performance_passed':overall['mean_seconds']<=4 and overall['p95_seconds']<=8 and reserved<=6.5
            and cpu==0 and overall['errors']==0 and overall['valid_json']==len(measurements)
            and overall['ended']==len(measurements),
        'release_approved':False,
        'scope':'Performance only; known development and approved training sources. No semantic accuracy claim. P95 empirical nearest-rank. Includes strict JSON parsing and synchronized generation.'})
    print(overall, flush=True)

if __name__ == '__main__':
    main()
