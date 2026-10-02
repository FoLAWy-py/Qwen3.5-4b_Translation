"""Isolated real-model compilation probe; never changes deployment defaults."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-id')
    parser.add_argument('--emulate-eager-casts', action='store_true')
    parser.add_argument('--emulate-eager-division', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Preserve existing model compilation evidence')
    report = {'at': now(), 'status': 'running', 'phase': 'toolchain',
        'release_approved': False,
        'scope': 'One frozen performance input, static KV cache, explicitly compiled base forward. No full speed or semantic acceptance.',
        'packages': {name: importlib.metadata.version(name) for name in
            ('torch', 'triton-windows', 'bitsandbytes', 'transformers', 'peft')}}
    started = time.perf_counter()
    write_json(output, report)
    try:
        from scripts.triton_toolchain import configure_bundled_toolchain
        report['toolchain'] = configure_bundled_toolchain()
        # Configure before torch/bitsandbytes imports: compiler discovery is cached.
        import torch
        from witrans import SYSTEM_PROMPT, WiTrans, parse_translation
        root = Path('data/prepared/short-benchmark-v1')
        rows = read_jsonl(root / 'inputs.jsonl')
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if fingerprint(rows) != manifest['hash']:
            raise ValueError('Frozen performance inputs changed')
        if args.input_id is None:
            row = rows[0]  # Protocol choice, never selected using measured speed.
        else:
            matches = [r for r in rows if r['id']==args.input_id]
            if len(matches)!=1:
                raise ValueError('Diagnostic input must exist in frozen performance corpus')
            row = matches[0]
        import torch._inductor.config
        if args.emulate_eager_casts:
            torch._inductor.config.emulate_precision_casts = True
        if args.emulate_eager_division:
            torch._inductor.config.emulate_divison_rounding = True
        report.update(phase='loading', input_id=row['id'], input=row['input'],
            input_hash=manifest['hash'], prompt_hash=fingerprint(SYSTEM_PROMPT),
            quantization='nf4', max_length=1024, max_new_tokens=128,
            compile_settings={'backend':'inductor', 'mode':'default',
                'fullgraph':False, 'dynamic':None, 'cache_implementation':'static',
                'automatic_hf_compile_disabled':True,
                'emulate_precision_casts':bool(torch._inductor.config.emulate_precision_casts),
                'emulate_division_rounding':bool(torch._inductor.config.emulate_divison_rounding)})
        write_json(output, report)
        translator = WiTrans('models/Qwen3-4B', args.adapter_dir,
            max_length=1024, quantization='nf4')
        report['adapter_sha256'] = translator.adapter_sha256
        def measured():
            torch.cuda.synchronize()
            begin = time.perf_counter()
            raw, ended = translator.generate_raw(**row['input'], max_new_tokens=128)
            torch.cuda.synchronize()
            return {'raw':raw, 'ended':ended, 'prediction':parse_translation(raw),
                'seconds':time.perf_counter()-begin,
                'compiler_graph_count':int(torch._dynamo.utils.counters['stats'].get('unique_graphs',0)),
                **translator.last_generation_stats}
        measured()  # One eager warmup.
        report['eager'] = measured()
        base = translator.model.get_base_model()
        report['hf_quantizer_is_compileable'] = base.hf_quantizer.is_compileable
        # Local GenerationConfig factory; do not override the quantizer's safety flag.
        factory = translator._generation_config
        translator._generation_config = lambda **kwargs: factory(**kwargs,
            cache_implementation='static', disable_compile=True)
        base.forward = torch.compile(base.forward, backend='inductor',
            mode='default', fullgraph=False, dynamic=None)
        report.update(phase='first_compiled_generation')
        write_json(output, report)
        print('Compiling actual NF4 + PEFT forward with static cache', flush=True)
        torch.cuda.reset_peak_memory_stats()
        report['first_compiled'] = measured()
        report.update(phase='repeated_compiled_generation')
        write_json(output, report)
        report['repeated_compiled'] = [measured(), measured()]
        outputs = [report['first_compiled'], *report['repeated_compiled']]
        report.update(status='success', phase='complete',
            exact_raw_match=all(r['raw']==report['eager']['raw'] for r in outputs),
            all_ended=all(r['ended'] for r in [report['eager'], *outputs]),
            cpu_parameter_count=sum(p.numel() for p in translator.model.parameters()
                if p.device.type=='cpu'),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:8000])
    report['elapsed_seconds'] = time.perf_counter()-started
    write_json(output, report)
    print(report, flush=True)
    if report['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
