"""Serial own-weight short-sentence timings after matched development decoding."""
import gc
import json
import statistics
import time
from pathlib import Path
from witrans import parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, write_json
from witrans_tools.qwen35 import Qwen35Translator

CASES = [
    dict(text='请问去机场的巴士几点出发？', target_lang='en'),
    dict(text='如果今天下雨，我们就取消徒步。', target_lang='en'),
    dict(text='这道菜不含花生，但含有牛奶。', target_lang='en'),
    dict(text='Please send me the revised schedule by Friday.', target_lang='zh-CN'),
    dict(text='Do not restart the device until the update finishes.', target_lang='zh-CN'),
    dict(text='The room rate includes breakfast for two people.', target_lang='zh-CN'),
]


def main():
    dest = Path('runs/qwen35-v2-short-benchmark.json')
    output = dest.with_suffix('.jsonl')
    assert not dest.exists() and not output.exists()
    for owned in ('qwen35-comparison-recovery2', 'qwen35-corrected-base'):
        while True:
            job = json.loads(Path(f'runs/{owned}-job.json').read_text(encoding='utf-8'))
            if job['status'] not in ('running', 'launching'): break
            time.sleep(5)
        assert job['status'] == 'finished' and job['exit_code'] == 0, owned
    import torch
    plan = json.loads(Path('data/prepared/qwen35-v2/plan.json').read_text(encoding='utf-8'))
    summaries = {}
    for role, adapter in [('base', None), ('finetuned', plan['cpo']['output'])]:
        torch.cuda.reset_peak_memory_stats()
        translator = Qwen35Translator(adapter_dir=adapter)
        for case in CASES:
            translator.generate_raw(**case, max_new_tokens=256)
        times, valid, ended = [], 0, 0
        for repeat in range(3):
            for index, case in enumerate(CASES):
                torch.cuda.synchronize()
                start = time.perf_counter()
                raw, eos = translator.generate_raw(**case, max_new_tokens=256)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                try:
                    parse_translation(raw)
                    json_valid = True
                except ValueError:
                    json_valid = False
                times.append(seconds); valid += json_valid; ended += eos
                append_jsonl(output, dict(role=role, repeat=repeat, case=index, input=case,
                    raw=raw, ended=eos, valid_json=json_valid, seconds=seconds,
                    **translator.last_generation_stats))
        ordered = sorted(times)
        # Empirical nearest-rank P95 with 18 measured calls after six warm-ups.
        p95 = ordered[__import__('math').ceil(.95 * len(ordered)) - 1]
        summaries[role] = dict(adapter_sha256=translator.adapter_sha256, calls=len(times),
            rounds=3, mean_seconds=statistics.mean(times), p95_seconds=p95,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            cpu_parameter_count=sum(p.numel() for p in translator.model.parameters() if p.device.type=='cpu'),
            valid_json=valid, ended=ended, mean_at_most4s=statistics.mean(times)<=4,
            p95_at_most8s=p95<=8)
        del translator
        gc.collect(); torch.cuda.empty_cache()
    write_json(dest, dict(at=now(), plan_hash=fingerprint(plan), cases=CASES, cases_hash=fingerprint(CASES),
        protocol='NF4; unchanged prompt; thinking disabled; greedy; total1024/output256; six warm-ups then three rounds; sequential GPU.',
        summaries=summaries, scope='Own-weight Windows PyTorch fallback timing. Fixed short-sentence sample; no production or long-text latency claim.',
        release_approved=False))
    print(summaries)


if __name__ == '__main__': main()
