"""Freeze a source-token-qualified short benchmark before candidate outputs."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

def main():
    destination = Path('data/prepared/short-benchmark-v1')
    if destination.exists():
        raise ValueError('Short benchmark already frozen')
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B', local_files_only=True, trust_remote_code=False)
    paths = ['data/prepared/v4/dev.jsonl', 'data/prepared/v8-multisentence/train.jsonl']
    source = [row for path in paths for row in read_jsonl(path)]
    buckets = defaultdict(list)
    lengths = {}
    for row in source:
        body = len(tokenizer.encode(row['input']['text'], add_special_tokens=False))
        reference = len(tokenizer.encode(json.dumps(row['output'], ensure_ascii=False, separators=(',', ':')), add_special_tokens=False)) + 1
        if 20 <= body <= 120 and reference <= 128:
            key = f"{row['category']}/{row['input']['target_lang']}"
            buckets[key].append(row)
            lengths[row['id']] = {'body_tokens':body, 'reference_json_and_eos_tokens':reference}
    chosen = [row for key in sorted(buckets) for row in sorted(buckets[key], key=lambda r:r['id'])[:3]]
    directions = Counter(row['input']['target_lang'] for row in chosen)
    if len(chosen) < 16 or any(directions[direction] < 6 for direction in ('en', 'zh-CN')):
        raise ValueError('Insufficient eligible short-input coverage')
    write_jsonl(destination / 'inputs.jsonl', chosen)
    report = {'at':now(), 'rows':len(chosen), 'hash':fingerprint(chosen), 'source_hash':fingerprint(source), 'source_paths':paths,
        'directions':dict(directions), 'eligible_by_stratum':{k:len(v) for k,v in buckets.items()},
        'selection':'First three IDs in each eligible category/direction; only source and reference length used; no model outputs or latency used',
        'lengths':{row['id']:lengths[row['id']] for row in chosen}, 'generation_budget':128, 'repetitions':3,
        'scope':'Known-development and reviewed future-training performance inputs only, not an accuracy test or held-out split. Short inputs may fail20-token floor and are not padded with artificial source text. Reference length is eligibility only; actual output token counts/end state must be measured.'}
    write_json(destination / 'manifest.json', report)
    print({k:v for k,v in report.items() if k != 'lengths'})

if __name__ == '__main__':
    main()
