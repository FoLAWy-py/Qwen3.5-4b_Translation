"""Freeze individually inspected real negatives; training mining is not release evidence."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from archive.qwen3.scripts.review_v8_mining import RAW_SHA
from archive.qwen3.runtime import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

PARENT = '1ed0446f66cdbcec444c95233355e31e7645c8c3bc70a70fe5759a79723b9736'

def main():
    raw_path = Path('runs/v8-multisentence-v5-mining.jsonl')
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Inspected generation changed')
    summary = json.loads(raw_path.with_suffix('.summary.json').read_text(encoding='utf-8'))
    if (summary['adapter_sha256'] != PARENT or summary['quantization'] != 'nf4'
            or summary['prompt_hash'] != fingerprint(SYSTEM_PROMPT)):
        raise ValueError('Generation provenance changed')
    refs = {r['id']:r for r in read_jsonl('data/prepared/v8-multisentence/train.jsonl')}
    decisions = {r['id']:r for r in read_jsonl('runs/v8-mining-semantic.jsonl')}
    raws = read_jsonl(raw_path)
    if len(raws) != 80 or len(refs) != 80 or set(decisions) != set(refs) or {r['id'] for r in raws} != set(refs):
        raise ValueError('Incomplete review')
    destination = Path('data/prepared/v8-mining/preferences.jsonl')
    if destination.exists():
        raise ValueError('Preferences already frozen')
    pairs = []
    for raw in raws:
        ref, review = refs[raw['id']], decisions[raw['id']]
        validate_record(ref, True, True)
        if (raw['input'] != ref['input'] or raw['reference'] != ref['output']
                or review['output_hash'] != fingerprint({k:raw[k] for k in ('input','raw','reference')})):
            raise ValueError('Source or reviewed output changed')
        if review['verdict'] != 'major':
            continue
        if not review['format_valid'] or not review['language_correct'] or not raw['ended']:
            raise ValueError('Semantic preference requires valid complete output')
        row = {**ref, 'rejected':raw['prediction'], 'preference_issue':review['note'],
            'negative_provenance':{'generation_sha256':RAW_SHA, 'generation_id':raw['id'],
                'quantization':'nf4', 'starting_adapter_sha256':PARENT}}
        validate_record({**row, 'output':row['rejected']}, True)
        if row['output'] == row['rejected']:
            raise ValueError('Identical preference')
        row['preference_review'] = {'reviewer':'Codex', 'at':now(),
            'hash':fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')})}
        pairs.append(row)
    report = {'at':now(), 'reviewed_generations':80,
        'verdicts':dict(Counter(r['verdict'] for r in decisions.values())),
        'real_error_preferences':len(pairs), 'negative_source_groups':len({r['group_id'] for r in pairs}),
        'preferences_hash':fingerprint(pairs), 'raw_sha256':RAW_SHA, 'starting_adapter_sha256':PARENT,
        'release_approved':False, 'scope':'Training-only mining yield, not independent accuracy.',
        'limitations':'Codex review, synthetic sources; natural negatives may contain multiple differences. Minor and equivalent translations excluded.'}
    write_jsonl(destination, pairs)
    write_json('runs/v8-mining-acceptance.json', report)
    print(report)

if __name__ == '__main__':
    main()
