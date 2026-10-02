"""Freeze four-arm NF4 SFT screening with exact full-sequence token budgets."""
import hashlib
import json
import random
from bisect import bisect_left
from collections import Counter
from pathlib import Path

from archive.qwen3.runtime import SYSTEM_PROMPT
from witrans_tools.common import base_manifest, fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record

DEST = Path('data/prepared/v12-factorial-v2')


def exact_schedule(rows, encoded, budget, updates, seed):
    lengths = [len(row['input_ids']) for row in encoded]
    total_budget = budget*updates
    tail_limit = min(total_budget, 8*max(lengths))
    reachable = [False] * (tail_limit+1)
    reachable[0] = True
    unique = sorted(set(lengths))
    for total in range(1, tail_limit+1):
        reachable[total] = any(length <= total and reachable[total-length] for length in unique)
    rng = random.Random(seed)
    remaining, flat = total_budget, []
    while remaining > tail_limit:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for chosen in order:
            if remaining <= tail_limit:
                break
            if lengths[chosen] > remaining:
                raise ValueError('Invalid whole-record budget')
            flat.append(chosen)
            remaining -= lengths[chosen]
    while not reachable[remaining] and flat:
        remaining += lengths[flat.pop()]
        if remaining > tail_limit:
            raise ValueError('Exact whole-record token budget is infeasible')
    if not reachable[remaining]:
        raise ValueError('Exact whole-record token budget is infeasible')
    while remaining:
        choices = [i for i, length in enumerate(lengths) if length <= remaining and reachable[remaining-length]]
        chosen = rng.choice(choices)
        flat.append(chosen)
        remaining -= lengths[chosen]
    if len(flat) < updates:
        raise ValueError('Too few whole records for optimizer updates')
    cumulative = [0]
    for chosen in flat:
        cumulative.append(cumulative[-1]+lengths[chosen])
    schedule, previous = [], 0
    for step in range(1, updates):
        cut = bisect_left(cumulative, step*budget)
        candidates = [index for index in (cut-1,cut) if previous < index <= len(flat)-(updates-step)]
        if not candidates:
            raise ValueError('Cannot partition nonempty optimizer updates')
        cut = min(candidates, key=lambda index: abs(cumulative[index]-step*budget))
        schedule.append([rows[index]['id'] for index in flat[previous:cut]])
        previous = cut
    schedule.append([rows[index]['id'] for index in flat[previous:]])
    return schedule


def main():
    if DEST.exists():
        raise ValueError('Preserve frozen factorial plan')
    comparison = json.loads(Path('runs/v12-blind-comparison.json').read_text(encoding='utf-8'))
    if comparison['rows'] != 200 or comparison['frozen_decision_hash'] != '6b77eab99ae56d60feb604ee2fcfe719cbada0b3cfee823ef5025c0d4f66df39':
        raise ValueError('Complete masked comparison required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B', local_files_only=True)
    old = read_jsonl('data/prepared/v11-mixed/train.jsonl')
    public = read_jsonl('data/prepared/v12-public-short/train.jsonl')
    public_dev = read_jsonl('data/prepared/v12-public-short/dev.jsonl')
    dev = read_jsonl('data/prepared/v4/dev.jsonl')
    if fingerprint(old) != 'fc408b20388c59afabab26d14f481600d54718287a98d0e369b0769b685ddefe':
        raise ValueError('Historical reviewed training pool changed')
    pm = json.loads(Path('data/prepared/v12-public-short/manifest.json').read_text(encoding='utf-8'))
    for name, rows in (('train', public), ('dev', public_dev)):
        if fingerprint(rows) != pm['splits'][name]['hash']:
            raise ValueError('Public frozen split changed')
    all_dev = dev + public_dev
    for purpose, rows in (('training', old+public), ('evaluation', all_dev)):
        for row in rows:
            validate_record(row, True, True, purpose=purpose)
    if {r['group_id'] for r in old+public} & {r['group_id'] for r in all_dev}:
        raise ValueError('Train/development family overlap')
    normalize = lambda s: ' '.join(s.split()).casefold()
    if {normalize(r['input']['text']) for r in old+public} & {normalize(r['input']['text']) for r in all_dev}:
        raise ValueError('Train/development exact source overlap')
    pools = {'old': old, 'public': public}
    encodings = {name: [encode_example(tokenizer, row, 1024) for row in rows] for name, rows in pools.items()}
    for row in all_dev:
        encode_example(tokenizer, row, 1024)
    updates, budget = 64, 4096
    old_schedule = exact_schedule(old, encodings['old'], budget, updates, 42)
    old_half = exact_schedule(old, encodings['old'], budget//2, updates, 42)
    public_half = exact_schedule(public, encodings['public'], budget//2, updates, 43)
    mixed_schedule = [a+b for a, b in zip(old_half, public_half)]
    datasets = {'old': old, 'mixed': old+public}
    schedules = {'old': old_schedule, 'mixed': mixed_schedule}
    lengths = {row['id']: len(enc['input_ids']) for name in pools for row, enc in zip(pools[name], encodings[name])}
    labels = {row['id']: sum(v != -100 for v in enc['labels']) for name in pools for row, enc in zip(pools[name], encodings[name])}
    plan = {'at': now(), 'base': base_manifest('models/Qwen3-4B'), 'prompt_hash': fingerprint(SYSTEM_PROMPT),
            'seed': 42, 'updates': updates, 'tokens_per_update': budget, 'training_input_tokens_per_arm': updates*budget,
            'learning_rate': 2e-5, 'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': .05,
            'max_length': 1024, 'quantization': 'nf4', 'checkpoints': [32,64],
            'loss': 'Answer/EOS cross entropy averaged over supervised tokens in each update; prompt ignored',
            'selection': 'Equal-weight mean of known DEV200 NLL and public auxiliary DEV116 NLL; lowest among checkpoint32/64. Semantic gates remain mandatory.',
            'budget_scope': 'Exact TOTAL prompt+answer+EOS input tokens, no truncation or padding, microbatch1. Mixed arm uses131072 old+131072 public tokens in total. Per-update token counts vary around4096. Shuffle-cycle coverage precedes exact whole-record tail fill; supervised tokens and attention costs differ, not identical FLOPs.',
            'supersedes': 'data/prepared/v12-factorial/plan.json rejected before training: exact per-update filling concentrated one public sample150 visits',
            'starting_adapter': 'models/witrans-4b-v7-critical-cpo',
            'starting_adapter_sha256': '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424',
            'dev': {'path': str(DEST/'dev.jsonl'), 'hash': fingerprint(dev)},
            'public_dev': {'path': str(DEST/'public-dev.jsonl'), 'hash': fingerprint(public_dev)},
            'datasets': {}, 'arms': [], 'release_approved': False,
            'limits': 'Single seed screening with common LR; fresh LoRA and continued adapter may have different optimum LR. Public pool mostly daily, so all ten known strata must be inspected before promotion. Masked review is not independent blinding.'}
    for name, rows in datasets.items():
        schedule = schedules[name]
        if sum(lengths[key] for batch in schedule for key in batch) != budget*updates:
            raise ValueError('Budget mismatch')
        visits = Counter(key for batch in schedule for key in batch)
        if max(visits.values()) > 10:
            raise ValueError('Excessive sample concentration; revise schedule before training')
        by_id = {row['id']: row for row in rows}
        diagnostics = Counter(by_id[key]['category']+'/'+by_id[key]['input']['target_lang'] for key in visits.elements())
        write_jsonl(DEST/f'{name}-train.jsonl', rows)
        write_json(DEST/f'{name}-schedule.json', schedule)
        plan['datasets'][name] = {'path': str(DEST/f'{name}-train.jsonl'), 'hash': fingerprint(rows),
                                  'schedule_path': str(DEST/f'{name}-schedule.json'), 'schedule_hash': fingerprint(schedule),
                                  'record_visits': sum(visits.values()), 'unique_visited_rows': len(visits),
                                  'max_repetition': max(visits.values()), 'visited_strata': dict(diagnostics),
                                  'update_full_tokens': [sum(lengths[key] for key in batch) for batch in schedule],
                                  'supervised_tokens': sum(labels[key] for batch in schedule for key in batch)}
        for start in ('base', 'v7'):
            plan['arms'].append({'name': f'{start}-{name}', 'start': start, 'dataset': name,
                                 'output': f'models/witrans-4b-v12-{start}-{name}',
                                 'selection_output': f'runs/v12-{start}-{name}-selection.json'})
    write_jsonl(DEST/'dev.jsonl', dev)
    write_jsonl(DEST/'public-dev.jsonl', public_dev)
    write_json(DEST/'plan.json', plan)
    print({'input_tokens_per_arm': updates*budget, 'datasets': plan['datasets'], 'arms': plan['arms']}, flush=True)


if __name__ == '__main__':
    main()
