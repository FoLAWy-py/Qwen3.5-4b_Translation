"""Freeze one broader actual-error repair trial with audited balanced replay."""
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from archive.qwen3.scripts.prepare_v15_error_repair import reviewed_error_pairs
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.critical_spans import encode_critical_spans, validated_annotations
from witrans_tools.data import encode_example, validate_record

NEW_SPANS = {
    'public-short-0355-zh-CN': ['这我就只跟你说'],
    'v10-travel-08-en': ['a message about arrival'],
    'v2-replay-v1-travel-005-02-zh-CN': ['托运行李'],
    'v9-constraint-012-zh-CN': ['导游'],
    'v2-train-053-en': ["we'd each like our own dessert"],
    'v9-constraint-022-en': ['eighty percent of the event happened'],
    'v2-train-083-zh-CN': ['速率', '有加速度'],
    'v9-constraint-024-zh-CN': ['测量结果一致'],
    'v7-context-020-zh-CN': ['吠叫声'],
}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    dest = Path('data/prepared/v19-balanced-repair')
    assert not dest.exists(), 'Preserve frozen experiment'
    prior = load('data/prepared/v17-strength-trial/plan.json')
    for path in ('runs/v18-sft-control-job.json', 'runs/v19-new-error-triage-job.json'):
        owner = load(path)
        assert owner['status'] == 'finished' and owner['exit_code'] == 0
    triage = load('data/prepared/v17-new-error-triage/plan.json')
    refs, generated = read_jsonl(triage['input']), read_jsonl(triage['output'])
    notes = read_jsonl('runs/v17-new-error-triage-manual.jsonl')
    assert len(refs) == len(generated) == len(notes) == 40
    assert fingerprint(refs) == triage['data_hash']
    summary = load('runs/v17-new-error-triage.summary.json')
    assert summary['adapter_sha256'] == triage['adapter_sha256']
    assert summary['data_hash'] == fingerprint(refs) and summary['prompt_hash'] == prior['prompt_hash']
    new_pairs = reviewed_error_pairs(refs, generated, notes)
    assert {p['id'] for p in new_pairs} == set(NEW_SPANS), 'Explicit source readings and spans must match'
    old_pairs = read_jsonl(prior['pairs_path'])
    pairs = old_pairs + new_pairs
    assert len(pairs) == len({p['id'] for p in pairs}) == 20
    manifest = load('data/prepared/v16-reviewed-families/resolved-manifest.json')
    replay = read_jsonl(manifest['train_ready_path'])
    assert fingerprint(replay) == manifest['train_ready_hash'] and len(replay) == 416
    heldout = read_jsonl('data/prepared/v12-factorial-v2/dev.jsonl') + read_jsonl('data/prepared/v12-factorial-v2/public-dev.jsonl')
    normalize = lambda s: ' '.join(s.split()).casefold()
    for row in pairs + replay:
        validate_record(row, True, True, purpose='training')
    assert not {r['group_id'] for r in pairs + replay} & {r['group_id'] for r in heldout}
    assert not {normalize(r['input']['text']) for r in pairs + replay} & {normalize(r['input']['text']) for r in heldout}
    old_annotations = load(prior['critical_annotations'])
    annotations = copy.deepcopy(old_annotations['annotations'])
    for row in new_pairs:
        spans = NEW_SPANS[row['id']]
        assert all(row['output']['translation'].count(s) == 1 for s in spans), row['id']
        content = {k: row[k] for k in ('input', 'output', 'rejected', 'preference_issue')}
        annotations.append(dict(id=row['id'], critical_spans=spans, multiplier=3.0, reviewer='Codex',
            at=now(), note=row['preference_issue'],
            content_hash=fingerprint({**content, 'critical_spans':spans, 'multiplier':3.0})))
    annotation_report = dict(at=now(), train_hash=fingerprint(pairs), annotations=annotations,
        annotation_hash=fingerprint(annotations), scope='Twenty individually inspected actual TRAIN errors only.',
        release_approved=False)
    validated_annotations(pairs, annotation_report)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B', local_files_only=True)
    entries = {a['id']:a for a in annotations}
    encoded_pairs = {r['id']:(encode_critical_spans(tokenizer,r,entries[r['id']]['critical_spans'],1024,3.0),
        encode_example(tokenizer,{**r,'output':r['rejected']},1024)) for r in pairs}
    encoded_replay = {r['id']:encode_example(tokenizer,r,1024) for r in replay}
    rng = random.Random(42)
    def cycles(rows, n):
        order = []
        while len(order) < n:
            cycle = [r['id'] for r in rows]
            rng.shuffle(cycle)
            order.extend(cycle)
        return order[:n]
    pair_ids, replay_ids = cycles(pairs,256), cycles(replay,512)
    schedule = [dict(pair=key,replay=replay_ids[2*i:2*i+2]) for i,key in enumerate(pair_ids)]
    full_tokens = sum(sum(len(e['input_ids']) for e in encoded_pairs[s['pair']]) +
        sum(len(encoded_replay[k]['input_ids']) for k in s['replay']) for s in schedule)
    assert len(set(replay_ids)) == 416 and len({r['group_id'] for r in replay}) == 200
    by_id = {r['id']:r for r in replay}
    visited_strata = Counter(by_id[k]['category']+'/'+by_id[k]['input']['target_lang'] for k in replay_ids)
    assert min(visited_strata.values()) >= 40 and max(visited_strata.values()) <= 65
    parent = Path(prior['starting_adapter'])/'adapter_model.safetensors'
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == prior['starting_adapter_sha256']
    plan = {k:prior[k] for k in ('starting_adapter','starting_adapter_sha256','base','prompt_hash')}
    plan.update(at=now(), pairs_path=str(dest/'pairs.jsonl'),pairs_hash=fingerprint(pairs),
        replay_path=str(dest/'replay.jsonl'),replay_hash=fingerprint(replay),
        critical_annotations=str(dest/'annotations.json'),critical_annotations_hash=fingerprint(annotation_report),
        schedule=schedule,output='models/witrans-4b-v19-balanced-error-cpo',seed=42,updates=64,accumulation=4,
        learning_rate=5e-6,beta=.1,pair_weight=.5,replay_weight=.5,selected_checkpoint=64,
        quantization='nf4',max_length=1024,memory_cap_gib=6.5,full_token_budget=full_tokens,
        method='reviewed_broader_actual_errors_and_balanced_replay_critical_span_cpo',repair_objective='cpo',
        hypothesis='Broader source-grounded errors in both directions and complete balanced200-family replay, with more repeated supervision at lower per-step strength, repair at least6 of20 actual TRAIN errors without a critical reversal; development gates still decide generalization.',
        changed_factor='One diagnostic engineering package: broader reviewed errors and balanced replay with64 lower-rate updates. Multiple implementation factors change; no causal component or stable algorithm claim.',
        selection='Predeclared final64 only; no loss-based DEV checkpoint selection.',
        budget_scope='Frozen full prompt+chosen+rejected+replay+EOS tokens;1024 forwards,768 backward invocations,64 updates. Every one of416 replay rows and200 families visited. No adaptive budget.',
        screening_rule=dict(train_recall_pass_at_least=6,train_recall_critical=0,format_direction_eos_all=True,
            original11_pass_at_least=3,new9_pass_at_least=2,
            if_failed='Preserve evidence and diagnose remaining TRAIN errors; no automatic DEV generation.',
            if_passed='Freeze candidate weights and decode known200/public116 once, then individually grade all changed translations and enforce stage regression gates.'),
        gpu_predecessor_job='runs/v19-new-error-triage-job.json',
        source_audit_manifest_hash=fingerprint(manifest),triage_decisions_hash=fingerprint(notes),
        replay_strata=dict(visited_strata),pair_strata=dict(Counter(p['category']+'/'+p['input']['target_lang'] for p in pairs)),
        original_base_comparison=False,scope='TRAIN-only semantic diagnosis then one candidate regression screen; knownDEV is repeatedly used, no independent confirmation yet.',
        release_approved=False)
    write_jsonl(dest/'pairs.jsonl',pairs)
    write_jsonl(dest/'replay.jsonl',replay)
    write_json(dest/'annotations.json',annotation_report)
    write_json(dest/'plan.json',plan)
    write_json('runs/v19-new-error-triage-screening.json',dict(at=now(),rows=40,source_groups=40,
        counts=dict(Counter(n['verdict'] for n in notes)),decisions_hash=fingerprint(notes),
        actual_generation_hash=fingerprint(generated),new_negative_pairs=9,
        scope='TRAIN-only high-NLL priority, individual reading. Not heldout accuracy.',release_approved=False))
    print({'plan_hash':fingerprint(plan),'full_tokens':full_tokens,'pairs':20,'replay_rows':416,
        'replay_strata':dict(visited_strata),'triage_counts':dict(Counter(n['verdict'] for n in notes))})


if __name__ == '__main__':
    main()
