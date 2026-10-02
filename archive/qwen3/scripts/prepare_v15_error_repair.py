"""Freeze actual reviewed TRAIN errors and diverse positive replay; no DEV mining."""
import copy
import hashlib
import json
import random
from pathlib import Path
from archive.qwen3.scripts.review_v12_factorial import accept_manual
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def reviewed_error_pairs(references, generations, judgments):
    refs = {r['id']:r for r in references}
    notes = {r['id']:r for r in judgments}
    pairs = []
    for generation in generations:
        judgment = notes[generation['id']]
        accept_manual(generation, judgment)
        reference = refs[generation['id']]
        if generation['input'] != reference['input'] or generation['reference'] != reference['output']:
            raise ValueError('Error evidence no longer binds training source')
        validate_record(reference, True, True, purpose='training')
        if judgment['verdict'] not in ('major', 'critical'):
            continue
        if not generation.get('ended') or not generation.get('prediction'):
            raise ValueError('Semantic repair requires complete parsed negative')
        row = copy.deepcopy(reference)
        row.update(rejected=generation['prediction'], preference_issue=judgment['note'],
                   evidence={'generation_hash':fingerprint(generation), 'judgment_hash':fingerprint(judgment),
                             'verdict':judgment['verdict'], 'scope':'Existing TRAIN only'})
        if row['output'] == row['rejected']:
            raise ValueError('Identical chosen and rejected')
        row['preference_review'] = {'reviewer':'Codex', 'status':'approved',
            'method':'Source-grounded actual-error judgment; explicitly authorize only major/critical actual outputs as negatives.',
            'hash':fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')})}
        pairs.append(row)
    return pairs


def main():
    destination = Path('data/prepared/v15-error-repair')
    if destination.exists():
        raise ValueError('Preserve frozen error-repair plan')
    decoding = json.loads(Path('runs/v15-training-triage-decoding-plan.json').read_text(encoding='utf-8'))
    refs, generations = read_jsonl(decoding['input']), read_jsonl(decoding['output'])
    if fingerprint(refs) != decoding['data_hash'] or len(generations) != 64:
        raise ValueError('Incomplete or changed TRAIN triage')
    pairs = reviewed_error_pairs(refs, generations, read_jsonl('runs/v15-training-triage-manual.jsonl'))
    if len(pairs) != 11 or sum(p['evidence']['verdict']=='critical' for p in pairs) != 1:
        raise ValueError('Expected eleven individually reviewed semantic errors')
    prior = json.loads(Path('data/prepared/v14-fact-trial/plan.json').read_text(encoding='utf-8'))
    old, public = read_jsonl('data/prepared/v11-mixed/train.jsonl'), read_jsonl('data/prepared/v12-public-short/train.jsonl')
    heldout = sum((read_jsonl(spec['path']) for spec in json.loads(Path('runs/v14-evaluation-plan.json').read_text(encoding='utf-8'))['datasets'].values()), [])
    excluded_groups = {r['group_id'] for r in heldout}
    excluded_texts = {r['input']['text'].strip().casefold() for r in heldout}
    for row in pairs+old+public:
        validate_record(row, True, True, purpose='training')
        if row['group_id'] in excluded_groups or row['input']['text'].strip().casefold() in excluded_texts:
            raise ValueError('Heldout source leaked into repair/replay')
    rng = random.Random(42)
    # Up to seven anchors per category/direction for each source pool, then select64.
    def anchors(pool):
        selected = []
        for category in ('daily','travel','food','academic','hard'):
            for direction in ('en','zh-CN'):
                bucket = [r for r in pool if r['category']==category and r['input']['target_lang']==direction]
                rng.shuffle(bucket)
                selected.extend(bucket[:7])
        rng.shuffle(selected)
        if len(selected)<64:
            raise ValueError('Insufficient diverse TRAIN anchors')
        return selected[:64]
    old_replay, public_replay = anchors(old), anchors(public)
    replay = old_replay+public_replay
    order = list(range(len(pairs)))
    visits = []
    while len(visits)<64:
        rng.shuffle(order)
        visits.extend(order)
    schedule = [{'pair':pairs[index]['id'],'replay':[old_replay[i]['id'],public_replay[i]['id']]}
                for i,index in enumerate(visits[:64])]
    start = Path(prior['starting_adapter'])
    if hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()!=prior['starting_adapter_sha256']:
        raise ValueError('Best parent changed')
    write_jsonl(destination/'pairs.jsonl',pairs)
    write_jsonl(destination/'replay.jsonl',replay)
    plan = {k:prior[k] for k in ('starting_adapter','starting_adapter_sha256','base','prompt_hash')}
    plan.update(at=now(), pairs_path=str(destination/'pairs.jsonl'), pairs_hash=fingerprint(pairs),
        replay_path=str(destination/'replay.jsonl'), replay_hash=fingerprint(replay), schedule=schedule,
        output='models/witrans-4b-v15-error-cpo', seed=42, updates=16, accumulation=4,
        learning_rate=2e-6, beta=.1, pair_weight=.5, replay_weight=.5,
        selected_checkpoint=16, quantization='nf4', max_length=1024, memory_cap_gib=6.5,
        selection='Predeclared final16; no DEV checkpoint selection', original_base_comparison=False,
        scope='11 existing TRAIN errors,64 preference visits,128 old/public positive anchors. Training recall is not generalization; no method superiority claim.',
        negative_review_hash=fingerprint(pairs), release_approved=False)
    write_json(destination/'plan.json',plan)
    print({'plan':str(destination/'plan.json'),'errors':len(pairs),'replay':len(replay),'plan_hash':fingerprint(plan)})


if __name__=='__main__':
    main()
