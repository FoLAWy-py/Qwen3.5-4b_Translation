"""Freeze one conservative, direct chosen-SFT engineering trial before GPU work."""
import hashlib
import random
from collections import Counter
from pathlib import Path

from scripts.decode_qwen35_repair_recall import load
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record


def main():
    dest = Path('data/prepared/qwen35-v3-reviewed23-sft-repair')
    assert not dest.exists()
    audit_path = 'runs/qwen35-v3-focused32-negative-final-audit.json'
    audit = load(audit_path)
    assert audit['reviewed'] == 32 and audit['retained_major_negatives'] == 23
    pairs = read_jsonl(audit['outputs']['pairs'])
    assert fingerprint(pairs) == audit['hashes']['pairs']
    original_path = 'data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl'
    original = read_jsonl(original_path)
    assert fingerprint(original) == 'fbeb94a8b28245a9cf5c3c91e7fdf9febc39d3aefab77214de6f627e57dca6e7'
    lexical = load('runs/qwen35-v3-lexical-negative-correction.json')
    added = read_jsonl(lexical['outputs']['supplement'])
    assert fingerprint(added) == lexical['new_hashes']['supplement']
    replay = original + added
    assert len(replay) == 456 and len(added) == 40
    assert len({r['group_id'] for r in replay}) == 220
    for row in pairs + replay:
        validate_record(row, True, True, purpose='training')
    gradient = load(audit['outputs']['gradient'])
    assert fingerprint(gradient) == audit['hashes']['gradient']
    assert gradient['clean_pairs_hash'] == fingerprint(pairs)
    assert gradient['overall']['rows'] == 23
    prior_diagnosis = load('runs/qwen35-v3-reviewed33-quarantined-recall-diagnosis.json')
    prior_update = load('runs/qwen35-v3-reviewed33-weight-update-diagnostics.json')
    assert prior_diagnosis['candidate_eligible'] is False
    owner_path = 'runs/qwen35-v3-reviewed33-recall-job.json'
    owner = load(owner_path)
    assert owner['status'] == 'finished' and owner['exit_code'] == 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-4B', local_files_only=True, trust_remote_code=False)
    pair_tokens = {r['id']: len(encode_example(tokenizer, r, 1024)['input_ids']) +
        len(encode_example(tokenizer, {**r, 'output': r['rejected']}, 1024)['input_ids']) for r in pairs}
    replay_tokens = {r['id']: len(encode_example(tokenizer, r, 1024)['input_ids']) for r in replay}
    rng = random.Random(20261002)
    def cycles(ids, total):
        result = []
        while len(result) < total:
            part = list(ids)
            rng.shuffle(part)
            result.extend(part)
        return result[:total]
    pair_ids = cycles([r['id'] for r in pairs], 256)
    keys = [c + '/' + d for c in ('daily', 'travel', 'food', 'academic', 'hard') for d in ('en', 'zh-CN')]
    strata = {key: [r['id'] for r in replay if r['category'] + '/' + r['input']['target_lang'] == key] for key in keys}
    rng.shuffle(keys)
    original_ids = {r['id'] for r in original}
    replay_ids = []
    for i, key in enumerate(keys):
        base_ids = [rid for rid in strata[key] if rid in original_ids]
        new_ids = [rid for rid in strata[key] if rid not in original_ids]
        mandatory = base_ids * 2 + new_ids * 5
        target = 122 if i < 6 else 121
        assert len(mandatory) <= target
        replay_ids.extend(mandatory + cycles(strata[key], target - len(mandatory)))
    rng.shuffle(replay_ids)
    assert len(replay_ids) == 1216
    exposure = Counter(replay_ids)
    assert all(exposure[r['id']] >= 2 for r in original)
    assert all(exposure[r['id']] >= 5 for r in added)
    schedule = []
    offset = 0
    for i, rid in enumerate(pair_ids):
        count = 5 if i % 4 < 3 else 4
        schedule.append(dict(pair=rid, replay=replay_ids[offset:offset+count]))
        offset += count
    assert offset == 1216 and len(schedule) == 256
    full_tokens = sum(pair_tokens[s['pair']] + sum(replay_tokens[rid] for rid in s['replay']) for s in schedule)
    old_plan = load('data/prepared/qwen35-v3-reviewed33-repair/plan.json')
    preservation = old_plan['recall_sets']['preservation_ids']
    view = read_jsonl(audit['outputs']['view'])
    assert fingerprint(view) == audit['hashes']['view']
    assert set(preservation) <= {r['id'] for r in view if r['verdict'] == 'pass'}
    original_by_id = {r['id']: r for r in original}
    recall = [{k: r[k] for k in ('id','group_id','category','input','output','source','review')} for r in pairs]
    recall += [original_by_id[rid] for rid in preservation] + added
    assert len(recall) == len({r['id'] for r in recall}) == 103
    start = Path('models/witrans-qwen35-v2-critical-cpo')
    with (start/'adapter_model.safetensors').open('rb') as stream:
        start_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert start_sha == 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
    plan = dict(at=now(), base_dir='models/Qwen3.5-4B', base=load('models/Qwen3.5-4B/witrans_base.json'),
        prompt_hash=fingerprint(SYSTEM_PROMPT), starting_adapter=str(start), starting_adapter_sha256=start_sha,
        owner_job=owner_path, pairs_path=str(dest/'pairs.jsonl'), pairs_hash=fingerprint(pairs),
        replay_path=str(dest/'replay.jsonl'), replay_hash=fingerprint(replay),
        schedule=schedule, full_token_budget=full_tokens, seed=20261002, updates=64, accumulation=4,
        learning_rate=2e-6, beta=.1, pair_weight=.1, replay_weight=.9, lora_dropout=.05,
        repair_objective='sft', method='direct_chosen_sft_with_lower_learning_rate_and_balanced_retention_replay',
        quantization='nf4', max_length=1024, memory_cap_gib=6.5, selected_checkpoint=64,
        output='models/witrans-qwen35-v3-reviewed23-sft',
        hypothesis='Direct chosen SFT from original frozen weights, at LR2e-6 and error/replay weights.1/.9, with each original positive visited at least2 times and each context supplement at least5 times, will repair at least14/23 actual major errors while preserving39/40 old pass and reaching32/40 supplement pass. Any frozen gate failure rejects this engineering candidate.',
        rationale='The preserved ineligible CPO trial still missed stock identity, breakfast morning, key timing and three context relations, and retained only36/40 with24/40 context pass. Positive chosen/preference gradient alignment did not imply generation success; all64 updates clipped and relative adapter change was1.96%. Test ordinary chosen-NLL supervision rather than another preference round, remove critical-span upweighting, lower learning rate, and increase replay exposure. Multiple factors and compute change; this is not a causal SFT versus CPO comparison.',
        selection='Predeclared final64 only; no intermediate loss or DEV checkpoint selection.',
        budget=dict(pair_visits=256, replay_visits=1216, forwards=1728, backwards=1472,
            full_prompt_answer_eos_tokens=full_tokens, replay_source_groups=220,
            minimum_original_positive_visits=2, minimum_supplement_positive_visits=5,
            error_visits=dict(Counter(pair_ids)), positive_exposure_hash=fingerprint(dict(exposure)),
            replay_strata=dict(Counter(next(r['category']+'/'+r['input']['target_lang'] for r in replay if r['id']==rid) for rid in replay_ids))),
        recall_path=str(dest/'recall.jsonl'), recall_hash=fingerprint(recall),
        recall_sets=dict(actual_error_ids=[r['id'] for r in pairs], preservation_ids=preservation,
            supplement_ids=[r['id'] for r in added], direction_error_ids=['v8-multi-011-zh-CN','v8-multi-018-zh-CN','v4-mining-009-zh-CN']),
        screening_rule=dict(error_pass_at_least=14, error_major_at_most=9, preservation_pass_at_least=39,
            hard_en_preservation_pass=4, supplement_pass_at_least=32, supplement_major_at_most=4,
            critical=0, all_format_direction_eos=True, all_three_old_direction_errors_correct=True,
            if_failed='Preserve final checkpoint, individually diagnose A/B/C. No DEV decoding, adaptive extension or retry.',
            if_passed='Decode known200/public116 once and individually review against both frozen controls; then formal own-weight timing before final freeze and new independent confirmation.'),
        stop_conditions=['Nonfinite loss/gradient, OOM or CPU parameter offload: stop and preserve logs.',
            'Peak reserved memory>6.5GiB: stop before optimizer update.',
            'Any frozen source/actual-negative/budget/start identity mismatch: stop before training.',
            'Fixed64 updates and exact token budget. TRAIN gate failure rejects candidate; no adaptive extension.'],
        evidence=dict(focused_negative_audit_path=audit_path, focused_negative_audit_hash=fingerprint(audit),
            actual_gradient_report_hash=fingerprint(gradient), prior_failed_diagnosis_hash=fingerprint(prior_diagnosis),
            prior_update_diagnostics_hash=fingerprint(prior_update),
            positive_age_view_hash=lexical['new_hashes']['supplement'], original416_hash=fingerprint(original),
            source_family_audit_hash=fingerprint(read_jsonl('runs/v16-family-source-audit.jsonl')),
            supplemental_audit_hash=fingerprint(load('runs/qwen35-v3-context-supplement-final-audit.json'))),
        runtime='Existing uncompiled NF4 trainer, ordinary chosen answer NLL with EOS; rejected forward is no-grad diagnostic only. BF16 tied vocabulary frozen; other nonquant preparationFP32; LoRA FP32 and dropout.05. No dependency or inference runtime change.',
        historical_scope='All prior executed Qwen35 optimization trials quarantined due to corrected negatives. Do not inherit their weights or count them as the valid optimization round. Prior raw outcomes guide diagnosis only.',
        scope='One predeclared engineering candidate; TRAIN recall does not prove generalization or release. No method superiority claim or new raw-base comparison.',
        stage_goal_complete=False, release_approved=False, default_promoted=False)
    write_jsonl(dest/'pairs.jsonl', pairs)
    write_jsonl(dest/'replay.jsonl', replay)
    write_jsonl(dest/'recall.jsonl', recall)
    write_json(dest/'plan.json', plan)
    print(dict(plan_hash=fingerprint(plan), full_tokens=full_tokens, pairs=23, recall=103, budget=plan['budget']))


if __name__ == '__main__':
    main()
