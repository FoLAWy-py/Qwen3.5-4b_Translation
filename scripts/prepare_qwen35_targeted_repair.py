"""Freeze one current-error repair trial after validated actual gradient diagnosis."""
import hashlib
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.critical_spans import encode_critical_spans,validated_annotations
from witrans_tools.data import encode_example,validate_record


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--clean40',action='store_true')
    modes.add_argument('--reviewed34',action='store_true')
    modes.add_argument('--reviewed33',action='store_true')
    args=parser.parse_args()
    corrected=args.clean40 or args.reviewed34 or args.reviewed33
    label='reviewed33' if args.reviewed33 else 'reviewed34' if args.reviewed34 else 'clean40' if args.clean40 else 'targeted'
    dest=Path(f'data/prepared/qwen35-v3-{label}-repair')
    assert not dest.exists(), 'Preserve frozen trials'
    gradient=load('runs/qwen35-v3-actual-error-gradient-fp64-diagnosis.json')
    owner=load('runs/qwen35-v3-gradient-fp64-job.json')
    state=load('runs/qwen35-v3-gradient-fp64.json')
    assert owner['status']=='finished' and owner['exit_code']==0 and state['status']=='finished'
    assert gradient['probe_hash']==fingerprint(state) and gradient['accounting']['optimizer_updates']==0
    assert gradient['draft_hash']==fingerprint(load('runs/qwen35-v3-error-span-draft-004.json'))
    if corrected:
        clean_gradient=load(f'runs/qwen35-v3-{label}-gradient-diagnosis.json')
        assert clean_gradient['original_probe_report_hash']==fingerprint(gradient)
        assert clean_gradient['original_executed_accounting']==gradient['accounting']
        gradient=clean_gradient
    expected_errors=33 if args.reviewed33 else 34 if args.reviewed34 else 40 if args.clean40 else 41
    assert gradient['overall']['rows']==expected_errors and gradient['overall']['chosen_preference_conflict']==0
    assert gradient['overall']['negative_summed_margin']>=20
    assert gradient['overall']['distributions']['chosen_preference_cosine']['minimum']>0
    pool=load('runs/qwen35-v3-reviewed33-current-error-pool-audit.json' if args.reviewed33 else 'runs/qwen35-v3-reviewed-current-error-pool-audit.json' if args.reviewed34 else 'runs/qwen35-v3-clean-current-error-pool-audit.json' if args.clean40 else 'runs/qwen35-v3-current-error-pool-final-audit.json')
    pairs=read_jsonl(pool['pool_path'])
    assert fingerprint(pairs)==pool['pairs_hash'] and len(pairs)==expected_errors
    if corrected:
        assert gradient['clean_pairs_hash']==pool['pairs_hash']
        measured=read_jsonl('runs/qwen35-v3-gradient-fp64.jsonl')
        selected=[row for row in measured if row['id'] in {pair['id'] for pair in pairs}]
        assert fingerprint(selected)==gradient['selected_rows_hash'] and len(selected)==expected_errors
        assert 'v10-daily-19-en' not in {pair['id'] for pair in pairs}
    if args.reviewed34 or args.reviewed33:
        rereview=read_jsonl('runs/qwen35-v3-negative-quality-rereview.jsonl')
        assert fingerprint(rereview)==pool['quality_rereview_hash'] and len(rereview)==40
        assert not set(pool['excluded_nonmajor_ids']) & {pair['id'] for pair in pairs}
    original=read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    assert fingerprint(original)==pool['train_hash'] and len(original)==416
    supplemental=load('runs/qwen35-v3-context-supplement-final-audit.json')
    assert supplemental['ready_for_training']
    added=read_jsonl(supplemental['train_path'])
    assert fingerprint(added)==supplemental['train_hash'] and len(added)==40
    if args.reviewed33:
        lexical=load('runs/qwen35-v3-lexical-negative-correction.json')
        assert fingerprint(lexical)==pool['correction_report_hash'] and lexical['old_trial_eligible'] is False
        added=read_jsonl(lexical['outputs']['supplement'])
        assert fingerprint(added)==lexical['new_hashes']['supplement'] and len(added)==40
    replay=original+added
    assert len({r['id'] for r in replay})==456 and len({r['group_id'] for r in replay})==220
    for row in pairs+replay:
        validate_record(row,True,True,purpose='training')
    draft=load('runs/qwen35-v3-error-span-draft-004.json')
    if not corrected:
        assert gradient['draft_hash']==fingerprint(draft)
    spans={entry['id']:entry for entry in draft['entries']}
    if corrected:
        spans={rid:entry for rid,entry in spans.items() if rid in {row['id'] for row in pairs}}
    assert set(spans)=={row['id'] for row in pairs}
    annotations=[]
    for row in pairs:
        entry=spans[row['id']]
        assert entry['chosen']==row['output'] and entry['rejected']==row['rejected']
        content={key:row[key] for key in ('input','output','rejected','preference_issue')}
        annotations.append(dict(id=row['id'],critical_spans=entry['critical_spans'],multiplier=3.0,
            reviewer='Codex',at=entry['at'],note=entry['note'],
            content_hash=fingerprint(dict(**content,critical_spans=entry['critical_spans'],multiplier=3.0))))
    annotation_report=dict(at=now(),train_hash=fingerprint(pairs),annotations=annotations,
        annotation_hash=fingerprint(annotations),scope=f'Explicit previously reviewed semantic spans in{expected_errors} real current errors only.')
    validated_annotations(pairs,annotation_report)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    encoded_pairs={row['id']:(encode_critical_spans(tokenizer,row,spans[row['id']]['critical_spans'],1024,3.0),
        encode_example(tokenizer,{**row,'output':row['rejected']},1024)) for row in pairs}
    encoded_replay={row['id']:encode_example(tokenizer,row,1024) for row in replay}
    rng=random.Random(20261002)
    def cycles(ids,total):
        result=[]
        while len(result)<total:
            cycle=list(ids);rng.shuffle(cycle);result.extend(cycle)
        return result[:total]
    pair_ids=cycles([row['id'] for row in pairs],256)
    strata={category+'/'+direction:[row['id'] for row in replay if row['category']==category and row['input']['target_lang']==direction]
        for category in ('daily','travel','food','academic','hard') for direction in ('en','zh-CN')}
    stratum_order=list(strata);rng.shuffle(stratum_order)
    replay_ids=[]
    for index,key in enumerate(stratum_order):
        if args.reviewed33:
            original_ids=[rid for rid in strata[key] if not rid.startswith('q35-v3-supplement-')]
            new_ids=[rid for rid in strata[key] if rid.startswith('q35-v3-supplement-')]
            mandatory=original_ids+new_ids*3
            target=71 if index<4 else 70
            assert len(mandatory)<=target
            replay_ids.extend(mandatory+cycles(strata[key],target-len(mandatory)))
        else:
            replay_ids.extend(cycles(strata[key],52 if index<2 else 51))
    rng.shuffle(replay_ids)
    expected_replay=704 if args.reviewed33 else 512
    assert len(replay_ids)==expected_replay and set(replay_ids)==set(encoded_replay)
    schedule=[]
    offset=0
    for i,key in enumerate(pair_ids):
        count=3 if args.reviewed33 and i%4<3 else 2
        schedule.append(dict(pair=key,replay=replay_ids[offset:offset+count]))
        offset+=count
    assert offset==expected_replay
    full_tokens=sum(sum(len(example['input_ids']) for example in encoded_pairs[visit['pair']])+
        sum(len(encoded_replay[key]['input_ids']) for key in visit['replay']) for visit in schedule)
    review=read_jsonl('runs/qwen35-v3-train-discovery-lexical-corrected-manual.jsonl' if args.reviewed33 else 'runs/qwen35-v3-train-discovery-conservative-manual.jsonl' if args.reviewed34 else 'runs/qwen35-v3-train-discovery-corrected-manual.jsonl' if args.clean40 else 'runs/qwen35-v3-train-discovery-recovery1-manual.jsonl')
    assert fingerprint(review)==pool['review_hash']
    pass_ids={row['id'] for row in review if row['verdict']=='pass'}
    preservation=[]
    for key,ids in strata.items():
        eligible=[rid for rid in ids if rid in pass_ids]
        assert len(eligible)>=4
        # Predetermined source order; no candidate outputs or loss selection.
        preservation.extend(eligible[:4])
    by_id={row['id']:row for row in original}
    recall_rows=[{key:row[key] for key in ('id','group_id','category','input','output','source','review')} for row in pairs]
    recall_rows.extend(by_id[rid] for rid in preservation)
    recall_rows.extend(added)
    assert len(recall_rows)==expected_errors+80 and len({row['id'] for row in recall_rows})==expected_errors+80
    start=Path('models/witrans-qwen35-v2-critical-cpo')
    with (start/'adapter_model.safetensors').open('rb') as stream:
        start_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert start_sha=='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
    plan=dict(at=now(),base_dir='models/Qwen3.5-4B',base=load('models/Qwen3.5-4B/witrans_base.json'),
        prompt_hash=fingerprint(SYSTEM_PROMPT),starting_adapter=str(start),starting_adapter_sha256=start_sha,
        owner_job='runs/qwen35-v3-targeted-recall-job.json' if corrected else 'runs/qwen35-v3-gradient-fp64-job.json',pairs_path=str(dest/'pairs.jsonl'),pairs_hash=fingerprint(pairs),
        replay_path=str(dest/'replay.jsonl'),replay_hash=fingerprint(replay),
        critical_annotations=str(dest/'annotations.json'),critical_annotations_hash=fingerprint(annotation_report),
        schedule=schedule,full_token_budget=full_tokens,seed=20261002,updates=64,accumulation=4,
        learning_rate=5e-6,beta=.1,pair_weight=.5,replay_weight=.5,lora_dropout=.05,
        quantization='nf4',max_length=1024,memory_cap_gib=6.5,selected_checkpoint=64,
        repair_objective='cpo',method='current_qwen35_reviewed_error_span_cpo_with_balanced_reference_replay',
        output=f'models/witrans-qwen35-v3-{label}-cpo',
        hypothesis=f'{min(Counter(pair_ids).values())} to {max(Counter(pair_ids).values())} visits to each of {expected_errors} verified current errors, supported by positively aligned nonvanishing preference gradients and reviewed semantic-span supervision, will repair at least14 to pass while balanced reference replay retains at least39/40 previously correct readings. Forty new context/glossary positives address the measured supervision gap. Independent development gates decide transfer and retention.',
        rationale='Keep the existing CPO implementation because actual current preference gradients align with chosen gradients; no evidence of a saturated or antagonistic preference component. Use half the starting CPO learning rate (1e-5) for64 fixed updates instead of22, repeat current errors rather than old Qwen3 mistakes, and visit all456 positive references in ten balanced strata. The prior original22 CPO steps all clipped and changed LoRA L2 by1.14%; this is an exposure/retention engineering trial, not proof CPO is superior.',
        historical_scope='Legacyv15/v16 failed0/11 recalls;v17CPO/v18SFT each2/11. Do not infer causal superiority. This changes actual starting model/error pool/source coverage; no repeats of the original old negative set or unreviewed synonym negatives.',
        selection='Predeclared final64 only; no intermediate loss or DEV checkpoint selection.',
        budget=dict(pair_visits=256,replay_visits=512,forwards=1024,backwards=768,full_prompt_answer_eos_tokens=full_tokens,
            replay_source_groups=220,replay_strata=dict(Counter(next(row['category']+'/'+row['input']['target_lang'] for row in replay if row['id']==rid) for rid in replay_ids))),
        recall_path=str(dest/'recall.jsonl'),recall_hash=fingerprint(recall_rows),
        recall_sets=dict(actual_error_ids=[row['id'] for row in pairs],preservation_ids=preservation,
            supplement_ids=[row['id'] for row in added],direction_error_ids=['v8-multi-011-zh-CN','v8-multi-018-zh-CN','v4-mining-009-zh-CN']),
        screening_rule=dict(error_pass_at_least=14,error_major_at_most=20,preservation_pass_at_least=39,
            hard_en_preservation_pass=4,supplement_pass_at_least=32,supplement_major_at_most=4,
            critical=0,all_format_direction_eos=True,all_three_old_direction_errors_correct=True,
            if_failed='Preserve final checkpoint, individually diagnose A/B/C; no automatic DEV decoding or adaptive training extension.',
            if_passed='Decode known200/public116 once at final weights and individually review actual changes versus both frozen controls.'),
        stop_conditions=['Nonfinite loss/gradient, OOM or CPU parameter offload: stop, preserve logs, no automatic retry.',
            'Peak reserved memory>6.5GiB: stop before optimizer update.',
            'Any source/span/budget/start identity mismatch: stop before model training.',
            'Fixed64 updates/token budget only. Recall failure stops DEV generation; no loss-based efficacy claim.'],
        evidence=dict(actual_gradient_report_hash=fingerprint(gradient),current_error_audit_hash=fingerprint(pool),
            supplemental_audit_hash=fingerprint(supplemental),starting_update_strength_hash=fingerprint(load('runs/qwen35-v3-start-cpo-weight-update-diagnostics.json'))),
        runtime='Uncompiled NF4 training; CUDA parameters, nonreentrant gradient checkpointing, tied BF16 vocabulary frozen, FP32 other nonquantized preparation, inheritedLoRAdropout.05. No new optimizer or dependencies.',
        scope='One predeclared engineering candidate trial; TRAIN recall is not generalization or release acceptance. No stable algorithm claim, no new raw-base comparisons.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    if corrected:
        plan['data_correction']=dict(report_hash=fingerprint(load('runs/qwen35-v3-borderline-negative-correction.json')),
            excluded_id='v10-daily-19-en',old_trial_eligible=False,
            starting_weights='Original frozenQwen35start, never inherit quarantined trial.',
            changed_factor='One semantic-quality correction: exclude reasonable rendering from negative pool, preserve all original artifacts. Same predeclared optimization budget/objective and 456 positive references; schedule recalculated with clean40 IDs.')
        plan['rationale']=plan['rationale'].replace('current errors rather than old Qwen3 mistakes','clean current errors rather than old Qwen3 mistakes or a reasonable-synonym negative')
    if args.reviewed34:
        plan['data_correction'].update(
            excluded_ids=['v10-daily-19-en']+pool['excluded_nonmajor_ids'],
            quality_rereview_hash=pool['quality_rereview_hash'],
            corrected_review_hash=pool['review_hash'],
            changed_factor='Full source-level re-review excludes seven reasonable/minor/ambiguous negatives from the original41; reviewed34 schedule is recalculated before training. Fixed64 updates/1024 forwards/768 backwards and456 positives remain; exact token budget changes and this is not a matched method-control experiment.')
        plan['data_correction'].pop('excluded_id')
    if args.reviewed33:
        plan.update(owner_job='runs/qwen35-v3-reviewed34-recall-job.json',
            pair_weight=.25,replay_weight=.75,
            method='reviewed33_error_span_cpo_with_stronger_balanced_positive_replay',
            hypothesis='At fixed64 updates from the original frozen adapter, lowering error-pair weight from.5 to.25 and raising positive replay weight to.75 with at least3 exposures per40 context-positive will reach the unchanged39/40 preservation and32/40 supplement pass gates while at least14/33 errors become pass and all three direction errors are repaired. Failure of any frozen gate rejects the candidate.',
            rationale='The quarantined prior run actually repaired19 errors but preserved only36/40 and passed22/40 context samples; all64 gradients clipped and adapter relativeL2 change was2.03%. Valid subset probe still shows aligned chosen/preference gradients. Test stronger positive supervision rather than repeat identical exposure and weighting; changed data/compute prevents a matched causal method claim.',
            data_correction=dict(report_hash=fingerprint(lexical),excluded_ids=['v10-daily-19-en']+pool['excluded_nonmajor_ids'],
                corrected_review_hash=pool['review_hash'],old_trial_eligible=False,
                starting_weights='Original frozen Qwen35start; no inheritance of quarantined candidate.',
                positive_age_view_hash=lexical['new_hashes']['supplement'],
                unchanged_original416_hash=pool['train_hash']),
            historical_scope='Original41 and reviewed34 trials are ineligible because reasonable translations entered negatives; clean40 was invalidated before GPU execution. Their preserved actual generation/update failures guide this engineering hypothesis only. No valid Goal optimization round claimed yet.')
        plan['budget'].update(replay_visits=704,forwards=1216,backwards=960,
            minimum_original_positive_visits=1,minimum_supplement_positive_visits=3)
        plan['evidence'].update(lexical_correction_hash=fingerprint(lexical),
            prior_failed_screen_hash=fingerprint(load('runs/qwen35-v3-reviewed34-recall-screening.json')),
            prior_update_diagnostics_hash=fingerprint(load('runs/qwen35-v3-reviewed34-weight-update-diagnostics.json')))
        exposure=Counter(replay_ids)
        assert all(exposure[row['id']]>=3 for row in added)
        assert all(exposure[row['id']]>=1 for row in original)
        plan['budget']['positive_exposure_hash']=fingerprint(dict(exposure))
    write_jsonl(dest/'pairs.jsonl',pairs);write_jsonl(dest/'replay.jsonl',replay)
    write_json(dest/'annotations.json',annotation_report);write_jsonl(dest/'recall.jsonl',recall_rows)
    write_json(dest/'plan.json',plan)
    print(dict(plan_hash=fingerprint(plan),full_tokens=full_tokens,pairs=expected_errors,replay=456,recall=expected_errors+80,
        replay_strata=plan['budget']['replay_strata']),flush=True)


if __name__=='__main__':
    main()
