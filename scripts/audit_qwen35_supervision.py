"""CPU-only signal/coverage audit; teacher-forced gradient work waits for real negatives."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.audit_qwen35_stage import legacy_known, load
from scripts.finalize_v13_optimization import complete_review
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import validated_annotations,encode_critical_spans
from witrans_tools.data import encode_example
from witrans_tools.qwen35_stage_gates import evaluate_stage


def coverage(rows):
    return dict(rows=len(rows),context_rows=sum(bool(r['input'].get('context')) for r in rows),
                glossary_rows=sum(bool(r['input'].get('glossary')) for r in rows),
                strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in rows)))


def main():
    dest=Path('runs/qwen35-v3-supervision-audit.json')
    assert not dest.exists()
    plan=load('data/prepared/qwen35-v2/plan.json')
    train=read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    sft=read_jsonl(plan['sft']['path']);pairs=read_jsonl(plan['cpo']['path'])
    annotations=validated_annotations(pairs,load(plan['cpo']['annotations']))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    encoded={}
    for row in pairs:
        encoded_chosen=encode_example(tokenizer,row,1024)
        entry=annotations.get(row['id'])
        if entry:
            encoded_chosen=encode_critical_spans(tokenizer,row,entry['critical_spans'],1024,entry['multiplier'])
        rejected=encode_example(tokenizer,{**row,'output':row['rejected']},1024)
        chosen_tokens=sum(x!=-100 for x in encoded_chosen['labels'])
        weights=[w for w,label in zip(encoded_chosen.get('token_weights',[1]*len(encoded_chosen['labels'])),encoded_chosen['labels']) if label!=-100]
        encoded[row['id']]=dict(chosen_tokens=chosen_tokens,rejected_tokens=sum(x!=-100 for x in rejected['labels']),
            prompt_tokens=sum(x==-100 for x in encoded_chosen['labels']),
            annotated=entry is not None,span_tokens=sum(w>1 for w in weights),
            span_weight_sum=sum(w for w in weights if w>1),total_weight=sum(weights),
            span_share_of_chosen_nll=sum(w for w in weights if w>1)/sum(weights),
            actual_parameter_gradient='not measured; requires reviewed current-model negatives and serial GPU probe')
    schedule_counts=Counter(plan['cpo']['order'])
    schedule_strata=Counter()
    by_id={r['id']:r for r in pairs}
    for key,n in schedule_counts.items():
        row=by_id[key];schedule_strata[row['category']+'/'+row['input']['target_lang']]+=n
    _,known=complete_review('runs/qwen35-finetuned-known',plan['datasets']['known'])
    _,public=complete_review('runs/qwen35-finetuned-public',plan['datasets']['public'])
    _,v7=legacy_known('runs/v15-error-start-dev',plan['datasets']['known'])
    gates=evaluate_stage(known,public,known,v7)
    result=dict(at=now(),plan_hash=fingerprint(plan),sft=coverage(sft),old_pairs=coverage(pairs),
        audited_discovery_train=coverage(train),old_pair_ids_covered_by_discovery=sum(r['id'] in {t['id'] for t in train} for r in pairs),
        actual_cpo_exposures_by_stratum=dict(schedule_strata),old_pair_encoded_signal=encoded,
        stage_screen_for_frozen_start=gates,
        observations=['Existing85 negatives are historic Qwen3 errors, not verified current Qwen3.5 errors.',
          'Current CPO has7 context rows,3 glossary rows and1 hard/en pair; coverage is asymmetric. No causal failure attribution from counts alone.',
          'Critical weighting affects chosen normalized NLL; preference margin still uses unweighted summed likelihood. Parameter and critical-span gradients not yet measured.',
          'Dual2pp n20 stratum constraints imply known pass>=166 for this frozen pair of baselines, despite headline threshold164. Both17 hard/zh and18 travel/en are required.',
          'Higher likelihood or lower loss alone cannot count as repair. Await real416 TRAIN outputs and semantic readings before new negative creation.'],
        required_next='Audit actual current errors, chosen/rejected margins and measured critical-span parameter gradient before one small frozen repair experiment.',
        stage_goal_complete=False,release_approved=False)
    write_json(dest,result)
    print({k:result[k] for k in ('sft','old_pairs','audited_discovery_train','actual_cpo_exposures_by_stratum','observations')},flush=True)
    print({'stage_gates':gates['gates'],'effective_known_pass_minimum':gates['effective_known_pass_minimum']},flush=True)


if __name__=='__main__':
    main()
