"""Content-bound summary of the completed, no-update Qwen3.5 gradient probe."""
import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def distribution(values):
    values=list(values)
    assert values and all(math.isfinite(v) for v in values)
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values))


def summarize(rows):
    metrics=[row['metrics'] for row in rows]
    ratios=[]
    for m in metrics:
        chosen=m['weighted_chosen_NLL']['gradient_norm']
        preference=m['preference_only']['gradient_norm']
        if chosen:
            ratios.append(preference/chosen)
    fields=('summed_margin','mean_margin','span_rest_cosine','chosen_preference_cosine',
            'span_fraction_of_component_norms','span_rest_additivity_relative_error')
    result=dict(rows=len(rows), distributions={key:distribution(m[key] for m in metrics if m[key] is not None)
                    for key in fields if any(m[key] is not None for m in metrics)},
                negative_summed_margin=sum(m['summed_margin']<0 for m in metrics),
                negative_mean_margin=sum(m['mean_margin']<0 for m in metrics),
                opposite_summed_and_mean_margin=sum(m['summed_margin']*m['mean_margin']<0 for m in metrics),
                chosen_preference_conflict=sum(m['chosen_preference_cosine'] is not None
                    and m['chosen_preference_cosine']<0 for m in metrics),
                zero_chosen_gradients=sum(m['weighted_chosen_NLL']['gradient_norm']==0 for m in metrics))
    if ratios:
        result['preference_to_weighted_chosen_gradient_norm_ratio']=distribution(ratios)
    result['gradient_norms']={component:distribution(m[component]['gradient_norm'] for m in metrics)
        for component in ('weighted_span_NLL_contribution','remaining_NLL_contribution',
                          'weighted_chosen_NLL','preference_only')}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',required=True)
    parser.add_argument('--draft',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists(), 'Preserve all reports'
    state=json.loads(Path(args.probe).read_text(encoding='utf-8'))
    draft=json.loads(Path(args.draft).read_text(encoding='utf-8'))
    rows=read_jsonl(Path(args.probe).with_suffix('.jsonl'))
    protocol=state['protocol']
    assert state['status']=='finished' and state['gpu_executed'] and state['optimizer_updates']==0
    assert fingerprint(protocol)==state['protocol_hash'] and fingerprint(draft)==protocol['draft_hash']
    assert protocol.get('vector_reduction_dtype','').startswith('CPU float64'), 'Earlier FP32 vector reductions produced impossible cosine; do not use for trial selection'
    assert state['cpu_parameter_count']==0 and state['peak_reserved_gib']<=protocol['memory_cap_gib']
    assert state['forward_calls']==protocol['expected_forward_calls']==5*len(rows)
    assert state['backward_calls']==protocol['expected_backward_calls']==4*len(rows)
    assert state['encoded_forward_tokens']==protocol['encoded_forward_tokens']
    entries={entry['id']:entry for entry in draft['entries']}
    assert len(rows)==len(entries)==len({row['id'] for row in rows})
    assert [row['id'] for row in rows]==protocol['ids']==list(entries)
    strata=defaultdict(list)
    for i,row in enumerate(rows,1):
        entry=entries[row['id']]
        assert row['draft_entry_hash']==fingerprint(entry) and row['protocol_hash']==state['protocol_hash']
        assert row['forward_calls']==5*i and row['backward_calls']==4*i
        assert row['peak_reserved_gib']<=protocol['memory_cap_gib']
        for component in protocol['components']:
            m=row['metrics'][component]
            assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) and m['gradient_norm']>=0
            assert 0<=m['active_tensors']<=state['trainable_tensors']
        for key in ('span_rest_cosine','chosen_preference_cosine'):
            assert row['metrics'][key] is None or -1.000000000001<=row['metrics'][key]<=1.000000000001
        strata[entry['category']+'/'+entry['target_lang']].append(row)
    assert rows[-1]['encoded_forward_tokens']==state['encoded_forward_tokens']
    report=dict(at=now(), probe_hash=fingerprint(state), raw_hash=fingerprint(rows), draft_hash=fingerprint(draft),
        protocol_hash=state['protocol_hash'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        overall=summarize(rows), strata={key:summarize(value) for key,value in strata.items()},
        most_conflicting_ids=[row['id'] for row in sorted(rows,
            key=lambda r:1 if r['metrics']['chosen_preference_cosine'] is None else r['metrics']['chosen_preference_cosine'])[:5]],
        accounting=dict(forward_calls=state['forward_calls'],backward_calls=state['backward_calls'],
            encoded_forward_tokens=state['encoded_forward_tokens'],optimizer_updates=0,
            cpu_parameter_count=0,peak_reserved_gib=state['peak_reserved_gib']),
        interpretation='Actual single-example parameter gradients at the frozen starting adapter, checkpoint-enabled train mode with dropout disabled. A negative cosine indicates local component opposition. Norm fractions are component magnitudes, not causal shares of learning. Summed-vs-mean margin disagreement flags length sensitivity, not reference correctness.',
        limitations='No optimizer step, aggregate batch gradient, Adam update, TRAIN repair, or heldout efficacy measured. Do not choose a method from loss or gradient size alone. No hard/en major errors existed in the reviewed pool; no corresponding negative gradients invented.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(dest,report)
    print(dict(output=str(dest),rows=len(rows),overall=report['overall']),flush=True)


if __name__=='__main__':
    main()
