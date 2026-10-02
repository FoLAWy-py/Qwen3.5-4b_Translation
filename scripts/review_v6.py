"""Reuse exact inspected outputs or explicit manual decisions; never grade unread outputs."""
import argparse
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.review_reuse import indexed, reuse_identical_reviews

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('cpo','sft_control','critical_cpo','sft','baseline'), required=True)
    parser.add_argument('--plan', default='runs/v6-evaluation-plan.json')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    refs = indexed(read_jsonl(plan['input']))
    current = read_jsonl(plan['outputs'][args.role])
    indexed(current)
    pools = []
    prior_stems = (['v7-critical','v6-cpo','v6-sft-control'] if args.role in ('sft','baseline') else
        (['v6-cpo','v6-sft-control'] if args.role == 'critical_cpo' else
        (['v6-cpo'] if args.role == 'sft_control' else []))) + ['v5-selected','v5-start','v4-selected','v4-start']
    prior_stems += plan.get('additional_review_cache_stems',[])
    for stem in prior_stems:
        pools.append((stem, indexed(read_jsonl(f'runs/{stem}-dev.jsonl')),
            indexed(read_jsonl(f'runs/{stem}-dev-semantic.jsonl'))))
    output_stem = str(Path(plan['outputs'][args.role]).with_suffix(''))
    manual_path = Path(output_stem.removesuffix('-dev')+'-manual-semantic.jsonl')
    manual = indexed(read_jsonl(manual_path)) if manual_path.exists() else {}
    if set(manual) - {r['id'] for r in current}:
        raise ValueError('Manual decision without current generation')
    decisions, pending = [], []
    for row in current:
        ref = refs[row['id']]
        if row['input'] != ref['input'] or row['reference'] != ref['output']:
            raise ValueError('Frozen reference changed')
        if row['id'] in manual:
            decision = manual[row['id']]
            if (decision.get('reviewer') != 'Codex' or decision.get('verdict') not in ('pass','minor','major','critical')
                    or decision.get('output_hash') != fingerprint({k:row[k] for k in ('input','raw','reference')})
                    or decision.get('format_valid') != ('prediction' in row) or decision.get('ended') != row.get('ended')
                    or decision.get('category') != ref['category'] or decision.get('target_lang') != ref['input']['target_lang']
                    or decision.get('group_id') != ref['group_id'] or not decision.get('note')
                    or type(decision.get('language_correct')) is not bool):
                raise ValueError('Manual evidence missing or stale')
            decisions.append({**decision, 'review_source':'new_individual_reading'})
            continue
        for stem, raw, reviewed in pools:
            try:
                decision = reuse_identical_reviews([raw[row['id']]], [reviewed[row['id']]], [row])[0]
            except (ValueError, KeyError):
                continue
            decisions.append({**decision, 'review_source':stem})
            break
        else:
            pending.append(row)
    summary = {'at':now(), 'generated_rows':len(current), 'reviewed':len(decisions), 'expected_total':len(refs),
        'complete':len(decisions)==len(refs), 'pending_new_reading':len(pending),
        'verdicts':dict(Counter(d['verdict'] for d in decisions)),
        'review_sources':dict(Counter(d['review_source'] for d in decisions)),
        'scope':'Exact source/raw/reference/structural equality only; changed outputs stay pending unless explicit content-bound Codex reading exists. No release approval.'}
    write_jsonl(output_stem+'-semantic.jsonl', decisions)
    write_jsonl(output_stem+'-unmatched.jsonl', pending)
    write_json(output_stem+'-semantic.summary.json', summary)
    print(summary)

if __name__ == '__main__':
    main()
