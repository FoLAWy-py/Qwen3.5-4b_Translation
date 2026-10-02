"""Reuse only identical already-read outputs; leave changed rows unreviewed."""
from collections import Counter
from witrans_tools.common import now, read_jsonl, write_json, write_jsonl
from witrans_tools.review_reuse import reuse_identical_reviews

def main():
    current = read_jsonl('runs/v5-start-dev.jsonl')
    refs = {r['id']:r for r in read_jsonl('data/prepared/v5-mixed/dev.jsonl')}
    pools = []
    for name, stem in (('v5_candidate_individually_read','v5-selected'), ('v4_start_individually_read','v4-start')):
        pools.append((name, {r['id']:r for r in read_jsonl(f'runs/{stem}-dev.jsonl')},
            {r['id']:r for r in read_jsonl(f'runs/{stem}-dev-semantic.jsonl')}))
    decisions, pending = [], []
    for row in current:
        ref = refs[row['id']]
        if row['input'] != ref['input'] or row['reference'] != ref['output']:
            raise ValueError('Frozen reference mismatch')
        for source, raw, reviewed in pools:
            try:
                cached = reuse_identical_reviews([raw[row['id']]], [reviewed[row['id']]], [row])[0]
            except (ValueError, KeyError):
                continue
            cached['review_source'] = source
            decisions.append(cached)
            break
        else:
            pending.append(row)
    if len({r['id'] for r in current}) != len(current):
        raise ValueError('Duplicate current outputs')
    historical = pools[1][2]
    changes = [{'id':d['id'], 'old_verdict':historical[d['id']]['verdict'], 'current_verdict':d['verdict'], 'note':d['note']}
        for d in decisions if d['output_hash'] == historical[d['id']]['output_hash'] and d['verdict'] != historical[d['id']]['verdict']]
    summary = {'at':now(), 'generated_rows':len(current), 'reviewed':len(decisions), 'expected_total':len(refs),
        'complete':len(decisions)==len(refs), 'unmatched_current_rows':len(pending),
        'verdicts':dict(Counter(r['verdict'] for r in decisions)), 'review_sources':dict(Counter(r['review_source'] for r in decisions)),
        'current_same_output_judgment_changes':changes,
        'scope':'Existing Codex reviews reused only after exact input/raw/reference/end/parse equality; changed rows remain unreviewed. No claim that all current outputs were newly read.'}
    write_jsonl('runs/v5-start-dev-semantic.jsonl', decisions)
    write_jsonl('runs/v5-start-dev-unmatched.jsonl', pending)
    write_json('runs/v5-start-dev-semantic.summary.json', summary)
    print(summary)

if __name__ == '__main__':
    main()
