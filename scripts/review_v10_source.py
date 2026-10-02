"""Persist only explicit source-bound decisions; retain unread candidates as pending."""
from collections import Counter
from pathlib import Path
from data.v10_reviews import DECISIONS
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    root = Path('data/prepared/v10-source')
    rows = read_jsonl(root / 'candidates.jsonl')
    expected = '576ee553b5b23e99c26e8d6d7eb7c9022249e781c4624856dbb4ca86f5e08c22'
    if fingerprint(rows)!=expected or len(rows)!=200 or len({r['id'] for r in rows})!=200:
        raise ValueError('Original candidate pool changed')
    keys = {r['id'].removeprefix('v10-') for r in rows}
    if set(DECISIONS)-keys:
        raise ValueError('Decision ID absent from frozen sources')
    approved,decisions,pending = [],[],[]
    for row in rows:
        key = row['id'].removeprefix('v10-')
        if key not in DECISIONS:
            pending.append(row)
            continue
        action,replacement,note = DECISIONS[key]
        if action not in ('accept','correct','exclude') or not note or (action=='correct' and not replacement):
            raise ValueError('Explicit individual decision required')
        raw_hash = fingerprint(row)
        decision = {'id':row['id'],'group_id':row['group_id'],'action':action,'note':note,
            'raw_record_hash':raw_hash,'reviewer':'Codex','at':now()}
        if action!='exclude':
            if replacement is not None:
                row['output'] = {'translation':replacement}
            reviewed(row,'Codex individually read actual source and teacher translation; each direction judged against its own immutable source, source-grounded repairs recorded')
            validate_record(row,True,True)
            decision['reviewed_hash'] = fingerprint({'input':row['input'],'output':row['output']})
            approved.append(row)
        decisions.append(decision)
    excluded = {r['group_id'] for r in decisions if r['action']=='exclude'}
    if excluded & {r['group_id'] for r in approved}:
        raise ValueError('Defective source family must be excluded in both directions')
    complete = len(decisions)==len(rows)
    write_jsonl(root / 'reviewed-partial.jsonl',approved)
    write_jsonl(root / 'pending-review.jsonl',pending)
    write_jsonl('data/generated/v10_source_decisions.jsonl',decisions)
    report = {'at':now(),'candidate_hash':expected,'reviewed_candidates':len(decisions),
        'accepted_rows':len(approved),'pending_rows':len(pending),'actions':dict(Counter(r['action'] for r in decisions)),
        'complete_individual_reading':complete,'source_family_and_split_audit_complete':False,
        'training_ready':False,'release_approved':False,
        'scope':'Partial individual source/translation reading only. No unread fallback acceptance; no train.jsonl produced. Full family near-duplicate/held-out audit still required after all200 reads.'}
    write_json('runs/v10-source-review-progress.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
