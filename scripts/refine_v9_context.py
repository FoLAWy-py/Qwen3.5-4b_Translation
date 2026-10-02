"""Preserve first review, then exclude an ambiguous bilingual source family."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    root = Path('data/prepared/v9-constraints-context-checked')
    if root.exists():
        raise ValueError('Preserve revised approved corpus')
    rows = read_jsonl('data/prepared/v9-constraints/train.jsonl')
    if fingerprint(rows) != '25f02c8bca09eae9055632798804568f220070c29210c6163a9b5711f71196b0':
        raise ValueError('First review changed')
    excluded = [r for r in rows if r['group_id']=='v9-constraint-003']
    kept = [r for r in rows if r['group_id']!='v9-constraint-003']
    if len(excluded)!=2 or len(kept)!=58:
        raise ValueError('Unexpected ambiguity exclusion scope')
    for row in kept:
        validate_record(row, True, True)
    report = {'at':now(), 'supersedes':'data/prepared/v9-constraints/train.jsonl',
        'prior_hash':fingerprint(rows), 'train_hash':fingerprint(kept), 'rows':len(kept), 'groups':29,
        'excluded_ids':[r['id'] for r in excluded],
        'reason':'Codex source recheck: English visit does not explicitly specify a home visit, whereas the Chinese source says上门. Whole bilingual family excluded rather than adding unsupported location or changing frozen source context.',
        'reviewer':'Codex (user-authorized AI acceptance)',
        'actions':{'accept_teacher':37,'correct':21,'exclude':2},
        'policy':'Use only this58-row revision for future training; original60-row first review preserved for audit. No release or model accuracy claim.'}
    write_jsonl(root / 'train.jsonl',kept)
    write_json(root / 'manifest.json',report)
    write_json('runs/v9-constraints-context-review.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
