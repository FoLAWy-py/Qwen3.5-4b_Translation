"""Bind manually written per-family source and label readings to selected records."""
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    root = Path('data/prepared/v16-family-audit')
    plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    rows = read_jsonl(root/'records.jsonl')
    families = read_jsonl(root/'families.jsonl')
    assert fingerprint(rows) == plan['record_hash'] and fingerprint(families) == plan['family_hash']
    indexed = {r['group_id']: r for r in families}
    path = Path('runs/v16-family-source-audit.jsonl')
    prior = read_jsonl(path) if path.exists() else []
    decisions = {r['group_id']: r for r in prior}
    assert len(decisions) == len(prior)
    for line in Path('data/v16-family-readings.tsv').read_text(encoding='utf-8').splitlines():
        group, status, note = line.split('|', 2)
        assert status in ('approved', 'exclude', 'needs_repair') and note
        family = indexed[group]
        records = [r for r in rows if r['group_id'] == group]
        assert fingerprint(records) == family['records_hash']
        for record in records:
            validate_record(record, True, True, purpose='training')
        decision = dict(group_id=group, reviewer='Codex', status=status, note=note,
                        records_hash=fingerprint(records), record_ids=[r['id'] for r in records],
                        source_licenses=[r['source']['license'] for r in records],
                        source_review_hashes=[r['review']['content_hash'] for r in records],
                        method='Explicit source-grounded reading of every displayed direction and reference; no NLL-derived error label.',
                        new_negative_authorized=False)
        assert group not in decisions or decisions[group] == decision, 'Preserve prior audit'
        decisions[group] = decision
    write_jsonl(path, list(decisions.values()))
    report = dict(at=now(), plan_hash=fingerprint(plan), reviewed_families=len(decisions), total_families=200,
                  reviewed_records=sum(len(d['record_ids']) for d in decisions.values()),
                  verdicts=dict(Counter(d['status'] for d in decisions.values())),
                  pending_families=[f['group_id'] for f in families if f['group_id'] not in decisions],
                  decision_hash=fingerprint(list(decisions.values())), release_approved=False)
    write_json('runs/v16-family-source-audit.summary.json', report)
    print({k: report[k] for k in ('reviewed_families', 'reviewed_records', 'total_families', 'verdicts')})


if __name__ == '__main__':
    main()
