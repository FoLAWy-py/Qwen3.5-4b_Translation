"""Bind explicitly read outputs to fixed generation snapshots; no inferred grades."""
from pathlib import Path

from data.v13_public_v7_readings import READINGS
from witrans_tools.common import fingerprint, read_jsonl, write_jsonl


def save(path, decisions):
    existing = read_jsonl(path) if Path(path).exists() else []
    by_id = {row['id']: row for row in existing}
    for decision in decisions:
        if decision['id'] in by_id and by_id[decision['id']] != decision:
            raise ValueError('Existing reading differs; preserve it for explicit reconsideration')
        by_id[decision['id']] = decision
    write_jsonl(path, list(by_id.values()))


def main():
    rows = read_jsonl('runs/v13-v7-public.jsonl')
    if fingerprint(rows) != '1c4e5e76dc8a0cf8d9ef5abb811819deaf6eb971fe6ba39c197f8963bc68ee81':
        raise ValueError('Read public output snapshot changed')
    by_id = {row['id']: row for row in rows}
    decisions = []
    for line in READINGS.strip().splitlines():
        number, direction, verdict, note = line.split('|', 3)
        row_id = f'public-short-{number}-{direction}'
        row = by_id[row_id]
        decisions.append({'id': row_id, 'reviewer': 'Codex', 'generation_hash': fingerprint(row),
                          'verdict': verdict, 'note': note, 'language_correct': True})
    if len(decisions) != len(rows) or len({d['id'] for d in decisions}) != len(rows):
        raise ValueError('Readings must cover every output exactly once')
    save('runs/v13-v7-public-manual.jsonl', decisions)
    # These two records were read from the unmatched snapshot before export.
    snapshot = read_jsonl('runs/v13-candidate-known-unmatched.jsonl')
    if fingerprint(snapshot) != 'f61112aaa1f0209623f0ab05983ea11100445e6065e1cf0e5ae25b7a1fa78ad5':
        raise ValueError('Read candidate snapshot changed')
    explicit = {
        'v4-dev-001-en': ('major', '钥匙粘在花盆底下误译为stuck in the pot，位置由底下变成盆内。'),
        'v4-dev-003-zh-CN': ('pass', '已归还借来的梯子而仍持有对方延长线，动作和转折完整。'),
    }
    save('runs/v13-candidate-known-manual.jsonl', [
        {'id': row['id'], 'reviewer': 'Codex', 'generation_hash': fingerprint(row),
         'verdict': explicit[row['id']][0], 'note': explicit[row['id']][1], 'language_correct': True}
        for row in snapshot
    ])
    print({'public_read': len(decisions), 'candidate_new_read': len(snapshot)})


if __name__ == '__main__':
    main()
