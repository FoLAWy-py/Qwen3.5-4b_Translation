"""Freeze content-bound masked decisions before any identity-key read."""
from collections import Counter
from pathlib import Path

from data.v12_blind_reviews import PACKETS_HASH, REVIEWS
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    if Path('runs/v12-blind-unmask-receipt.json').exists():
        raise ValueError('Identity map already unmasked; preserve frozen decisions')
    packets = read_jsonl('runs/v12-blind-packets.jsonl')
    if fingerprint(packets) != PACKETS_HASH:
        raise ValueError('Frozen masked packets changed; re-review required')
    decisions = []
    for number, packet in enumerate(packets, 1):
        if number not in REVIEWS:
            continue
        grade_a, grade_b, note_a, note_b = REVIEWS[number]
        for label, grade, note in (('A', grade_a, note_a), ('B', grade_b, note_b)):
            if grade not in ('pass', 'minor', 'major', 'critical') or not note:
                raise ValueError('Incomplete source-based decision')
            decisions.append({'packet_id': packet['id'], 'position': number, 'label': label,
                              'grade': grade, 'note': note, 'packet_hash': fingerprint(packet),
                              'input_raw_hash': fingerprint({'input': packet['input'], 'raw': packet['outputs'][label]}),
                              'at': now(), 'reviewer': 'Codex (prior outputs known; identity map withheld)',
                              'method': 'Read source/context/glossary and normalized semantic view; format separate. Same evaluator has prior knowledge; not an independent blind evaluator.'})
    write_jsonl('runs/v12-blind-decisions.jsonl', decisions)
    report = {'at': now(), 'packets_hash': PACKETS_HASH, 'decision_hash': fingerprint(decisions),
              'read_pairs': len(decisions)//2, 'total_pairs': len(packets),
              'complete': len(decisions) == 2*len(packets), 'identity_map_read': False,
              'grades_without_identities': dict(Counter(row['grade'] for row in decisions)),
              'release_approved': False}
    write_json('runs/v12-blind-review-progress.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
