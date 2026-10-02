"""Bind explicit readings to immutable public candidates; never approve unlisted rows."""
from collections import Counter
from pathlib import Path

from data.public_short_reviews import CANDIDATE_HASH, REVIEWS
from archive.qwen3.scripts.prepare_public_short_pilot import DEST
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    rows = read_jsonl(DEST / 'candidates.jsonl')
    if fingerprint(rows) != CANDIDATE_HASH:
        raise ValueError('Previously read public candidates changed; explicit re-review required')
    evidence, reviewed = [], []
    for row in rows:
        number = int(row['id'].rsplit('-', 1)[1])
        if number not in REVIEWS:
            continue
        category, verdict, corrected, note = REVIEWS[number]
        if verdict not in ('accepted', 'revised', 'rejected') or not note:
            raise ValueError('Incomplete explicit decision')
        if (verdict == 'revised') != (corrected is not None):
            raise ValueError('Correction/verdict mismatch')
        decision = {'id': row['id'], 'candidate_hash': fingerprint(row), 'reviewer': 'Codex',
                    'at': now(), 'verdict': verdict, 'category': category, 'note': note,
                    'corrected_zh': corrected, 'original_en': row['en'], 'original_zh': row['zh']}
        evidence.append(decision)
        if verdict != 'rejected':
            reviewed.append({**row, 'zh': corrected or row['zh'], 'category': category,
                             'status': 'individually_reviewed', 'training_ready': False,
                             'review': decision,
                             'source': {**row['source'], 'modifications': note if corrected else 'none'}})
    write_jsonl(DEST / 'decisions.jsonl', evidence)
    write_jsonl(DEST / 'reviewed-partial.jsonl', reviewed)
    report = {'at': now(), 'candidate_hash': fingerprint(rows), 'total_pending_candidates': len(rows),
              'individually_read': len(evidence), 'remaining_unread': len(rows) - len(evidence),
              'verdicts': dict(Counter(row['verdict'] for row in evidence)),
              'retained_groups': len(reviewed), 'categories': dict(Counter(row['category'] for row in reviewed)),
              'training_ready': False, 'release_approved': False,
              'scope': 'Explicit Codex reading only; further family audit, tokenizer budget validation and frozen split required. Both directions remain in the same family.'}
    write_json(DEST / 'review-progress.json', report)
    print(report)


if __name__ == '__main__':
    main()
