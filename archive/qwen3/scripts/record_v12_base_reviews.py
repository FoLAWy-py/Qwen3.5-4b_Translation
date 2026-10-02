"""Bind explicit166-row reading to frozen raw outputs; never parse-repair outputs."""
import hashlib
from pathlib import Path

from data.v12_base_reviews import GENERATION_SHA256, NOTES, OVERRIDES, READ_KEYS, WRONG_LANGUAGE
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl


def main():
    path = Path('runs/v12-base-dev.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != GENERATION_SHA256:
        raise ValueError('Previously read generations changed')
    refs = {row['id']: row for row in read_jsonl('data/prepared/v4/dev.jsonl')}
    decisions = []
    for row in read_jsonl(path):
        _, _, number, lang = row['id'].split('-', 3)
        key = (int(number), lang)
        if key not in READ_KEYS:
            continue
        ref = refs[row['id']]
        verdict, note = OVERRIDES.get(key, ('pass', NOTES[key[0]]))
        decisions.append({'id': row['id'], 'group_id': ref['group_id'], 'category': ref['category'],
                          'target_lang': ref['input']['target_lang'], 'verdict': verdict, 'note': note,
                          'reviewer': 'Codex', 'at': now(), 'format_valid': 'prediction' in row,
                          'ended': row.get('ended'), 'language_correct': key not in WRONG_LANGUAGE,
                          'output_hash': fingerprint({k: row[k] for k in ('input','raw','reference')}),
                          'purpose': 'Known-development screening, not release',
                          'review_limit': 'Evaluator knew baseline identity; not an independent blinded review.'})
    if len(decisions) != 166 or len({row['id'] for row in decisions}) != 166:
        raise ValueError('Explicit reading set differs from166 inspected rows')
    write_jsonl('runs/v12-base-manual-semantic.jsonl', decisions)
    print({'recorded_explicit_readings': len(decisions), 'generation_sha256': GENERATION_SHA256})


if __name__ == '__main__':
    main()
