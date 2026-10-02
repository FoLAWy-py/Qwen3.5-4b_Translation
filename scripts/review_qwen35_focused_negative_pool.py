"""Append individual source readings of draft32 negatives; no automatic semantic grading."""
import argparse
from collections import Counter
from pathlib import Path

from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--readings',required=True)
    args=parser.parse_args()
    pool=read_jsonl('data/prepared/qwen35-v3-reviewed32-current-errors/pairs.jsonl')
    indexed={r['id']:r for r in pool}
    actual={r['id']:r for r in read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')}
    dest=Path('runs/qwen35-v3-focused32-negative-quality-rereview.jsonl')
    old=read_jsonl(dest) if dest.exists() else []
    seen={r['id'] for r in old}
    assert len(seen)==len(old)
    for note in old:
        assert note['pair_hash']==fingerprint(indexed[note['id']])
        assert note['actual_generation_hash']==fingerprint(actual[note['id']])
    incoming=[]
    for line in Path(args.readings).read_text(encoding='utf-8-sig').splitlines():
        rid,verdict,note=line.split('\t',2)
        assert rid in indexed and rid not in seen and verdict in ('major','minor','pass') and note.strip()
        seen.add(rid)
        pair=indexed[rid];raw=actual[rid]
        assert pair['input']==raw['input'] and pair['rejected']==raw['prediction']
        incoming.append(dict(id=rid,at=now(),reviewer='Codex AI; not independent human review',
            pair_hash=fingerprint(pair),actual_generation_hash=fingerprint(raw),verdict=verdict,
            valid_major_negative=verdict=='major',note=note,
            scope='Conservative individual reread of actual original-start TRAIN pair; do not freeze a next trial until all32 are reviewed.'))
    for note in incoming:
        append_jsonl(dest,note)
    print(dict(reviewed=len(old)+len(incoming),expected=32,
        counts=dict(Counter(r['verdict'] for r in old+incoming))),flush=True)


if __name__=='__main__':
    main()
