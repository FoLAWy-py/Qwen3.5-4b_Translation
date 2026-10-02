"""Export only explicit read decisions whose whole snapshot still matches."""
import argparse
from data.v13_candidate_readings import BATCHES
from scripts.export_v13_first_readings import save
from witrans_tools.common import fingerprint, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--series',default='v13',choices=('v13','v14'))
    args = parser.parse_args()
    batches = BATCHES
    if args.series=='v14':
        from data.v14_candidate_readings import BATCHES as fact_batches
        batches = fact_batches
    batch = batches[args.batch]
    rows = read_jsonl(batch['stem']+'-unmatched.jsonl')
    if fingerprint(rows) != batch['snapshot_hash']:
        raise ValueError('Read snapshot changed')
    annotations = {}
    for line in batch['readings'].strip().splitlines():
        row_id, verdict, note = line.split('|', 2)
        if row_id in annotations or verdict not in ('pass','minor','major','critical') or not note:
            raise ValueError('Invalid explicit reading')
        annotations[row_id] = (verdict, note)
    if set(annotations) != {row['id'] for row in rows}:
        raise ValueError('Read IDs do not match snapshot')
    save(batch['stem']+'-manual.jsonl', [
        {'id': row['id'], 'reviewer': 'Codex', 'generation_hash': fingerprint(row),
         'verdict': annotations[row['id']][0], 'note': annotations[row['id']][1], 'language_correct': True}
        for row in rows
    ])
    print({'batch': args.batch, 'new_individual_readings': len(rows)})


if __name__ == '__main__':
    main()
