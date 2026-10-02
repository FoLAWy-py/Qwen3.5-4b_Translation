"""Display differences only after the full source/context/output reading."""
from pathlib import Path
from witrans_tools.common import read_jsonl

for label in ('known','public'):
    old={r['id']:r for r in read_jsonl(f'runs/qwen35-finetuned-{label}.jsonl')}
    reviews={r['id']:r for r in read_jsonl(f'runs/qwen35-finetuned-{label}-semantic.jsonl')}
    for row in read_jsonl(f'runs/takeover-20261002/development-final/{label}.jsonl'):
        prior=old[row['id']]
        if prior.get('raw')!=row['raw']:
            review=reviews[row['id']]
            print(label,row['id'],'OLD',prior.get('raw'),'NEW',row['raw'],'NOTE',review['verdict'],review['note'])
