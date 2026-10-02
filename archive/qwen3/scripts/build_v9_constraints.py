"""Freeze new training-only paragraphs before DeepInfra candidate generation."""
from collections import Counter
from pathlib import Path
from data.v9_constraint_pairs import PAIRS
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    destination = Path('data/prepared/v9-constraints')
    if destination.exists():
        raise ValueError('Preserve frozen training sources')
    prior = {r['input']['text'].strip().casefold() for p in Path('data/prepared').glob('**/*.jsonl')
        for r in read_jsonl(p) if 'input' in r}
    rows = []
    for index, (category,en,zh,context,terms) in enumerate(PAIRS,1):
        for target,text,answer,glossary in (('zh-CN',en,zh,terms), ('en',zh,en,{v:k for k,v in terms.items()})):
            if text.strip().casefold() in prior:
                raise ValueError(f'Exact prior-source duplicate at{index}/{target}')
            row = {'id':f'v9-constraint-{index:03}-{target}', 'group_id':f'v9-constraint-{index:03}',
                'category':category, 'input':{'text':text,'target_lang':target,'context':context,'glossary':glossary},
                'output':{'translation':answer},
                'source':{'name':'Original Codex-authored constraint paragraphs',
                    'license':'Original synthetic project data', 'training_allowed':True,
                    'external_labeling_allowed':True, 'construction':'One individually authored bilingual paragraph family; both directions grouped together; training only'}}
            reviewed(row, 'Codex individually authored and read source-grounded paragraph reference before teacher generation; preserves speaker, condition, quantifier, negation, digits and glossary')
            validate_record(row, True, True)
            rows.append(row)
    if len(rows) != 60 or len({r['input']['text'].strip().casefold() for r in rows}) != len(rows):
        raise ValueError('Unexpected or duplicate source count')
    write_jsonl(destination / 'references.jsonl', rows)
    write_jsonl('data/v9_constraint_label_inputs.jsonl', [{k:v for k,v in r.items() if k not in ('output','review')} for r in rows])
    report = {'at':now(), 'rows':len(rows), 'groups':len(PAIRS), 'reference_hash':fingerprint(rows),
        'categories':dict(Counter(r['category'] for r in rows)),
        'context_rows':sum(bool(r['input']['context']) for r in rows),
        'glossary_rows':sum(bool(r['input']['glossary']) for r in rows), 'multi_sentence_rows':len(rows),
        'policy':'Training-only original sources, not release tests. Codex-author acceptance, not independent professional human review. Prior exact-source exclusion checked; no full semantic near-duplicate guarantee. Full teacher candidates require later individual review.'}
    write_json(destination / 'manifest.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
