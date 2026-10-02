"""Freeze original bilingual multi-sentence training references before API labels."""
from collections import Counter
from pathlib import Path
from data.v8_multisentence_pairs import PAIRS
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

def main():
    destination = Path('data/prepared/v8-multisentence')
    if destination.exists():
        raise ValueError('Multi-sentence inputs already frozen')
    prior = {r['input']['text'].strip().casefold() for p in Path('data/prepared').glob('**/*.jsonl') for r in read_jsonl(p) if 'input' in r}
    rows = []
    for i, (category, en, zh, context, glossary) in enumerate(PAIRS, 1):
        if en.strip().casefold() in prior or zh.strip().casefold() in prior:
            raise ValueError(f'New source exact-duplicates prior source at{i}')
        for target, text, answer, terms in (('zh-CN', en, zh, glossary), ('en', zh, en, {v:k for k,v in glossary.items()})):
            row = {'id': f'v8-multi-{i:03}-{target}', 'group_id': f'v8-multi-{i:03}', 'category': category,
                'input': {'text': text, 'target_lang': target, 'context': context, 'glossary': terms}, 'output': {'translation': answer},
                'source': {'name': 'Original Codex-authored multi-sentence training paragraphs', 'license': 'Original synthetic project data',
                    'training_allowed': True, 'external_labeling_allowed': True}}
            reviewed(row, 'Individually authored and checked two-or-more sentence references before teacher output; condition, negation, objects and glossary preserved; not copied from dev/test sources')
            validate_record(row, True, True)
            rows.append(row)
    write_jsonl(destination / 'references.jsonl', rows)
    write_jsonl('data/v8_multisentence_label_inputs.jsonl', [{k:v for k,v in r.items() if k not in ('output', 'review')} for r in rows])
    report = {'at': now(), 'rows': len(rows), 'groups': len(PAIRS), 'hash': fingerprint(rows),
        'categories': dict(Counter(r['category'] for r in rows)), 'glossary_rows': sum(bool(r['input']['glossary']) for r in rows),
        'multi_sentence_rows': len(rows), 'policy': 'Training-only;80 source texts individually authored as multi-sentence paragraphs; hard cases are idioms, not claimed context-only ambiguity; exact prior-source exclusion only; not release evidence'}
    write_json(destination / 'manifest.json', report)
    print(report)

if __name__ == '__main__':
    main()
