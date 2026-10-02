"""Freeze next training-only seed inputs and references for32B labeling."""
from pathlib import Path
from data.v5_seed_pairs import PAIRS
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

def main():
    directory = Path('data/prepared/v5-seed')
    if directory.exists():
        raise ValueError('Seed batch already frozen')
    old = {r['input']['text'].strip().casefold() for p in Path('data/prepared').glob('**/*.jsonl') for r in read_jsonl(p) if 'input' in r}
    excluded_families = {family for family, _, en, zh, _ in PAIRS if en.strip().casefold() in old or zh.strip().casefold() in old}
    rows = []
    for i, (family, category, en, zh, context) in enumerate(PAIRS, 1):
        if family in excluded_families:
            continue
        for target, text, translation in (('en', zh, en), ('zh-CN', en, zh)):
            if text.strip().casefold() in old:
                raise ValueError(f'Old source repeated: {text}')
            row = {'id': f'v5-seed-{i:03}-{target}', 'group_id': f'v5-seed-{family}', 'category': category,
                'input': {'text': text, 'target_lang': target, 'context': context, 'glossary': {}}, 'output': {'translation': translation},
                'source': {'name': 'Original Codex-authored v5 training seed', 'license': 'Original synthetic project data',
                    'training_allowed': True, 'external_labeling_allowed': True}}
            reviewed(row, 'Bilingual seed individually composed and checked before32B generation')
            validate_record(row, True, True)
            rows.append(row)
    write_jsonl(directory / 'references.jsonl', rows)
    write_jsonl('data/v5_seed_label_inputs.jsonl', [{k: v for k,v in r.items() if k not in ('output','review')} for r in rows])
    write_json(directory / 'manifest.json', {'at': now(), 'rows': len(rows), 'groups': len({r['group_id'] for r in rows}),
        'references_hash': fingerprint(rows), 'excluded_duplicate_families': sorted(excluded_families),
        'scope': 'Next training batch only, no v4 train/dev mutation; same text across contexts stays in one family; whole family excluded if any source duplicates existing corpus'})
    print({'rows': len(rows), 'groups': len({r['group_id'] for r in rows})})

if __name__ == '__main__':
    main()
