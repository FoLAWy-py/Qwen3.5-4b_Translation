"""Freeze the explicitly read 600-candidate pilot with conservative family isolation."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from data.public_short_reviews import CANDIDATE_HASH, REVIEWS
from scripts.prepare_public_short_pilot import DEST
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record

OUTPUT = Path('data/prepared/v12-public-short')
APPROVED_LINKS = {('public-short-0148', 'public-short-0005'),
                  ('public-short-0457', 'public-short-0052'),
                  ('public-short-0478', 'public-short-0230')}
MANUAL_LINKS = [('public-short-0382', 'public-short-0421')]


def checked_rows():
    candidates = read_jsonl(DEST / 'candidates.jsonl')
    if fingerprint(candidates) != CANDIDATE_HASH:
        raise ValueError('Frozen public candidates changed')
    original = {row['id']: row for row in candidates}
    rows = read_jsonl(DEST / 'reviewed-partial.jsonl')
    for row in rows:
        source = original[row['id']]
        category, verdict, correction, note = REVIEWS[int(row['id'].rsplit('-', 1)[1])]
        if (verdict == 'rejected' or row['en'] != source['en']
                or row['zh'] != (correction or source['zh']) or row['category'] != category
                or row['review']['candidate_hash'] != fingerprint(source)
                or row['review']['note'] != note):
            raise ValueError('Public reviewed content changed without explicit acceptance')
    audit = json.loads((DEST / 'family-audit.json').read_text(encoding='utf-8'))
    if fingerprint(rows) != audit['reviewed_hash']:
        raise ValueError('Family audit no longer binds reviewed content')
    if {(link['a'], link['b']) for link in audit['family_links']} != APPROVED_LINKS:
        raise ValueError('New family links require individual review')
    if set(audit['exclusion_triggers']) != {'public-short-0350'}:
        raise ValueError('Changed exclusions require source review')
    return rows, audit


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve frozen public split; use a new version')
    rows, audit = checked_rows()
    assignments = read_jsonl(DEST / 'family-assignments.jsonl')
    if {row['id'] for row in assignments} != {row['id'] for row in rows}:
        raise ValueError('Incomplete family assignment')
    # Validate the derived assignments against the audited graph rather than trust a file edit.
    parent = {row['id']: row['id'] for row in rows}
    def find(key):
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    for a, b in sorted(APPROVED_LINKS) + MANUAL_LINKS:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    excluded = {find(key) for key in audit['exclusion_triggers']}
    groups = defaultdict(list)
    for row in rows:
        if find(row['id']) not in excluded:
            groups[find(row['id'])].append(row)
    if len(groups) < 500:
        raise ValueError('At least 500 reviewed source families required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B', local_files_only=True)
    family_keys = sorted(groups)
    random.Random(20261001).shuffle(family_keys)
    dev_keys = set(family_keys[:round(len(family_keys)*.10)])
    datasets = {'train': [], 'dev': []}
    lengths, supervised = [], []
    for family in family_keys:
        # One pair per family prevents template variants receiving extra sampling weight.
        representative = min(groups[family], key=lambda row: row['id'])
        split = 'dev' if family in dev_keys else 'train'
        for target, text, translation in (('zh-CN', representative['en'], representative['zh']),
                                          ('en', representative['zh'], representative['en'])):
            inp = {'text': text, 'target_lang': target, 'context': '', 'glossary': {}}
            output = {'translation': translation}
            record = {'id': representative['id'] + '-' + target, 'group_id': 'public-v12-' + family,
                      'category': representative['category'], 'input': inp, 'output': output,
                      'source': {**representative['source'], 'archive_family': representative['group_id'],
                                 'training_allowed': split == 'train',
                                 'construction': 'Explicitly reviewed attributed pair; Chinese may be revised. Both directions share a conservative family.'},
                      'review': {'status': 'approved', 'reviewer': 'Codex (user-authorized AI acceptance)',
                                 'method': 'Individual bilingual source review plus conservative family exclusion',
                                 'at': now(), 'content_hash': fingerprint({'input': inp, 'output': output}),
                                 'notes': representative['review']['note']},
                      'public_provenance': {'candidate_id': representative['id'],
                                            'candidate_hash': representative['review']['candidate_hash'],
                                            'original_en': representative['review']['original_en'],
                                            'original_zh': representative['review']['original_zh'],
                                            'linked_candidate_ids': sorted(row['id'] for row in groups[family])}}
            validate_record(record, True, True, purpose='training' if split == 'train' else 'evaluation')
            encoded = encode_example(tokenizer, record, 1024)
            lengths.append(len(encoded['input_ids']))
            supervised.append(sum(label != -100 for label in encoded['labels']))
            datasets[split].append(record)
    train_groups = {row['group_id'] for row in datasets['train']}
    dev_groups = {row['group_id'] for row in datasets['dev']}
    if train_groups & dev_groups:
        raise ValueError('Public family split leakage')
    all_inputs = {}
    for split, records in datasets.items():
        for row in records:
            key = ' '.join(row['input']['text'].split()).casefold()
            if key in all_inputs and all_inputs[key] != row['group_id']:
                raise ValueError('Exact public source spans different families')
            all_inputs[key] = row['group_id']
    for split, records in datasets.items():
        write_jsonl(OUTPUT / (split + '.jsonl'), records)
    manifest = {'at': now(), 'seed': 20261001, 'candidate_hash': CANDIDATE_HASH,
                'reviewed_hash': fingerprint(rows), 'family_audit_hash': fingerprint(audit),
                'audited_family_links': sorted(APPROVED_LINKS),
                'manual_family_links': [{'a': a, 'b': b, 'reason': 'Same evening eating-out event; invitation versus negative inclination.'} for a, b in MANUAL_LINKS],
                'excluded_candidate_ids': sorted(audit['exclusion_triggers']),
                'reviewed_candidate_count': 600, 'retained_source_families': len(groups),
                'representative_policy': 'Lexicographically first accepted pair per family; both translation directions stay together',
                'splits': {split: {'rows': len(records), 'groups': len({r['group_id'] for r in records}),
                                  'hash': fingerprint(records), 'categories': dict(Counter(r['category'] for r in records))}
                           for split, records in datasets.items()},
                'tokenizer': 'Local frozen original Qwen3-4B; shared JSON chat prompt; answer and EOS included',
                'token_budget': {'limit': 1024, 'max_encoded_tokens': max(lengths), 'total_encoded_tokens': sum(lengths),
                                 'total_supervised_tokens': sum(supervised), 'truncated_rows': 0},
                'training_ready': True, 'release_approved': False,
                'scope': 'Public training/auxiliary development pilot only. Not a release test. Source family review is conservative, not exhaustive semantic or pretraining decontamination.',
                'limitations': audit['limitations']}
    write_json(OUTPUT / 'manifest.json', manifest)
    print({'families': len(groups), 'splits': manifest['splits'], 'tokens': manifest['token_budget']}, flush=True)


if __name__ == '__main__':
    main()
