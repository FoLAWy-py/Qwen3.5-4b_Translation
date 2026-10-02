"""Conservative family grouping and held-out isolation using source fields only."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_v10_sources import IDIOMS, grams, normalized
from scripts.prepare_public_short_pilot import DEST
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

EXTRA_IDIOMS = {
    'cut-corners': r'\bcut\w* corners\b',
    'first-sight': r'\bat first sight\b',
    'eleventh-hour': r'\beleventh hour\b',
    'look-down-on': r'\blook\w* down on\b',
    'will-way': r'\bwhere there.{0,12}will.{0,20}way\b',
    'hands-full': r'\bhands full\b',
}


def template(text):
    text = normalized(text).replace("n't", ' not')
    text = re.sub(r'\b(tom|mary|john|bob|alice)\b', 'person', text)
    text = re.sub(r'\b(i|you|he|she|we|they|me|him|her|us|them)\b', 'person', text)
    text = re.sub(r'\b(my|your|his|her|our|their)\b', 'possessive', text)
    text = re.sub(r'\b\d+(?:[.,]\d+)*\b', 'number', text)
    # Polarity variants belong to one family even though their labels differ.
    text = re.sub(r'\b(not|never)\b', '', text)
    return re.sub(r'\W+', '', text)


def main():
    rows = read_jsonl(DEST / 'reviewed-partial.jsonl')
    prior, scanned = {}, []
    for path in sorted(Path('data/prepared').glob('**/*.jsonl')):
        records = read_jsonl(path)
        source_snapshot = []
        for record in records:
            inp = record.get('input')
            if not isinstance(inp, dict) or not inp.get('text'):
                continue
            heldout = (any(term in path.name.casefold() for term in ('dev', 'test', 'acceptance'))
                       or record.get('source', {}).get('training_allowed') is False)
            text = inp['text']
            source_snapshot.append({'id': record.get('id'), 'group_id': record.get('group_id'),
                                    'text': text, 'heldout': heldout})
            item = prior.setdefault(normalized(text), {'text': text, 'heldout': False,
                                                       'ids': set(), 'groups': set()})
            item['heldout'] |= heldout
            item['ids'].add(record.get('id', ''))
            item['groups'].add(record.get('group_id', ''))
        scanned.append({'path': str(path), 'source_fields_hash': fingerprint(source_snapshot)})
    entries = list(prior.values())
    features = [grams(item['text']) for item in entries]
    postings = defaultdict(list)
    for index, feature in enumerate(features):
        for gram in feature:
            postings[gram].append(index)
    patterns = IDIOMS | EXTRA_IDIOMS
    prior_idioms = defaultdict(set)
    for item in entries:
        if item['heldout']:
            for name, pattern in patterns.items():
                if re.search(pattern, normalized(item['text'])):
                    prior_idioms[name].update(item['groups'])
    parent = {row['id']: row['id'] for row in rows}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    template_owner, original_owner = {}, {}
    public_features, public_postings = [], defaultdict(list)
    links, exclusions, overlap_rows = [], {}, []
    for index, row in enumerate(rows):
        for owner, key, reason in ((template_owner, template(row['en']), 'delexicalized-polarity-template'),
                                   (original_owner, row['group_id'], 'archive-linked-family')):
            if key in owner:
                union(row['id'], owner[key])
                links.append({'a': row['id'], 'b': owner[key], 'reason': reason})
            else:
                owner[key] = row['id']
        feature = grams(row['en'])
        counts = Counter(i for gram in feature for i in public_postings[gram])
        for other, common in counts.items():
            score = common / math.sqrt(len(feature)*len(public_features[other])) if feature and public_features[other] else 0
            if score >= .70:
                union(row['id'], rows[other]['id'])
                links.append({'a': row['id'], 'b': rows[other]['id'], 'reason': 'english-fourgram-conservative', 'score': round(score, 6)})
        public_features.append(feature)
        for gram in feature:
            public_postings[gram].append(index)
        matches = []
        for language in ('en', 'zh'):
            feature = grams(row[language])
            counts = Counter(i for gram in feature for i in postings[gram])
            for other, common in counts.items():
                score = common / math.sqrt(len(feature)*len(features[other])) if feature and features[other] else 0
                item = entries[other]
                cutoff = .45 if item['heldout'] else .55
                if score >= cutoff or normalized(row[language]) == normalized(item['text']):
                    matches.append({'language': language, 'score': round(score, 6), 'heldout': item['heldout'],
                                    'ids': sorted(item['ids']), 'groups': sorted(item['groups'])})
        senses = [name for name, pattern in patterns.items()
                  if re.search(pattern, normalized(row['en'])) and prior_idioms[name]]
        if matches or senses:
            exclusions[row['id']] = {'source_matches': matches, 'heldout_idiom_families': senses}
        overlap_rows.append({'id': row['id'], 'matches': matches, 'heldout_idiom_families': senses})
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['id'])].append(row['id'])
    excluded_roots = {find(key) for key in exclusions}
    kept = [row for row in rows if find(row['id']) not in excluded_roots]
    assignments = [{'id': row['id'], 'family_id': 'public-v12-' + find(row['id']),
                    'excluded': find(row['id']) in excluded_roots} for row in rows]
    write_jsonl(DEST / 'family-assignments.jsonl', assignments)
    write_jsonl(DEST / 'source-overlaps.jsonl', overlap_rows)
    report = {'at': now(), 'reviewed_hash': fingerprint(rows), 'reviewed_pairs': len(rows),
              'prior_unique_sources': len(entries), 'scanned_source_snapshots': scanned,
              'family_links': links, 'exclusion_triggers': exclusions,
              'excluded_families': len(excluded_roots), 'excluded_pairs': len(rows)-len(kept),
              'retained_pairs': len(kept), 'retained_families': len(groups)-len(excluded_roots),
              'multi_pair_families': {key: value for key, value in groups.items() if len(value)>1},
              'rules': 'Same archive family, exact delexicalized polarity template, English 4gram cosine >=0.70 union. Exclude whole family at prior heldout cosine >=0.45 or prior train >=0.55, or shared heldout idiom. Conservative exclusion is not proof of semantic equivalence.',
              'limitations': 'Archive links do not include the full multilingual Tatoeba graph; lexical retrieval cannot prove absence of semantic or pretraining overlap. Chinese script differences may escape retrieval.',
              'training_ready': False, 'family_review_complete': False, 'release_approved': False}
    write_json(DEST / 'family-audit.json', report)
    print({key: report[key] for key in ('reviewed_pairs', 'retained_pairs', 'retained_families', 'excluded_families', 'excluded_pairs')}, flush=True)


if __name__ == '__main__':
    main()
