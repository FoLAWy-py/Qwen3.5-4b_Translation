"""Fetch attributed Tatoeba pairs; create pending candidates, never training labels."""
import hashlib
import io
import json
import math
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from archive.qwen3.scripts.audit_v10_sources import grams
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

DEST = Path('data/public/tatoeba-manythings-20261001')
URL = 'https://www.manythings.org/anki/cmn-eng.zip'


def normalized(text):
    return re.sub(r'\W+', '', unicodedata.normalize('NFKC', text).casefold())


def main():
    if DEST.exists():
        raise ValueError('Preserve acquisition snapshot; use a fresh directory for another release')
    DEST.mkdir(parents=True)
    evidence = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name, url in [('cmn-eng.zip', URL), ('manythings-index.html', 'https://www.manythings.org/anki/'),
                          ('tatoeba-downloads.html', 'https://tatoeba.org/en/downloads')]:
            data = bytearray()
            with client.stream('GET', url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > 32 * 1024 * 1024:
                        raise ValueError('Acquisition exceeds 32MiB cap')
                evidence.append({'url': url, 'resolved_url': str(response.url), 'file': name,
                                 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                 'etag': response.headers.get('etag'),
                                 'last_modified': response.headers.get('last-modified'), 'at': now()})
            (DEST / name).write_bytes(data)
    with zipfile.ZipFile(io.BytesIO((DEST / 'cmn-eng.zip').read_bytes())) as archive:
        if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError('Expanded archive too large')
        members = archive.namelist()
        if 'cmn.txt' not in members:
            raise ValueError('Expected attributed Mandarin file absent')
        lines = archive.read('cmn.txt').decode('utf-8-sig').splitlines()
        about = [name for name in members if 'about' in name.casefold()]
        for i, name in enumerate(about):
            (DEST / f'about-{i}.txt').write_bytes(archive.read(name))
    # Read source fields only. Held-out labels are never used to create candidates.
    prior = {}
    prior_snapshots = []
    for path in sorted(Path('data/prepared').glob('**/*.jsonl')):
        rows = read_jsonl(path)
        prior_snapshots.append({'path': str(path), 'hash': fingerprint(rows)})
        for row in rows:
            text = row.get('input', {}).get('text') if isinstance(row.get('input'), dict) else None
            if not text:
                continue
            heldout = any(term in path.name for term in ('dev', 'test', 'acceptance')) or row.get('source', {}).get('training_allowed') is False
            key = normalized(text)
            value = prior.setdefault(key, {'features': grams(text), 'heldout': False})
            value['heldout'] |= heldout
    entries = list(prior.values())
    postings = defaultdict(list)
    for i, entry in enumerate(entries):
        for gram in entry['features']:
            postings[gram].append(i)
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    raw, rejected = [], Counter()
    attribution_re = re.compile(r'^CC-BY 2\.0 \(France\) Attribution: tatoeba\.org #(\d+) \(([^)]+)\) & #(\d+) \(([^)]+)\)$')
    for line_no, line in enumerate(lines, 1):
        fields = line.split('\t')
        if len(fields) != 3:
            rejected['malformed_fields'] += 1
            continue
        en, zh, attribution = fields
        match = attribution_re.fullmatch(attribution)
        if not match:
            rejected['unparsed_attribution'] += 1
            continue
        en_id, en_author, zh_id, zh_author = match.groups()
        # Links and identical sentences define a conservative family within this snapshot.
        for node in ('id:' + en_id, 'en:' + normalized(en), 'id:' + zh_id, 'zh:' + normalized(zh)):
            union('id:' + en_id, node)
        raw.append({'line': line_no, 'en': en, 'zh': zh, 'en_id': en_id, 'zh_id': zh_id,
                    'en_author': en_author, 'zh_author': zh_author, 'attribution': attribution})
    eligible = []
    for row in raw:
        en, zh = row['en'], row['zh']
        if not (6 <= len(en.split()) <= 25 and 6 <= len(zh) <= 90
                and en.endswith(('.', '?', '!')) and re.search(r'[\u3400-\u9fff]', zh)):
            rejected['length_or_language_candidate_filter'] += 1
            continue
        if any(normalized(text) in prior for text in (en, zh)):
            rejected['prior_exact_overlap'] += 1
            continue
        scores = []
        for text in (en, zh):
            features = grams(text)
            counts = Counter(i for gram in features for i in postings[gram])
            for i, count in counts.items():
                other = entries[i]
                score = count / math.sqrt(len(features) * len(other['features'])) if features and other['features'] else 0
                scores.append((score, other['heldout']))
        if any(score >= .65 for score, _ in scores):
            rejected['conservative_prior_near_overlap'] += 1
            continue
        row['max_prior_similarity'] = max((score for score, _ in scores), default=0)
        row['family'] = find('id:' + row['en_id'])
        eligible.append(row)
    random.Random(20261001).shuffle(eligible)
    selected, seen = [], set()
    candidate_features, candidate_postings = [], defaultdict(list)
    for row in eligible:
        if row['family'] in seen:
            continue
        features = grams(row['en']) | grams(row['zh'])
        counts = Counter(i for gram in features for i in candidate_postings[gram])
        if any(count / math.sqrt(len(features) * len(candidate_features[i])) >= .70 for i, count in counts.items() if features and candidate_features[i]):
            continue
        seen.add(row['family'])
        i = len(candidate_features)
        candidate_features.append(features)
        for gram in features:
            candidate_postings[gram].append(i)
        selected.append({'id': f'public-short-{len(selected)+1:04d}',
                         'group_id': 'tatoeba-' + row['family'], **row, 'status': 'pending',
                         'training_ready': False, 'category': None,
                         'source': {'name': 'Tatoeba via ManyThings attributed sentence pairs',
                                    'license': 'CC BY 2.0 FR', 'license_url': 'https://creativecommons.org/licenses/by/2.0/fr/',
                                    'download_url': URL, 'attribution': row['attribution'],
                                    'sentence_urls': [f'https://tatoeba.org/en/sentences/show/{row[key]}' for key in ('en_id', 'zh_id')],
                                    'training_allowed': True, 'external_labeling_allowed': True}})
        if len(selected) == 1000:
            break
    write_jsonl(DEST / 'candidates.jsonl', selected)
    report = {'at': now(), 'files': evidence, 'zip_members': members, 'raw_lines': len(lines),
              'parsed_pairs': len(raw), 'eligible_pairs': len(eligible), 'pending_groups': len(selected),
              'rejected_filters': dict(rejected), 'prior_source_count': len(prior),
              'prior_snapshots': prior_snapshots, 'candidate_hash': fingerprint(selected),
              'training_ready': False, 'release_approved': False,
              'limits': 'Pending source/translation/license review. ID/link families and lexical retrieval are not exhaustive semantic clustering. Public corpus may have appeared in base pretraining. No held-out reference labels used.'}
    write_json(DEST / 'acquisition.json', report)
    print({key: value for key, value in report.items() if key not in ('files', 'prior_snapshots')}, flush=True)


if __name__ == '__main__':
    main()
