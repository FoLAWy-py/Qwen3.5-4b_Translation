"""Retrieve source overlap for explicit family review; never inspect held-out labels."""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

IDIOMS = {
    'long-shot':r'\blong shot\b', 'on-fence':r'\bon the fence\b',
    'learn-ropes':r'\b(?:learn\w*|know\w*) the ropes\b',
    'grain-salt':r'\bgrain of salt\b', 'nail-head':r'\bnail on the head\b',
    'pass-buck':r'\bpass\w* the buck\b', 'draw-line':r'\bdraw\w* the line\b',
    'square-one':r'\bsquare one\b', 'red-herring':r'\bred herring\b',
    'brush-up':r'\bbrush\w* up\b', 'sit-on':r'\b(?:sit\w*|sat) on (?:it|the report)\b',
    'finger-on':r'\bput\w* (?:\w+ )?finger on\b',
    'eye-on':r'\bkeep\w* an eye on\b', 'join-forces':r'\bjoin\w* forces\b',
    'bump-into':r'\bbump\w* into\b', 'tied-up':r'\btied up\b',
}


def normalized(text):
    return re.sub(r'\s+', ' ', text.strip().casefold())


def grams(text):
    compact = re.sub(r'[^\w]', '', normalized(text))
    return set(compact[i:i+4] for i in range(max(0, len(compact)-3)))


def main():
    candidates = read_jsonl('data/prepared/v10-source/reviewed-partial.jsonl')
    progress = __import__('json').loads(Path('runs/v10-source-review-progress.json').read_text(encoding='utf-8'))
    if not progress['complete_individual_reading'] or len(candidates)!=186:
        raise ValueError('Complete individual source review required')
    prior = {}
    scanned = []
    for path in sorted(Path('data/prepared').glob('**/*.jsonl')):
        if 'v10-source' in path.parts:
            continue
        rows = read_jsonl(path)
        scanned.append({'path':str(path),'hash':fingerprint(rows)})
        for row in rows:
            if not isinstance(row.get('input'),dict) or not row['input'].get('text'):
                continue
            heldout = (any(term in path.name for term in ('dev','test','acceptance'))
                       or row.get('source',{}).get('training_allowed') is False)
            key = normalized(row['input']['text'])
            entry = prior.setdefault(key, {'input':row['input'], 'ids':set(), 'groups':set(),
                'paths':set(), 'heldout':False})
            entry['ids'].add(row.get('id',''))
            entry['groups'].add(row.get('group_id',''))
            entry['paths'].add(str(path))
            entry['heldout'] |= heldout
    entries = list(prior.values())
    posting = defaultdict(list)
    features = []
    for i, row in enumerate(entries):
        feature = grams(row['input']['text'])
        features.append(feature)
        for gram in feature:
            posting[gram].append(i)
    matches = []
    for row in candidates:
        feature = grams(row['input']['text'])
        counts = Counter(i for gram in feature for i in posting[gram])
        top = {'train':[], 'heldout':[]}
        for i, common in counts.items():
            other = entries[i]
            if other['input']['target_lang'] != row['input']['target_lang']:
                continue
            score = common/math.sqrt(len(feature)*len(features[i])) if feature and features[i] else 0
            if score < .25:
                continue
            split = 'heldout' if other['heldout'] else 'train'
            result = {'score':round(score,6),'ids':sorted(other['ids']),
                'groups':sorted(other['groups']), 'paths':sorted(other['paths']),
                'exact':normalized(row['input']['text'])==normalized(other['input']['text'])}
            if split=='train':
                result['source'] = other['input']['text']
            top[split].append(result)
        for split in top:
            top[split] = sorted(top[split],key=lambda x:-x['score'])[:3]
        matches.append({'id':row['id'],'group_id':row['group_id'],
            'source':row['input']['text'],'matches':top})
    write_jsonl('runs/v10-source-overlap-candidates.jsonl',matches)
    idiom_families = []
    for row in candidates:
        if row['category']!='hard' or row['input']['target_lang']!='zh-CN':
            continue
        for sense, pattern in IDIOMS.items():
            if not re.search(pattern,normalized(row['input']['text'])):
                continue
            linked = {'train':set(),'heldout':set()}
            for prior_row in entries:
                if prior_row['input']['target_lang']=='zh-CN' and re.search(pattern,normalized(prior_row['input']['text'])):
                    linked['heldout' if prior_row['heldout'] else 'train'].update(prior_row['groups'])
            idiom_families.append({'group_id':row['group_id'],'sense':sense,
                'prior_groups':{split:sorted(groups) for split,groups in linked.items()}})
    report = {'at':now(),'reviewed_hash':fingerprint(candidates), 'rows':len(candidates),
        'prior_unique_sources':len(entries), 'scanned':scanned,
        'exact_overlaps':[r['id'] for r in matches if any(m['exact'] for s in r['matches'].values() for m in s)],
        'idiom_family_links':idiom_families,
        'retrieval':'Character four-gram cosine >=0.25; top three per split. Held-out source text and labels never emitted. Candidate retrieval is not a complete semantic duplicate detector.',
        'family_review_complete':False,'training_ready':False,'release_approved':False}
    write_json('runs/v10-source-overlap-audit.json',report)
    print({k:v for k,v in report.items() if k!='scanned'},flush=True)


if __name__=='__main__':
    main()
