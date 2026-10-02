"""Freeze individually reviewed training sources after conservative family exclusion."""
import json
import re
import copy
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

# Explicit root judgement after viewing all original sources and training-side
# retrievals. Related training scenarios stay in their earlier source family.
TRAIN_FAMILY_JOINS = {
    'v10-travel-05':'v3-train-013',
    'v10-food-03':'v4-mining-10',
    'v10-academic-07':'v2-a258020700a29c45',
    'v10-academic-13':'v3-train-022',
}


def main():
    destination = Path('data/prepared/v10-source-family-checked')
    if destination.exists():
        raise ValueError('Preserve frozen family-checked training sources')
    root = Path('data/prepared/v10-source')
    reviewed = read_jsonl(root/'reviewed-partial.jsonl')
    decisions = read_jsonl('data/generated/v10_source_decisions.jsonl')
    progress = json.loads(Path('runs/v10-source-review-progress.json').read_text(encoding='utf-8'))
    audit = json.loads(Path('runs/v10-source-overlap-audit.json').read_text(encoding='utf-8'))
    matches = read_jsonl('runs/v10-source-overlap-candidates.jsonl')
    if not progress['complete_individual_reading'] or len(decisions)!=200 or len(reviewed)!=186:
        raise ValueError('All200 individual decisions required')
    if audit['reviewed_hash']!=fingerprint(reviewed) or audit['exact_overlaps']:
        raise ValueError('Source retrieval evidence changed or exact overlap found')
    reviewed_source_hash = fingerprint(reviewed)
    for entry in audit['scanned']:
        if fingerprint(read_jsonl(entry['path']))!=entry['hash']:
            raise ValueError('Prior split changed after overlap retrieval')
    # Exclude whole pairs for every >=.25 held-out retrieval, rather than opening
    # held-out source text or labels to decide if the overlap is harmless.
    excluded = {r['group_id'] for r in matches if r['matches']['heldout']}
    excluded.update(r['group_id'] for r in audit['idiom_family_links'] if r['prior_groups']['heldout'])
    expected_excluded = {'v10-travel-10','v10-travel-14','v10-food-04',
                         'v10-academic-03','v10-academic-10'}
    if excluded!=expected_excluded:
        raise ValueError('Root-reviewed exclusion set changed')
    bound = {d['id']:d for d in decisions}
    heldout_groups,heldout_texts = set(),set()
    for entry in audit['scanned']:
        path = Path(entry['path'])
        for row in read_jsonl(path):
            if 'input' in row and (any(term in path.name for term in ('dev','test','acceptance'))
                                 or row.get('source',{}).get('training_allowed') is False):
                heldout_groups.add(row.get('group_id'))
                heldout_texts.add(row['input']['text'].strip().casefold())
    train = []
    for original in reviewed:
        row = copy.deepcopy(original)
        if bound[row['id']]['reviewed_hash']!=fingerprint({'input':row['input'],'output':row['output']}):
            raise ValueError('Individual approval changed')
        original_group = row['group_id']
        if original_group in excluded:
            continue
        row['source']['family_audit'] = {'original_group':original_group,
            'reviewer':'Codex','decision':'Conservative training-family join' if original_group in TRAIN_FAMILY_JOINS else 'Individually distinct scenario after source reading and bounded overlap retrieval',
            'retrieval_report':'runs/v10-source-overlap-audit.json'}
        row['group_id'] = TRAIN_FAMILY_JOINS.get(original_group,original_group)
        validate_record(row,True,True)
        if row['group_id'] in heldout_groups or row['input']['text'].strip().casefold() in heldout_texts:
            raise ValueError('Held-out family or exact source leakage')
        train.append(row)
    if len(train)!=176 or len({r['id'] for r in train})!=176:
        raise ValueError('Unexpected frozen source count')
    if any(count!=2 for count in Counter(r['group_id'] for r in train).values()):
        raise ValueError('Both directions must share a family')
    manifest = {'at':now(),'train_hash':fingerprint(train),'rows':len(train),
        'groups':len({r['group_id'] for r in train}),
        'new_groups':len({r['group_id'] for r in train if r['group_id'].startswith('v10-')}),
        'categories':dict(Counter(r['category'] for r in train)),
        'directions':dict(Counter(r['input']['target_lang'] for r in train)),
        'context_rows':sum(bool(r['input']['context']) for r in train),
        'glossary_rows':sum(bool(r['input']['glossary']) for r in train),
        'multisentence_rows_by_punctuation':sum(len(re.findall(r'[.!?](?:\s|$)|[。！？]',r['input']['text']))>=2 for r in train),
        'reviewed_source_hash':reviewed_source_hash,'individual_decisions_hash':fingerprint(decisions),
        'source_quality_excluded_rows':14,'heldout_retrieval_excluded_groups':sorted(excluded),
        'heldout_retrieval_excluded_rows':10,'training_family_joins':TRAIN_FAMILY_JOINS,
        'complete_individual_reading':True,'bounded_family_and_split_audit_complete':True,
        'training_ready':True,'release_approved':False,
        'limitations':'Original fictional model-generated project data with explicit generation provenance; AI acceptance, not professional human review or independent references. Character four-gram and listed idiom retrieval do not prove absence of every semantic near duplicate. Whole held-out matches conservatively excluded without reading held-out texts or labels. Training only, never a release test. Multi-sentence count is punctuation-based, not a release coverage assertion.'}
    write_jsonl(destination/'train.jsonl',train)
    write_json(destination/'manifest.json',manifest)
    write_json('runs/v10-source-family-review.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':
    main()
