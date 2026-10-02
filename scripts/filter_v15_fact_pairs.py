"""Exclude a coarse ambiguous deadline contrast; preserve the failed pilot."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.fact_pairs import validate_fact_packet


def main():
    destination = Path('data/prepared/v15-public-fact-pairs')
    if destination.exists():
        raise ValueError('Preserve immutable correction')
    old = read_jsonl('data/prepared/v14-public-fact-pairs/pairs.jsonl')
    old_manifest = json.loads(Path('data/prepared/v14-public-fact-pairs/manifest.json').read_text(encoding='utf-8'))
    if fingerprint(old)!=old_manifest['pairs_hash']:
        raise ValueError('Legacy training evidence changed')
    excluded = [row for row in old if row['group_id']=='public-v12-public-short-0310']
    kept = [row for row in old if row['group_id']!='public-v12-public-short-0310']
    if len(excluded)!=2 or len(kept)!=26:
        raise ValueError('Unexpected audited deadline family')
    for row in kept:
        validate_fact_packet(row)
    write_jsonl(destination/'pairs.jsonl',kept)
    manifest = {'at':now(),'parent_manifest':'data/prepared/v14-public-fact-pairs/manifest.json',
                'parent_pairs_hash':fingerprint(old),'pairs_hash':fingerprint(kept),
                'families':len({row['group_id'] for row in kept}),'matrices':len(kept),
                'excluded':[{'id':row['id'],'group_id':row['group_id'],'record_hash':fingerprint(row)} for row in excluded],
                'reviewer':'Codex','review_decision':'Exclude from contrastive negatives; positive sentence itself need not be rejected.',
                'reason':'Coarse tomorrow-afternoon interval gives insufficient precision for a strict by/before boundary contrast. In ordinary translation the off-diagonal cannot reliably be declared incorrect without a precise clock deadline or explicit inclusive/exclusive context.',
                'interpretation':'Conservative negative-data audit; parent teacher-forced preference is not proof of a decoded factual error. Legacy v14 used these two matrices and is not clean causal method evidence.',
                'training_allowed':True,'training_started':False,'new_independent_sources':0,'release_approved':False}
    write_json(destination/'manifest.json',manifest)
    print({'retained_matrices':len(kept),'families':manifest['families'],'excluded':len(excluded)})


if __name__=='__main__':
    main()
