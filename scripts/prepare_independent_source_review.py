"""Local-only review packet and full-corpus neighbor retrieval, before outputs."""
import argparse
import json
from pathlib import Path
from rapidfuzz import fuzz,process
from witrans_tools.common import fingerprint,read_jsonl,write_json,write_jsonl
from scripts.freeze_independent_tests import normalize,retrieve_neighbors

ROOT=Path('data/independent-20261002-v2-drafts')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=0);parser.add_argument('--end',type=int,default=50)
    args=parser.parse_args()
    batches=read_jsonl(ROOT/'batches.jsonl');entries=[]
    for batch in sorted(batches,key=lambda b:b['batch_index']):
        byid={r['id']:r for r in batch['advisory_audit']['reviews']}
        cardbyid={r['id']:r for r in batch['cards']}
        for scene in sorted(batch['authored']['scenes'],key=lambda s:s['id']):
            entries.append(dict(id=scene['id'],card=cardbyid[scene['id']],scene=scene,audit=byid[scene['id']],
                author_metadata=batch['author_metadata'],audit_metadata=batch['audit_metadata']))
    old=json.loads(Path('runs/takeover-20261002/historic-source-catalog.json').read_text(encoding='utf-8'))['texts']
    pool={normalize(e['scene'][k]):e['id'] for e in entries for k in ('en','zh')}
    packet=[]
    for entry in entries:
        scene=entry['scene'];neighbors={};threshold={}
        for key in ('en','zh'):
            text=normalize(scene[key])
            neighbors[key]=[dict(text=match,ratio=score/100,locations=old[match][:2]) for match,score,_ in process.extract(text,list(old),scorer=fuzz.ratio,limit=3)]
            threshold[key]=retrieve_neighbors(text,old,{k:v for k,v in pool.items() if v!=entry['id']})
        packet.append(dict(**entry,closest_historic=neighbors,threshold_matches=threshold,
            semantic_family_adjudication='pending; matches are retrieval only, no automatic independence proof'))
    write_jsonl(ROOT/'local-source-review-packet.jsonl',packet)
    write_json(ROOT/'local-retrieval-receipt.json',dict(groups=len(packet),old_normalized_texts=len(old),
        packet_hash=fingerprint(packet),candidate_outputs_read=False,external_transmission=False,
        threshold='normalized full-corpus RapidFuzz ratio>=65',all_source_semantic_review_required=True))
    for i,entry in enumerate(packet):
        if not args.start<=i<args.end:continue
        s=entry['scene']
        print(i,entry['id'],'EN',s['en'],'ZH',s['zh'],'CTX',s['context_en'],'/',s['context_zh'],
            'MULTI',s['multisentence'],'API_AUDIT',entry['audit'],
            'OLD',entry['closest_historic']['en'][0],entry['closest_historic']['zh'][0],
            'THRESHOLD',entry['threshold_matches'])

if __name__=='__main__':main()
