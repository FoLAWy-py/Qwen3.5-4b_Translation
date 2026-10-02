"""Freeze the missing original-base comparison on already-known DEV only."""
from pathlib import Path
import argparse
import json
from witrans import SYSTEM_PROMPT
from witrans_tools.common import base_manifest, fingerprint, now, read_jsonl, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-plan',action='store_true')
    args = parser.parse_args()
    destination = Path('runs/v11-base-dev-plan.json')
    if args.review_plan:
        original = json.loads(destination.read_text(encoding='utf-8'))
        review_path = Path('runs/v11-base-dev-review-plan.json')
        if review_path.exists():
            raise ValueError('Preserve baseline review plan')
        plan = {**original,'outputs':{'baseline':original['output']},
            'additional_review_cache_stems':['v8-selected'],
            'source_protocol_plan_hash':fingerprint(original)}
        write_json(review_path,plan)
        print({'review_plan':str(review_path),'source_protocol_plan_hash':plan['source_protocol_plan_hash']},flush=True)
        return
    if destination.exists():
        raise ValueError('Preserve frozen baseline protocol')
    source = 'data/prepared/v4/dev.jsonl'
    rows = read_jsonl(source)
    expected = 'a4ec0478d5aca8b263e605bfe3e7ec2b57dd9411510135cfaf227575ede03237'
    if fingerprint(rows)!=expected or len(rows)!=200:
        raise ValueError('Known DEV changed')
    plan = {'at':now(),'input':source,'development_hash':expected,
        'base':base_manifest('models/Qwen3-4B'),'adapter':None,
        'prompt_hash':fingerprint(SYSTEM_PROMPT),'quantization':'nf4',
        'decoding':{'do_sample':False,'max_length':1024,'max_new_tokens':256},
        'output':'runs/v11-base-dev.jsonl','outputs':{'baseline':'runs/v11-base-dev.jsonl'},
        'additional_review_cache_stems':['v8-selected'],'release_approved':False,
        'scope':'Already-known200-row DEV original-base comparison only. No new release test; normal strict JSON parser, no repair. Results require individual semantic review.'}
    write_json(destination,plan)
    print(plan,flush=True)

if __name__=='__main__':
    main()
