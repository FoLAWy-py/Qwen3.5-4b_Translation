"""Freeze a small reviewed fact pilot from v7, with old/public replay anchors."""
import hashlib
import json
import random
from pathlib import Path
from archive.qwen3.runtime import SYSTEM_PROMPT
from witrans_tools.common import base_manifest,fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.data import validate_record
from witrans_tools.fact_pairs import validate_fact_packet


def main():
    destination = Path('data/prepared/v14-fact-trial')
    if destination.exists():
        raise ValueError('Preserve immutable pilot plan')
    pairs = read_jsonl('data/prepared/v14-public-fact-pairs/pairs.jsonl')
    manifest = json.loads(Path('data/prepared/v14-public-fact-pairs/manifest.json').read_text(encoding='utf-8'))
    if fingerprint(pairs)!=manifest['pairs_hash']:
        raise ValueError('Reviewed fact data changed')
    for pair in pairs:
        validate_fact_packet(pair)
    v13 = json.loads(Path('data/prepared/v13-public-optimization/plan.json').read_text(encoding='utf-8'))
    old = read_jsonl('data/prepared/v11-mixed/train.jsonl')
    public = read_jsonl('data/prepared/v12-public-short/train.jsonl')
    for row in old+public:
        validate_record(row,True,True,purpose='training')
    rng = random.Random(42)
    order = list(range(len(pairs)))
    visits = []
    while len(visits)<64:
        rng.shuffle(order)
        visits.extend(order)
    visits = visits[:64]
    old_order = rng.sample(old,64)
    public_order = rng.sample(public,64)
    replay = old_order+public_order
    schedule = [{'pair':pairs[index]['id'],'replay':[old_order[i]['id'],public_order[i]['id']]}
                for i,index in enumerate(visits)]
    start = Path('models/witrans-4b-v7-critical-cpo')
    sha = hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()
    if sha!=v13['starting_adapter_sha256']:
        raise ValueError('Best starting weights changed')
    write_jsonl(destination/'pairs.jsonl',pairs)
    write_jsonl(destination/'replay.jsonl',replay)
    plan = {'at':now(),'starting_adapter':str(start),'starting_adapter_sha256':sha,
            'base':base_manifest('models/Qwen3-4B'),'prompt_hash':fingerprint(SYSTEM_PROMPT),
            'pairs_path':str(destination/'pairs.jsonl'),'pairs_hash':fingerprint(pairs),
            'replay_path':str(destination/'replay.jsonl'),'replay_hash':fingerprint(replay),
            'parent_fact_manifest':'data/prepared/v14-public-fact-pairs/manifest.json',
            'schedule':schedule,'seed':42,'updates':16,'accumulation':4,'checkpoints':[8,16],
            'learning_rate':2e-6,'beta':.1,'temperature':.1,'matching_weight':.1,
            'pair_weight':.5,'replay_weight':.5,'quantization':'nf4','max_length':1024,
            'selected_checkpoint':16,'selection':'Fixed final update; no DEV-based checkpoint selection.',
            'arms':{mode:f'models/witrans-4b-v14-fact-{mode}' for mode in ('matching','sft','cpo')},
            'first_arm':'matching','other_arms_started':False,
            'compute_protocol':'All arms score four pairs without gradients then recompute four forwards/backwards with restored CPU/CUDA dropout RNG; two replay SFT forwards/backwards per matrix. Equal calls/tokens, not a proven equal FLOP or runtime claim.',
            'expected_training_forward_calls':640,'expected_training_backward_calls':384,
            'scope':'Pilot:14 existing TRAIN families/28 matrices,64 matrix visits,128 replay records; no new independent source or generalization claim. No original-base arm.',
            'require_v13_semantic_acceptance_complete':True,'release_approved':False}
    write_json(destination/'plan.json',plan)
    print({'path':str(destination/'plan.json'),'plan_hash':fingerprint(plan),'pairs':len(pairs),'replay':len(replay)})


if __name__=='__main__':
    main()
