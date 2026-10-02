"""Freeze the transferred best-validated SFT + critical-span CPO recipe."""
import copy
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record
from witrans_tools.critical_spans import validated_annotations, encode_critical_spans


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',default='v1',choices=('v1','v2'))
    args=parser.parse_args()
    dest = Path(f'data/prepared/qwen35-{args.version}')
    assert not dest.exists(), 'Preserve frozen comparison'
    acquisition = load('runs/qwen35-acquisition.json')
    sft = read_jsonl('data/prepared/v5-mixed/train.jsonl')
    assert fingerprint(sft)==load('data/prepared/v5-mixed/manifest.json')['train']['hash']
    reviewed = {r['id']:r for r in read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')}
    repairs = []
    for i,row in enumerate(sft):
        newer = reviewed.get(row['id'])
        if newer and newer.get('reference_repair'):
            assert row['input']==newer['input'] and row['group_id']==newer['group_id']
            if row['output'] != newer['output']:
                repairs.append(dict(id=row['id'],old_output=row['output'],new_output=newer['output']))
                sft[i] = copy.deepcopy(newer)
    pairs = read_jsonl('data/prepared/v6-preference/train.jsonl')
    spans = load('data/prepared/v7-critical-spans/annotations.json')
    annotations = validated_annotations(pairs,spans)
    known = read_jsonl('data/prepared/v12-factorial-v2/dev.jsonl')
    public = read_jsonl('data/prepared/v12-factorial-v2/public-dev.jsonl')
    for row in sft+pairs:
        validate_record(row,True,True,purpose='training')
    for row in pairs:
        assert row['preference_review']['hash']==fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')})
    normalize = lambda s:' '.join(s.split()).casefold()
    assert not {r['group_id'] for r in sft+pairs}&{r['group_id'] for r in known+public}
    assert not {normalize(r['input']['text']) for r in sft+pairs}&{normalize(r['input']['text']) for r in known+public}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    enc_sft = {r['id']:encode_example(tokenizer,r,1024) for r in sft}
    enc_pairs = {}
    for row in pairs:
        chosen = encode_example(tokenizer,row,1024)
        if row['id'] in annotations:
            entry = annotations[row['id']]
            chosen = encode_critical_spans(tokenizer,row,entry['critical_spans'],1024,entry['multiplier'])
        enc_pairs[row['id']] = (chosen,encode_example(tokenizer,{**row,'output':row['rejected']},1024))
    for row in known+public:
        example = encode_example(tokenizer,row,1024)
        prompt_tokens = sum(v==-100 for v in example['labels'])
        assert prompt_tokens+256<=1024
    def schedule(rows,updates,accumulation):
        order = [r['id'] for r in rows]
        random.Random(42).shuffle(order)
        return [order[i%len(order)] for i in range(updates*accumulation)]
    sft_order, cpo_order = schedule(sft,85,16),schedule(pairs,22,8)
    sft_tokens = sum(len(enc_sft[k]['input_ids']) for k in sft_order)
    cpo_tokens = sum(sum(len(e['input_ids']) for e in enc_pairs[k]) for k in cpo_order)
    write_jsonl(dest/'sft.jsonl',sft)
    write_jsonl(dest/'pairs.jsonl',pairs)
    write_json(dest/'annotations.json',spans)
    plan = dict(at=now(),model_id=acquisition['model_id'],base=acquisition,
        base_dir='models/Qwen3.5-4B',prompt_hash=fingerprint(SYSTEM_PROMPT),seed=42,
        method='Transfer current validated v7 recipe: supervised translation then reviewed critical-span CPO.',
        original_base_comparison=True,scope_override='User explicitly requested Qwen3.5-4B base versus fine-tuned. This does not restart historical Qwen3/v12 factorial experiments.',
        runtime='isolated .venv-qwen35 via uv; text-only Qwen3_5ForCausalLM, official vision/MTP weights unused',
        quantization='nf4',max_length=1024,max_new_tokens=256,memory_cap_gib=6.5,
        lora=dict(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']),
        sft=dict(path=str(dest/'sft.jsonl'),hash=fingerprint(sft),updates=85,accumulation=16,learning_rate=1e-5,
            order=sft_order,full_tokens=sft_tokens,output=f'models/witrans-qwen35-{args.version}-sft'),
        cpo=dict(path=str(dest/'pairs.jsonl'),hash=fingerprint(pairs),updates=22,accumulation=8,learning_rate=1e-5,beta=.1,
            annotations=str(dest/'annotations.json'),annotations_hash=fingerprint(spans),order=cpo_order,
            full_tokens=cpo_tokens,output=f'models/witrans-qwen35-{args.version}-critical-cpo'),
        datasets=dict(known=dict(path='data/prepared/v12-factorial-v2/dev.jsonl',hash=fingerprint(known),rows=200),
            public=dict(path='data/prepared/v12-factorial-v2/public-dev.jsonl',hash=fingerprint(public),rows=116)),
        selection='Fixed final85 SFT and final22 CPO. No loss-based DEV selection; generated outputs will be individually reviewed.',
        training_repairs=repairs,existing_best='models/witrans-4b-v7-critical-cpo',
        limitations='Fresh Qwen3.5 LoRA cannot inherit Qwen3/v2/v5 weights. Recipe transfer is not exact reproduction of the historical multi-round lineage, nor an equal-total-training architecture ablation. One seed, repeated DEV only.',
        gpu_predecessor='runs/v19-balanced-repair-job.json',release_approved=False)
    write_json(dest/'plan.json',plan)
    write_json(f'runs/qwen35-{args.version}-user-scope.json',dict(at=now(),authorized_comparison='Qwen3.5 official checkpoint before project fine-tuning versus transferred v7-recipe candidate',
        plan_hash=fingerprint(plan),previous_scope='No new historical Qwen3 base comparison remains; specific user instruction authorizes this new-model control.',release_approved=False))
    print({'plan_hash':fingerprint(plan),'sft_rows':len(sft),'cpo_pairs':len(pairs),'sft_full_tokens':sft_tokens,
        'cpo_full_tokens':cpo_tokens,'reference_repairs':len(repairs)})


if __name__=='__main__':
    main()
