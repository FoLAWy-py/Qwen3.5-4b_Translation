"""Reviewed four-score fact pilot with sequential recomputation and SFT replay."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from witrans import SYSTEM_PROMPT,_LocalTranslator
from witrans_tools.common import base_manifest,fingerprint,now,read_jsonl,write_json
from witrans_tools.data import encode_example,validate_record
from witrans_tools.fact_pairs import encode_fact_packet
from witrans_tools.fact_objective import fact_objective
from witrans_tools.preference import answer_scores
from witrans_tools.recomputed_matrix import recomputed_matrix_backward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('matching','sft','cpo'))
    parser.add_argument('--plan',default='data/prepared/v14-fact-trial/plan.json')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    acceptance = json.loads(Path('runs/v13-model-acceptance.json').read_text(encoding='utf-8'))
    if not acceptance['comparison_complete']:
        raise ValueError('Finish v13 individual acceptance before another GPU trial')
    if acceptance['optimization_screen_passed']:
        raise ValueError('Reconsider best parent if v13 semantic screening succeeds')
    output = Path(plan['arms'][args.mode])
    if output.exists():
        raise ValueError('Preserve prior trial artifacts')
    pairs = read_jsonl(plan['pairs_path'])
    replay = read_jsonl(plan['replay_path'])
    if fingerprint(pairs)!=plan['pairs_hash'] or fingerprint(replay)!=plan['replay_hash']:
        raise ValueError('Frozen training data changed')
    if base_manifest('models/Qwen3-4B')!=plan['base'] or fingerprint(SYSTEM_PROMPT)!=plan['prompt_hash']:
        raise ValueError('Frozen model or prompt changed')
    start = Path(plan['starting_adapter'])
    if hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()!=plan['starting_adapter_sha256']:
        raise ValueError('Best starting adapter changed')
    if (len(plan['schedule'])!=plan['updates']*plan['accumulation']
            or plan['pair_weight']+plan['replay_weight']!=1 or plan['selected_checkpoint']!=plan['updates']):
        raise ValueError('Frozen schedule or selection invalid')
    for row in replay:
        validate_record(row,True,True,purpose='training')
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import set_seed
    set_seed(plan['seed'])
    translator = _LocalTranslator('models/Qwen3-4B',None,max_length=1024,quantization='nf4')
    encoded_pairs = {p['id']:encode_fact_packet(translator.tokenizer,p,1024) for p in pairs}
    encoded_replay = {r['id']:encode_example(translator.tokenizer,r,1024) for r in replay}
    if len(encoded_replay)!=len(replay) or len(encoded_pairs)!=len(pairs):
        raise ValueError('Duplicate frozen training IDs')
    expected_tokens = sum(2*sum(len(e['input_ids']) for row in encoded_pairs[s['pair']][0] for e in row)
                          +sum(len(encoded_replay[key]['input_ids']) for key in s['replay']) for s in plan['schedule'])
    model = prepare_model_for_kbit_training(translator.model,use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant':False})
    model = PeftModel.from_pretrained(model,start,is_trainable=True,local_files_only=True)
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters,lr=plan['learning_rate'],weight_decay=.01)
    output.mkdir(parents=True)
    write_json(output/'run_config.json',{'at':now(),'mode':args.mode,'plan':plan,'plan_hash':fingerprint(plan),
               'expected_full_tokens_including_recomputation':expected_tokens,
               'optimizer':'AdamW8bit','clip_grad_norm':1.,'weight_decay':.01,'release_approved':False})
    counters = {'forward_calls':0,'backward_calls':0,'full_tokens':0}

    def score(example):
        counters['forward_calls']+=1
        counters['full_tokens']+=len(example['input_ids'])
        tensors = {k:torch.tensor([v],device='cuda:0') for k,v in example.items()}
        labels = tensors.pop('labels')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits,labels)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    beginning = time.perf_counter()
    history,checkpoints = [],[]
    for step in range(1,plan['updates']+1):
        optimizer.zero_grad(set_to_none=True)
        loss_total,margin_total = 0.,0.
        visits = plan['schedule'][(step-1)*plan['accumulation']:step*plan['accumulation']]
        for visit in visits:
            matrix,counts = encoded_pairs[visit['pair']]
            count_tensor = torch.tensor(counts,device='cuda:0')
            objective = lambda values:fact_objective(values,count_tensor,args.mode,beta=plan['beta'],
                                       temperature=plan['temperature'],matching_weight=plan['matching_weight'])
            diagnostics = recomputed_matrix_backward(lambda i,j:score(matrix[i][j])[0],objective,
                                       scale=plan['pair_weight']/plan['accumulation'])
            counters['backward_calls']+=4
            replay_loss = 0.
            for key in visit['replay']:
                _,nll = score(encoded_replay[key])
                if not torch.isfinite(nll):
                    raise ValueError('Nonfinite replay loss')
                (nll*plan['replay_weight']/len(visit['replay'])/plan['accumulation']).backward()
                counters['backward_calls']+=1
                replay_loss+=float(nll.detach())/len(visit['replay'])
            loss_total+=(plan['pair_weight']*diagnostics['loss']+plan['replay_weight']*replay_loss)/plan['accumulation']
            margin_total+=diagnostics['assignment_margin']/plan['accumulation']
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradients')
        optimizer.step()
        record = {'step':step,'loss':loss_total,'train_assignment_margin':margin_total,**counters,
                  'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
        history.append(record)
        write_json(output/'progress.json',{'at':now(),'mode':args.mode,'history':history,'release_approved':False})
        print(record,flush=True)
        if record['peak_reserved_gib']>6.5:
            raise ValueError('Pilot exceeded frozen6.5GiB memory budget')
        if step in plan['checkpoints']:
            checkpoint = output/f'checkpoint-{step}'
            model.save_pretrained(checkpoint,safe_serialization=True)
            translator.tokenizer.save_pretrained(checkpoint)
            digest = hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()
            metadata = {'base_model_id':'Qwen/Qwen3-4B','base_revision':plan['base']['revision'],
                        'prompt_hash':plan['prompt_hash'],'adapter_sha256':digest,'smoke_only':False,
                        'parent_adapter_sha256':plan['starting_adapter_sha256'],'created_at':now(),
                        'recommended_quantization':'nf4','training_method':f'fact_{args.mode}_with_replay',
                        'quality_status':'Experimental14-family pilot; semantic acceptance pending'}
            write_json(checkpoint/'witrans_adapter.json',metadata)
            checkpoints.append({'step':step,'directory':str(checkpoint),'adapter_sha256':digest})
    torch.cuda.synchronize()
    if (counters['full_tokens']!=expected_tokens or counters['forward_calls']!=plan['expected_training_forward_calls']
            or counters['backward_calls']!=plan['expected_training_backward_calls']):
        raise ValueError('Actual training compute counters differ from frozen schedule')
    selected = next(row for row in checkpoints if row['step']==plan['selected_checkpoint'])
    write_json(output/'metrics.json',{'at':now(),'mode':args.mode,'plan_hash':fingerprint(plan),
               'history':history,'selected':selected,'checkpoints':checkpoints,**counters,
               'seconds_including_save':time.perf_counter()-beginning,
               'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
               'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
               'semantic_accepted':False,'release_approved':False})
    print({'selected':selected,**counters,'release_approved':False},flush=True)


if __name__=='__main__':
    main()
