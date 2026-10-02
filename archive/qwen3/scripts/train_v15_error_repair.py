"""Low-rate actual-error CPO plus diverse SFT replay from the best NF4 adapter."""
import hashlib
import argparse
import json
import time
from pathlib import Path
from archive.qwen3.runtime import SYSTEM_PROMPT, WiTrans
from witrans_tools.common import base_manifest, fingerprint, now, read_jsonl, write_json
from witrans_tools.data import encode_example, validate_record
from witrans_tools.preference import answer_scores, cpo_loss
from witrans_tools.qwen35_trial_eligibility import require_eligible_trial


def repair_base_receipt(base_dir, is_qwen35):
    if not is_qwen35:
        return base_manifest(base_dir)
    if base_dir!='models/Qwen3.5-4B':
        raise ValueError('Qwen3.5 repair requires its frozen local base directory')
    receipt=json.loads((Path(base_dir)/'witrans_base.json').read_text(encoding='utf-8'))
    if (receipt['model_id']!='Qwen/Qwen3.5-4B' or receipt['status']!='complete'
            or receipt['revision']!='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
            or len(receipt['files'])!=2
            or not all(item['verified_official_lfs'] for item in receipt['files'])):
        raise ValueError('Qwen3.5 official frozen base receipt invalid')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='data/prepared/v15-error-repair/plan.json')
    parser.add_argument('--preflight-output',help='CPU-only frozen data/token-budget verification; no model load')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    require_eligible_trial(plan)
    objective = plan.get('repair_objective','cpo')
    if objective not in ('cpo','sft'):
        raise ValueError('Only frozen CPO or direct chosen SFT objectives allowed')
    output = Path(plan['output'])
    if output.exists():
        raise ValueError('Preserve previous optimization attempt')
    pairs, replay = read_jsonl(plan['pairs_path']), read_jsonl(plan['replay_path'])
    if fingerprint(pairs)!=plan['pairs_hash'] or fingerprint(replay)!=plan['replay_hash']:
        raise ValueError('Frozen repair/replay changed')
    base_dir=plan.get('base_dir','models/Qwen3-4B')
    is_qwen35=plan['base']['model_id']=='Qwen/Qwen3.5-4B'
    invalidated=Path('runs/qwen35-v3-clean40-preflight-invalidated.json')
    if is_qwen35 and invalidated.exists():
        invalidation=json.loads(invalidated.read_text(encoding='utf-8'))
        if fingerprint(plan)==invalidation['plan_hash']:
            raise ValueError('Preserve invalidated40-error plan; conservative negative review required a new frozen trial')
    receipt=repair_base_receipt(base_dir,is_qwen35)
    if receipt!=plan['base'] or fingerprint(SYSTEM_PROMPT)!=plan['prompt_hash']:
        raise ValueError('Base revision/prompt changed')
    for row in pairs+replay:
        validate_record(row,True,True,purpose='training')
    for row in pairs:
        if row['preference_review']['hash']!=fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')}):
            raise ValueError('Unreviewed negative')
    annotations = None
    if plan.get('critical_annotations'):
        from witrans_tools.critical_spans import validated_annotations
        report = json.loads(Path(plan['critical_annotations']).read_text(encoding='utf-8'))
        if fingerprint(report) != plan['critical_annotations_hash']:
            raise ValueError('Frozen critical annotations changed')
        annotations = validated_annotations(pairs, report)
    starting=Path(plan['starting_adapter'])
    with (starting/'adapter_model.safetensors').open('rb') as stream:
        starting_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if starting_sha!=plan['starting_adapter_sha256']:
        raise ValueError('Best starting adapter changed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False)
    def chosen_example(row):
        if annotations is None:
            return encode_example(tokenizer,row,1024)
        from witrans_tools.critical_spans import encode_critical_spans
        entry = annotations[row['id']]
        return encode_critical_spans(tokenizer,row,entry['critical_spans'],1024,entry['multiplier'])
    encoded_pairs = {r['id']:(chosen_example(r),
        encode_example(tokenizer,{**r,'output':r['rejected']},1024)) for r in pairs}
    encoded_replay = {r['id']:encode_example(tokenizer,r,1024) for r in replay}
    expected_tokens = sum(sum(len(e['input_ids']) for e in encoded_pairs[s['pair']])+
        sum(len(encoded_replay[key]['input_ids']) for key in s['replay']) for s in plan['schedule'])
    if len(plan['schedule'])!=plan['updates']*plan['accumulation']:
        raise ValueError('Frozen schedule does not match update/accumulation budget')
    if is_qwen35 and expected_tokens!=plan['full_token_budget']:
        raise ValueError('Frozen full-token budget changed before training')
    if args.preflight_output:
        dest=Path(args.preflight_output)
        if dest.exists():
            raise ValueError('Preserve previous CPU preflight')
        if is_qwen35:
            config=json.loads((starting/'adapter_config.json').read_text(encoding='utf-8'))
            if config['lora_dropout']!=plan['lora_dropout']:
                raise ValueError('Frozen adapter dropout changed')
        write_json(dest,dict(at=now(),status='CPU data/token-budget preflight passed',plan_hash=fingerprint(plan),
            full_tokens=expected_tokens,forward_calls=sum(2+len(s['replay']) for s in plan['schedule']),
            backward_calls=sum(1+len(s['replay']) for s in plan['schedule']),updates=plan['updates'],
            pairs=len(pairs),replay_rows=len(replay),starting_adapter_sha256=starting_sha,
            gpu_executed=False,optimizer_updates=0,stage_goal_complete=False))
        print(dict(preflight=str(dest),full_tokens=expected_tokens),flush=True)
        return
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed
    set_seed(plan['seed'])
    if is_qwen35:
        import psutil
        owner=json.loads(Path(plan['owner_job']).read_text(encoding='utf-8'))
        if owner['status']!='finished' or owner['exit_code']!=0:
            raise ValueError('Serial GPU predecessor must finish successfully')
        own=psutil.Process();family={own.pid,*(p.pid for p in own.parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid in family or 'python' not in (process.info['name'] or '').lower():
                continue
            command=' '.join(process.info['cmdline'] or [])
            if any(term in command for term in ('scripts.low_cpu_run','scripts.evaluate_qwen35',
                    'scripts.train_', 'scripts.probe_qwen35_', 'scripts.diagnose_qwen35_stage')):
                raise ValueError(f'Other model process remains: {process.pid}')
        metadata=json.loads((starting/'witrans_adapter.json').read_text(encoding='utf-8'))
        if (metadata['base_revision']!=plan['base']['revision'] or metadata['prompt_hash']!=plan['prompt_hash']
                or metadata['adapter_sha256']!=starting_sha):
            raise ValueError('Starting adapter identity changed')
        from witrans_tools.qwen35 import Qwen35Translator
        translator=Qwen35Translator(base_dir=base_dir)
        base=translator.model
        vocabulary=base.get_input_embeddings().weight
        # Match validated Qwen3.5 preparation without briefly allocating its
        # tied vocabulary in FP32. Keep FP32 normalization, frozen NF4 weights.
        for parameter in base.parameters():
            parameter.requires_grad=False
            if (parameter is not vocabulary and parameter.__class__.__name__!='Params4bit'
                    and parameter.dtype in (torch.float16,torch.bfloat16)):
                parameter.data=parameter.data.to(torch.float32)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        assert not vocabulary.requires_grad and vocabulary.data_ptr()==base.get_output_embeddings().weight.data_ptr()
    else:
        translator = WiTrans(base_dir,plan['starting_adapter'],max_length=1024,quantization='nf4')
        if translator.adapter_sha256!=starting_sha:
            raise ValueError('Loaded starting adapter identity changed')
        base = translator.model.unload()
        if hasattr(base,'peft_config'):
            delattr(base,'peft_config')
        base = prepare_model_for_kbit_training(base,use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant':False})
    model = PeftModel.from_pretrained(base,plan['starting_adapter'],is_trainable=True,local_files_only=True)
    model.config.use_cache = False
    if any(p.device.type!='cuda' for p in model.parameters()):
        raise ValueError('CPU/disk parameter offload forbidden')
    if is_qwen35 and model.peft_config['default'].lora_dropout!=plan['lora_dropout']:
        raise ValueError('Frozen adapter dropout changed')
    counters = {'forward_calls':0,'backward_calls':0,'full_tokens':0}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters,lr=plan['learning_rate'],weight_decay=.01)
    write_json(output/'run_config.json',{'at':now(),'plan':plan,'plan_hash':fingerprint(plan),
        'expected_full_tokens':expected_tokens,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method':plan.get('method','actual_error_cpo_with_diverse_replay'),'release_approved':False})
    def score(example):
        counters['forward_calls']+=1
        counters['full_tokens']+=len(example['input_ids'])
        tensors = {k:torch.tensor([v],device='cuda:0') for k,v in example.items()}
        labels = tensors.pop('labels')
        weights = tensors.pop('token_weights',None)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits,labels,token_weights=weights)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    beginning, history = time.perf_counter(), []
    for step in range(1,plan['updates']+1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        for visit in plan['schedule'][(step-1)*plan['accumulation']:step*plan['accumulation']]:
            chosen,rejected = encoded_pairs[visit['pair']]
            lp,nll = score(chosen)
            if objective == 'cpo':
                rp,_ = score(rejected)
                loss = cpo_loss(lp,rp,nll,beta=plan['beta'])
            else:
                # Keep rejection forward scores as a TRAIN-only diagnostic with
                # the same input/visit budget; no negative gradient enters SFT.
                with torch.no_grad():
                    rp,_ = score(rejected)
                loss = nll
            (loss*plan['pair_weight']/plan['accumulation']).backward()
            counters['backward_calls']+=1
            loss_total+=float(loss.detach())*plan['pair_weight']/plan['accumulation']
            del lp,rp,nll,loss
            for key in visit['replay']:
                _,nll = score(encoded_replay[key])
                (nll*plan['replay_weight']/len(visit['replay'])/plan['accumulation']).backward()
                counters['backward_calls']+=1
                loss_total+=float(nll.detach())*plan['replay_weight']/len(visit['replay'])/plan['accumulation']
                del nll
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient')
        if torch.cuda.max_memory_reserved()/1024**3>plan['memory_cap_gib']:
            raise ValueError('NF4 memory budget exceeded before optimizer update')
        optimizer.step()
        record = {'step':step,'loss':loss_total,'gradient_norm_before_clip':float(norm.detach()),**counters,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
        if record['peak_reserved_gib']>plan['memory_cap_gib']:
            raise ValueError('NF4 memory budget exceeded')
        history.append(record)
        write_json(output/'progress.json',{'at':now(),'history':history,'release_approved':False})
        print(record,flush=True)
    visits = plan['updates'] * plan['accumulation']
    expected_forwards = sum(2 + len(s['replay']) for s in plan['schedule'])
    expected_backwards = sum(1 + len(s['replay']) for s in plan['schedule'])
    if (len(plan['schedule']) != visits or counters['full_tokens'] != expected_tokens
            or counters['full_tokens'] != plan.get('full_token_budget', expected_tokens)
            or counters['forward_calls'] != expected_forwards or counters['backward_calls'] != expected_backwards):
        raise ValueError('Actual compute differs from frozen schedule')
    model.save_pretrained(output,safe_serialization=True)
    translator.tokenizer.save_pretrained(output)
    sha = hashlib.sha256((output/'adapter_model.safetensors').read_bytes()).hexdigest()
    write_json(output/'witrans_adapter.json',{'base_model_id':plan['base']['model_id'],'base_revision':plan['base']['revision'],
        'prompt_hash':plan['prompt_hash'],'adapter_sha256':sha,'parent_adapter_sha256':plan['starting_adapter_sha256'],
        'created_at':now(),'smoke_only':False,'plan_hash':fingerprint(plan),'recommended_quantization':'nf4',
        'training_method':plan.get('method','actual_error_cpo_with_diverse_replay'),'quality_status':'Experimental; semantic acceptance pending'})
    write_json(output/'metrics.json',{'at':now(),'plan_hash':fingerprint(plan),'history':history,**counters,
        'selected_checkpoint':plan['selected_checkpoint'],'adapter_sha256':sha,'seconds':time.perf_counter()-beginning,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'cpu_parameter_count':sum(p.numel() for p in model.parameters() if p.device.type=='cpu'),'release_approved':False})


if __name__=='__main__':
    main()
