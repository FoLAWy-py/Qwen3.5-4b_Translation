"""Transfer the validated SFT/critical-CPO recipe to a fresh Qwen3.5 adapter."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import encode_example, validate_record
from witrans_tools.critical_spans import encode_critical_spans, validated_annotations
from witrans_tools.preference import answer_scores,cpo_loss
from witrans_tools.qwen35 import Qwen35Translator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('preflight','sft','cpo'))
    parser.add_argument('--plan',default='data/prepared/qwen35-v1/plan.json')
    parser.add_argument('--preflight-tag',default='')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig,get_peft_model,PeftModel
    from transformers import set_seed
    phase='sft' if args.phase=='preflight' else args.phase
    spec=plan[phase]
    output=Path('runs/qwen35-preflight'+('-'+args.preflight_tag if args.preflight_tag else '')) if args.phase=='preflight' else Path(spec['output'])
    assert not output.exists(), 'Preserve training evidence'
    rows=read_jsonl(spec['path'])
    assert fingerprint(rows)==spec['hash'] and fingerprint(SYSTEM_PROMPT)==plan['prompt_hash']
    for row in rows:
        validate_record(row,True,True,purpose='training')
    annotations={}
    if phase=='cpo':
        report=json.loads(Path(spec['annotations']).read_text(encoding='utf-8'))
        assert fingerprint(report)==spec['annotations_hash']
        annotations=validated_annotations(rows,report)
        for row in rows:
            assert row['preference_review']['hash']==fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')})
    set_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats()
    translator=Qwen35Translator()
    model=translator.model
    vocabulary=model.get_input_embeddings().weight
    # Equivalent freezing/checkpoint preparation, explicitly excluding the
    # frozen tied vocabulary from PEFT's blanket FP32 conversion. Avoid even
    # a temporary FP32 vocabulary allocation; preserve FP32 normalization.
    for parameter in model.parameters():
        parameter.requires_grad=False
        if (parameter is not vocabulary and parameter.__class__.__name__!='Params4bit'
                and parameter.dtype in (torch.float16,torch.bfloat16)):
            parameter.data=parameter.data.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert not model.get_input_embeddings().weight.requires_grad
    assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    parent_sha=None
    if phase=='cpo':
        parent=Path(plan['sft']['output'])
        metadata=json.loads((parent/'witrans_adapter.json').read_text(encoding='utf-8'))
        with (parent/'adapter_model.safetensors').open('rb') as stream:
            parent_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        assert parent_sha==metadata['adapter_sha256'] and metadata['plan_hash']==fingerprint(plan)
        model=PeftModel.from_pretrained(model,parent,is_trainable=True,local_files_only=True)
    else:
        model=get_peft_model(model,LoraConfig(**plan['lora'],bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    def chosen(row):
        if row['id'] in annotations:
            entry=annotations[row['id']]
            return encode_critical_spans(translator.tokenizer,row,entry['critical_spans'],1024,entry['multiplier'])
        return encode_example(translator.tokenizer,row,1024)
    encoded={r['id']:(chosen(r),encode_example(translator.tokenizer,{**r,'output':r['rejected']},1024))
        if phase=='cpo' else (chosen(r),) for r in rows}
    expected_tokens=sum(sum(len(e['input_ids']) for e in encoded[k]) for k in spec['order'])
    assert expected_tokens==spec['full_tokens']
    parameters=[p for p in model.parameters() if p.requires_grad]
    matched_modules=[name for name,module in model.named_modules() if hasattr(module,'lora_A')]
    write_json(output/'run_config.json',dict(at=now(),plan_hash=fingerprint(plan),phase=args.phase,
        trainable_parameters=sum(p.numel() for p in parameters),matched_modules=matched_modules,
        loading_info=translator.loading_info,frozen_vocabulary_dtype='bfloat16',norm_dtype='float32',
        budget=spec,release_approved=False))
    optimizer=bnb.optim.AdamW8bit(parameters,lr=spec['learning_rate'])
    model.train()
    forwards=backwards=tokens=0
    history=[]
    beginning=time.perf_counter()
    steps=1 if args.phase=='preflight' else spec['updates']
    accumulation=1 if args.phase=='preflight' else spec['accumulation']
    def score(example):
        nonlocal forwards,tokens
        forwards+=1;tokens+=len(example['input_ids'])
        tensors={k:torch.tensor([v],device='cuda:0') for k,v in example.items()}
        labels=tensors.pop('labels');weights=tensors.pop('token_weights',None)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits,labels,weights)
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        total=0.
        batch=spec['order'][(step-1)*accumulation:step*accumulation]
        if args.phase=='preflight':
            batch=[max(encoded,key=lambda k:len(encoded[k][0]['input_ids']))]
        for key in batch:
            examples=encoded[key]
            lp,nll=score(examples[0])
            if phase=='cpo':
                rp,_=score(examples[1]);loss=cpo_loss(lp,rp,nll,beta=spec['beta'])
            else:
                loss=nll
            (loss/accumulation).backward();backwards+=1
            total+=float(loss.detach())/accumulation
            del lp,nll,loss
            if phase=='cpo': del rp
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        assert torch.isfinite(norm), 'Nonfinite gradient'
        optimizer.step()
        reserved=torch.cuda.max_memory_reserved()/1024**3
        item=dict(step=step,loss=total,gradient_norm_before_clip=float(norm),
            forward_calls=forwards,backward_calls=backwards,full_tokens=tokens,peak_reserved_gib=reserved)
        history.append(item)
        write_json(output/'progress.json',dict(at=now(),history=history,release_approved=False))
        print(item,flush=True)
        if reserved>plan['memory_cap_gib']:
            raise ValueError('Frozen memory ceiling exceeded; preserve attempt')
    if args.phase!='preflight':
        assert tokens==expected_tokens and backwards==spec['updates']*spec['accumulation']
        assert forwards==backwards*(2 if phase=='cpo' else 1)
        model.save_pretrained(output,safe_serialization=True)
        translator.tokenizer.save_pretrained(output)
        with (output/'adapter_model.safetensors').open('rb') as stream:
            sha=hashlib.file_digest(stream,'sha256').hexdigest()
        write_json(output/'witrans_adapter.json',dict(base_model_id=plan['model_id'],base_revision=plan['base']['revision'],
            prompt_hash=plan['prompt_hash'],adapter_sha256=sha,parent_adapter_sha256=parent_sha,
            created_at=now(),smoke_only=False,plan_hash=fingerprint(plan),recommended_quantization='nf4',
            training_method=plan['method'],quality_status='Experimental; matched semantic comparison pending'))
    write_json(output/'metrics.json',dict(at=now(),phase=args.phase,plan_hash=fingerprint(plan),history=history,
        seconds=time.perf_counter()-beginning,full_tokens=tokens,forward_calls=forwards,backward_calls=backwards,
        cpu_parameter_count=sum(p.numel() for p in model.parameters() if p.device.type=='cpu'),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,release_approved=False))


if __name__=='__main__':
    main()
