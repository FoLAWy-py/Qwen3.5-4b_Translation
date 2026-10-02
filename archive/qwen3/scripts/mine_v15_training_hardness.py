"""Rank reviewed TRAIN-only examples; selection is triage, never an automatic grade."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from archive.qwen3.runtime import SYSTEM_PROMPT
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl,append_jsonl
from witrans_tools.data import encode_example,validate_record


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    destination = Path('runs/v15-training-hardness')
    if destination.exists():
        raise ValueError('Preserve mining attempt and partial scores')
    for path in ('runs/v14-matching-evaluation-job.json','runs/v14-parent-fact-diagnostics-job.json'):
        job = load(path)
        if job['status']!='finished' or job.get('exit_code')!=0:
            raise ValueError('Owned GPU predecessor has not finished successfully')
    prior = load('data/prepared/v13-public-optimization/plan.json')
    spec = prior['datasets']['mixed']
    records = read_jsonl(spec['path'])
    if fingerprint(records)!=spec['hash'] or fingerprint(SYSTEM_PROMPT)!=prior['prompt_hash']:
        raise ValueError('Frozen authorized TRAIN pool or prompt changed')
    heldout = []
    for split in ('dev','public_dev'):
        rows = read_jsonl(prior[split]['path'])
        if fingerprint(rows)!=prior[split]['hash']:
            raise ValueError('Heldout source snapshot changed')
        heldout.extend(rows)
    heldout_groups = {r['group_id'] for r in heldout}
    heldout_sources = {r['input']['text'].strip().casefold() for r in heldout}
    eligible_records,excluded = [],[]
    for row in records:
        validate_record(row,True,True,purpose='training')
        if row['group_id'] in heldout_groups or row['input']['text'].strip().casefold() in heldout_sources:
            excluded.append({'id':row['id'],'reason':'Known/public heldout source or group overlap; triage excluded'})
        else:
            eligible_records.append(row)
    start = Path(prior['starting_adapter'])
    if hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()!=prior['starting_adapter_sha256']:
        raise ValueError('Best parent weights changed')
    destination.mkdir(parents=True)
    state = {'at':now(),'phase':'loading_parent','train_hash':spec['hash'],
             'parent_adapter_sha256':prior['starting_adapter_sha256'],'scored':0,'total':len(eligible_records),
             'heldout_source_exclusions':excluded,'release_approved':False}
    write_json(destination/'progress.json',state)
    try:
        import torch
        from archive.qwen3.runtime import WiTrans
        from witrans_tools.preference import answer_scores
        translator = WiTrans('models/Qwen3-4B',str(start),max_length=1024,quantization='nf4')
        translator.model.eval()
        torch.cuda.reset_peak_memory_stats()
        beginning = time.perf_counter()
        scores = []
        with torch.inference_mode():
            for index,row in enumerate(eligible_records,1):
                encoded = encode_example(translator.tokenizer,row,1024)
                answer_tokens = sum(value!=-100 for value in encoded['labels'])
                prompt_tokens = len(encoded['input_ids'])-answer_tokens
                tensors = {key:torch.tensor([value],device='cuda:0') for key,value in encoded.items()}
                labels = tensors.pop('labels')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,nll = answer_scores(translator.model(**tensors).logits,labels)
                if not torch.isfinite(nll):
                    raise ValueError('Nonfinite training triage score')
                score = {'id':row['id'],'group_id':row['group_id'],'category':row['category'],
                         'target_lang':row['input']['target_lang'],'record_hash':fingerprint(row),
                         'answer_eos_nll':float(nll),'answer_tokens':answer_tokens,'prompt_tokens':prompt_tokens,
                         'full_tokens':len(encoded['input_ids']),
                         'short_decode_eligible':answer_tokens<=80 and prompt_tokens+256<=1024}
                scores.append(score)
                append_jsonl(destination/'scores.jsonl',score)
                if index%64==0 or index==len(eligible_records):
                    state.update(phase='scoring_authorized_training',scored=index,updated_at=now(),
                                 peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                    write_json(destination/'progress.json',state)
                    print({'scored':index,'total':len(eligible_records),'peak_reserved_gib':state['peak_reserved_gib']},flush=True)
        from witrans_tools.training_triage import select_training_triage
        selected = select_training_triage(scores)
        groups = {row['group_id'] for row in selected}
        indexed = {row['id']:row for row in eligible_records}
        selected_records = [indexed[score['id']] for score in selected]
        write_jsonl(destination/'selected-records.jsonl',selected_records)
        write_jsonl(destination/'selected-scores.jsonl',selected)
        report = {'at':now(),'parent_adapter_sha256':prior['starting_adapter_sha256'],
                  'train_hash':spec['hash'],'scored':len(scores),'selected':64,'selected_groups':len(groups),
                  'selected_hash':fingerprint(selected_records),'selected_score_hash':fingerprint(selected),
                  'selected_strata':dict(Counter(f"{r['category']}/{r['target_lang']}" for r in selected)),
                  'selected_public_rows':sum(r['id'].startswith('public-short-') for r in selected),
                  'protocol':'Rank highest TRAIN answer/EOS NLL,6 per category/direction,one record per source group,then fill to64; answer<=80 tokens and reserve256 decode tokens. No DEV output/target mining.',
                  'scope':'TF likelihood triage only. High NLL may mean wording/label ambiguity, not a model error. Individually read sources, contexts, references and actual generations before constructing any new training contrast.',
                  'seconds':time.perf_counter()-beginning,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                  'new_contrasts_training_allowed':False,'release_approved':False}
        write_json(destination/'summary.json',report)
        state.update(phase='individual_source_review_and_decoding_pending',updated_at=now(),summary=str(destination/'summary.json'))
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',updated_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(destination/'progress.json',state)


if __name__=='__main__':
    main()
