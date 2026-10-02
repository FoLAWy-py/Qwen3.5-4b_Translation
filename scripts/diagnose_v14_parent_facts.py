"""Training-only gradient-strength audit after the owned GPU decoding job exits."""
import hashlib
import json
import math
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    state_path = Path('runs/v14-parent-fact-diagnostics-progress.json')
    summary_path = Path('runs/v14-parent-fact-diagnostics.summary.json')
    if state_path.exists() or summary_path.exists():
        raise ValueError('Preserve diagnostic evidence')
    job_path = Path('runs/v14-matching-evaluation-job.json')
    job = load(job_path)
    if ('scripts.run_v14_evaluation' not in job['command'] or Path(job['cwd']).resolve()!=Path.cwd().resolve()):
        raise ValueError('Unexpected evaluation owner')
    live = psutil.Process(job['process_pid']) if job['status']=='running' else None
    if live is not None and live.cmdline()!=job['command']:
        raise ValueError('Evaluation PID is not owned')
    state = {'at':now(),'phase':'waiting_for_owned_decoding','pid':job.get('process_pid'),
             'created':live.create_time() if live is not None else None,'release_approved':False}
    write_json(state_path,state)
    try:
        while job['status'] in ('launching','running'):
            if live is not None and live.is_running() and live.create_time()!=state['created']:
                raise ValueError('Evaluation PID reused')
            time.sleep(5)
            job = load(job_path)
        if job['status']!='finished' or job.get('exit_code')!=0:
            raise ValueError('Owned decoding did not finish successfully')
        state.update(phase='auditing_training_only_scores',updated_at=now())
        write_json(state_path,state)
        plan = load('data/prepared/v14-fact-trial/plan.json')
        pairs = read_jsonl(plan['pairs_path'])
        if fingerprint(pairs)!=plan['pairs_hash']:
            raise ValueError('Training fact snapshot changed')
        start = Path(plan['starting_adapter'])
        if hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()!=plan['starting_adapter_sha256']:
            raise ValueError('Best parent weights changed')
        import torch
        from witrans import WiTrans
        from witrans_tools.fact_pairs import encode_fact_packet
        from witrans_tools.preference import answer_scores
        translator = WiTrans('models/Qwen3-4B',str(start),max_length=1024,quantization='nf4')
        model = translator.model
        model.eval()
        rows = []
        torch.cuda.reset_peak_memory_stats()
        beginning = time.perf_counter()
        with torch.inference_mode():
            for pair in pairs:
                matrix,counts = encode_fact_packet(translator.tokenizer,pair,1024)
                scores = []
                for i in range(2):
                    row = []
                    for j in range(2):
                        tensors = {key:torch.tensor([value],device='cuda:0') for key,value in matrix[i][j].items()}
                        labels = tensors.pop('labels')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logp,_ = answer_scores(model(**tensors).logits,labels)
                        row.append(float(logp)/counts[j])
                    scores.append(row)
                margin = scores[0][0]+scores[1][1]-scores[0][1]-scores[1][0]
                negative_probability = float(torch.sigmoid(torch.tensor(-margin/plan['temperature'],dtype=torch.float64)))
                # Derivatives wrt summed logps: length factors cancel from this ratio.
                relative_norm = 2*math.sqrt(2)*plan['matching_weight']/plan['temperature']*negative_probability
                row = {'id':pair['id'],'group_id':pair['group_id'],'fact_axis':pair['fact_axis'],
                       'target_lang':pair['inputs'][0]['target_lang'],'scores':scores,'answer_token_counts':counts,
                       'assignment_margin':margin,'source_margins':[scores[0][0]-scores[0][1],scores[1][1]-scores[1][0]],
                       'matching_to_sft_logp_gradient_norm_ratio':relative_norm,
                       'both_diagonal_answers_outrank_cross_answers':scores[0][0]>scores[0][1] and scores[1][1]>scores[1][0]}
                rows.append(row)
                print({'id':row['id'],'margin':margin,'matching_sft_gradient_ratio':relative_norm},flush=True)
        torch.cuda.synchronize()
        write_jsonl('runs/v14-parent-fact-diagnostics.jsonl',rows)
        report = {'at':now(),'plan_hash':fingerprint(plan),'parent_adapter_sha256':plan['starting_adapter_sha256'],
                  'rows':len(rows),'families':len({row['group_id'] for row in rows}),
                  'ratio_below_one_percent':sum(row['matching_to_sft_logp_gradient_norm_ratio']<.01 for row in rows),
                  'ratio_median':sorted(row['matching_to_sft_logp_gradient_norm_ratio'] for row in rows)[len(rows)//2],
                  'one_or_more_cross_answers_not_lower':sum(not row['both_diagonal_answers_outrank_cross_answers'] for row in rows),
                  'seconds':time.perf_counter()-beginning,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
                  'scope':'Teacher-forced TRAIN-only scores, dropout disabled; gradient diagnostic, not decoded semantic accuracy or held-out efficacy.',
                  'threshold':'Below0.01 means matching-component score-gradient norm less than1% of pair SFT; diagnostic only.',
                  'release_approved':False}
        write_json(summary_path,report)
        state.update(phase='diagnostic_complete',updated_at=now(),summary=str(summary_path))
        print(report,flush=True)
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',updated_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(state_path,state)


if __name__=='__main__':
    main()
