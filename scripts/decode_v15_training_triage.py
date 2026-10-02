"""Bind completed TRAIN score mining, deduplicate families, decode for human audit."""
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.training_triage import select_training_triage


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    state_path = Path('runs/v15-training-triage-decoding-progress.json')
    if state_path.exists():
        raise ValueError('Preserve decoding attempt')
    job_path = Path('runs/v15-training-hardness-job.json')
    job = load(job_path)
    if ('scripts.mine_v15_training_hardness' not in job['command'] or Path(job['cwd']).resolve()!=Path.cwd().resolve()):
        raise ValueError('Unexpected training mining owner')
    live = psutil.Process(job['process_pid']) if job['status']=='running' else None
    if live is not None and live.cmdline()!=job['command']:
        raise ValueError('Mining PID command changed')
    state = {'at':now(),'phase':'waiting_for_owned_training_mining','pid':job.get('process_pid'),
             'created':live.create_time() if live is not None else None,'release_approved':False}
    write_json(state_path,state)
    try:
        while job['status'] in ('launching','running'):
            if live is not None and live.is_running() and live.create_time()!=state['created']:
                raise ValueError('Mining PID reused')
            time.sleep(5)
            job = load(job_path)
        if job['status']!='finished' or job.get('exit_code')!=0:
            raise ValueError('Mining did not complete successfully')
        prior = load('data/prepared/v13-public-optimization/plan.json')
        pool = read_jsonl(prior['datasets']['mixed']['path'])
        if fingerprint(pool)!=prior['datasets']['mixed']['hash']:
            raise ValueError('Authorized TRAIN sources changed')
        indexed = {row['id']:row for row in pool}
        scores = read_jsonl('runs/v15-training-hardness/scores.jsonl')
        score_progress = load('runs/v15-training-hardness/progress.json')
        if len(scores)!=score_progress['total'] or len({row['id'] for row in scores})!=len(scores):
            raise ValueError('Complete distinct TRAIN scores required')
        for score in scores:
            record = indexed[score['id']]
            if (score['record_hash']!=fingerprint(record) or score['group_id']!=record['group_id']
                    or score['category']!=record['category'] or score['target_lang']!=record['input']['target_lang']):
                raise ValueError('Mining score no longer binds source record')
        # Recalculate from completed scores even if an older worker selected variants.
        # Never overwrite the original mining selection or its provenance.
        selected = select_training_triage(scores)
        records = [indexed[score['id']] for score in selected]
        input_path = Path('runs/v15-training-hardness/review-selected-records.jsonl')
        output_path = Path('runs/v15-training-triage-generations.jsonl')
        if input_path.exists() or output_path.exists():
            raise ValueError('Preserve existing audit inputs and generations')
        write_jsonl(input_path,records)
        write_jsonl('runs/v15-training-hardness/review-selected-scores.jsonl',selected)
        plan = {'at':now(),'data_hash':fingerprint(records),'source_score_hash':fingerprint(scores),
                'records':64,'source_groups':len({row['group_id'] for row in records}),
                'input':str(input_path),'output':str(output_path),'adapter':prior['starting_adapter'],
                'adapter_sha256':prior['starting_adapter_sha256'],'base':prior['base'],
                'prompt_hash':prior['prompt_hash'],'quantization':'nf4','max_length':1024,'max_new_tokens':256,
                'scope':'TRAIN-only error and label audit; never a heldout model-quality comparison. All changed translations need individual source-based reading. No new contrasts authorized by NLL alone.',
                'new_contrasts_training_allowed':False,'release_approved':False}
        write_json('runs/v15-training-triage-decoding-plan.json',plan)
        state.update(phase='decoding_training_only_sources',updated_at=now(),plan='runs/v15-training-triage-decoding-plan.json')
        write_json(state_path,state)
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                        '--adapter-dir',prior['starting_adapter'],'--input',str(input_path),
                        '--output',str(output_path),'--quantization','nf4','--max-length','1024',
                        '--max-new-tokens','256'],check=True)
        summary = load(output_path.with_suffix('.summary.json'))
        if (summary['data_hash']!=plan['data_hash'] or summary['adapter_sha256']!=plan['adapter_sha256']
                or summary['prompt_hash']!=plan['prompt_hash'] or summary['base']!=plan['base']
                or summary['quantization']!='nf4'):
            raise ValueError('Training-source decoding protocol changed')
        state.update(phase='individual_training_source_and_output_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',updated_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(state_path,state)


if __name__=='__main__':
    main()
