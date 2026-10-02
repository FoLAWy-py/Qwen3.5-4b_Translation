"""Wait for the owned fact training job; freeze fixed-step candidate and decode."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint,now,read_jsonl,write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_fact_selection(plan,mode):
    output = Path(plan['arms'][mode])
    config,metrics = load(output/'run_config.json'),load(output/'metrics.json')
    if (config['plan_hash']!=fingerprint(plan) or metrics['plan_hash']!=fingerprint(plan)
            or config['mode']!=mode or metrics['mode']!=mode
            or metrics['forward_calls']!=plan['expected_training_forward_calls']
            or metrics['backward_calls']!=plan['expected_training_backward_calls']
            or metrics['full_tokens']!=config['expected_full_tokens_including_recomputation']
            or [row['step'] for row in metrics['history']]!=list(range(1,plan['updates']+1))
            or metrics['peak_reserved_gib']>6.5
            or any(row['forward_calls']!=40*row['step'] or row['backward_calls']!=24*row['step'] for row in metrics['history'])
            or any(metrics['history'][-1][key]!=metrics[key] for key in ('forward_calls','backward_calls','full_tokens'))):
        raise ValueError('Training schedule, counters or memory differ from frozen trial')
    selected = metrics['selected']
    if selected['step']!=plan['selected_checkpoint'] or selected['step']!=plan['updates']:
        raise ValueError('Fixed final checkpoint required; DEV selection prohibited')
    if [row['step'] for row in metrics['checkpoints']]!=plan['checkpoints']:
        raise ValueError('Missing frozen checkpoint')
    for checkpoint in metrics['checkpoints']:
        directory = Path(checkpoint['directory'])
        if directory.resolve()!=(output/f"checkpoint-{checkpoint['step']}").resolve():
            raise ValueError('Checkpoint points outside frozen output')
        digest = hashlib.sha256((directory/'adapter_model.safetensors').read_bytes()).hexdigest()
        metadata = load(directory/'witrans_adapter.json')
        if (digest!=checkpoint['adapter_sha256'] or metadata['adapter_sha256']!=digest
                or metadata['parent_adapter_sha256']!=plan['starting_adapter_sha256']
                or metadata['base_revision']!=plan['base']['revision'] or metadata['prompt_hash']!=plan['prompt_hash']):
            raise ValueError('Checkpoint weights or lineage changed')
    if selected!=next(row for row in metrics['checkpoints'] if row['step']==plan['updates']):
        raise ValueError('Selection does not bind saved final checkpoint')
    return selected


def main():
    state_path = Path('runs/v14-matching-evaluation-progress.json')
    if state_path.exists():
        raise ValueError('Preserve existing evaluation attempt')
    job_path = Path('runs/v14-matching-job.json')
    job = load(job_path)
    command = job['command']
    if ('scripts.train_v14_fact_trial' not in command or command[-2:]!=['--mode','matching']
            or Path(job['cwd']).resolve()!=Path.cwd().resolve()):
        raise ValueError('Unexpected training owner')
    live = None
    if job['status']=='running':
        live = psutil.Process(job['process_pid'])
        if live.cmdline()!=command:
            raise ValueError('Training PID command no longer belongs to this job')
    state = {'at':now(),'phase':'waiting_for_owned_training','training_job':str(job_path),
             'training_pid':job.get('process_pid'),'training_created':live.create_time() if live else None,
             'completed_generations':[],'release_approved':False}
    write_json(state_path,state)
    try:
        while job['status'] in ('launching','running'):
            if live is not None and live.is_running() and live.create_time()!=state['training_created']:
                raise ValueError('Training PID was reused')
            time.sleep(5)
            job = load(job_path)
        if job['status']!='finished' or job.get('exit_code')!=0:
            raise ValueError('Owned training did not finish successfully')
        plan = load('data/prepared/v14-fact-trial/plan.json')
        selected = validate_fact_selection(plan,'matching')
        write_json('runs/v14-fixed-selection.json',{'at':now(),'training_plan_hash':fingerprint(plan),
                   'selected':selected,'selection':'Predeclared final checkpoint16, no DEV checkpoint selection.',
                   'release_approved':False})
        old = load('runs/v13-evaluation-plan.json')
        datasets = old['datasets']
        for spec in datasets.values():
            if fingerprint(read_jsonl(spec['path']))!=spec['hash']:
                raise ValueError('Frozen evaluation sources changed')
        control = next(target for target in old['targets'] if target['role']=='v7' and target['split']=='public')
        targets = [control]+[{'role':'v14-fact-matching','split':split,'baseline':False,
                              'adapter':selected['directory'],'adapter_sha256':selected['adapter_sha256'],
                              'output':f'runs/v14-matching-{split}.jsonl'} for split in datasets]
        if any(Path(target['output']).exists() for target in targets[1:]):
            raise ValueError('Preserve existing candidate generations')
        evaluation = {'at':now(),'training_plan_hash':fingerprint(plan),'datasets':datasets,
                      'semantic_cache_roles':['v7'],'base':plan['base'],'prompt_hash':plan['prompt_hash'],
                      'targets':targets,'quantization':'nf4','max_length':1024,'max_new_tokens':256,
                      'policy':'Fixed checkpoint16 fact pilot versus best v7; auxiliary DEV only; no new original-base comparison or release claim.'}
        write_json('runs/v14-evaluation-plan.json',evaluation)
        for target in targets[1:]:
            state.update(phase='generating_candidate',current=target['split'],updated_at=now())
            write_json(state_path,state)
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                            '--adapter-dir',target['adapter'],'--input',datasets[target['split']]['path'],
                            '--output',target['output'],'--quantization','nf4','--max-length','1024',
                            '--max-new-tokens','256'],check=True)
            summary = load(Path(target['output']).with_suffix('.summary.json'))
            if (summary['adapter_sha256']!=target['adapter_sha256'] or summary['prompt_hash']!=plan['prompt_hash']
                    or summary['data_hash']!=datasets[target['split']]['hash'] or summary['base']!=plan['base']):
                raise ValueError('Decoded candidate differs from frozen protocol')
            state['completed_generations'].append(target['split'])
        state.update(phase='individual_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',updated_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(state_path,state)


if __name__=='__main__':
    main()
