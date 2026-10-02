"""One owned GPU pipeline for the explicitly requested new-model comparison."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint, now, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',default='')
    parser.add_argument('--download-job',default='runs/qwen35-range-download-job.json')
    parser.add_argument('--data-version',default='v1',choices=('v1','v2'))
    args=parser.parse_args()
    state_path=Path(f"runs/qwen35-comparison{'-'+args.attempt if args.attempt else ''}-progress.json")
    assert not state_path.exists(), 'Preserve existing attempt'
    state=dict(at=now(),phase='waiting_for_download_and_owned_gpu_predecessor',
        predecessor='runs/v19-balanced-repair-job.json',download=args.download_job,
        completed=[],release_approved=False)
    write_json(state_path,state)
    def run(module,*args):
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',module,*args],check=True)
    try:
        observed={}
        while True:
            ready=True
            for path in (state['predecessor'],state['download']):
                job=load(path)
                if job['status'] in ('launching','running'):
                    ready=False
                    if job['status']=='running':
                        try:
                            process=psutil.Process(job['process_pid'])
                            assert process.cmdline()==job['command'], 'Owned process command mismatch'
                            if path in observed:
                                assert process.create_time()==observed[path], 'PID was reused'
                            observed[path]=process.create_time()
                        except psutil.NoSuchProcess:
                            pass  # Supervisor may be writing the terminal receipt.
                else:
                    assert job['status']=='finished' and job['exit_code']==0, f'Predecessor failed: {path}'
            if ready:
                break
            time.sleep(5)
        state.update(phase='freezing_data_recipe_and_token_budgets',updated_at=now())
        write_json(state_path,state)
        run('scripts.prepare_qwen35_comparison','--version',args.data_version)
        plan_path=f'data/prepared/qwen35-{args.data_version}/plan.json'
        plan=load(plan_path)
        state['plan_hash']=fingerprint(plan)
        steps=[('preflight',lambda:run('scripts.train_qwen35_recipe','--phase','preflight','--plan',plan_path,
            '--preflight-tag',args.data_version))]
        for split,spec in plan['datasets'].items():
            steps.append((f'base_{split}',lambda split=split,spec=spec:run('scripts.evaluate_qwen35',
                '--input',spec['path'],'--output',f'runs/qwen35-base-{split}.jsonl')))
        steps.append(('sft_training',lambda:run('scripts.train_qwen35_recipe','--phase','sft','--plan',plan_path)))
        steps.append(('critical_cpo_training',lambda:run('scripts.train_qwen35_recipe','--phase','cpo','--plan',plan_path)))
        for split,spec in plan['datasets'].items():
            steps.append((f'finetuned_{split}',lambda split=split,spec=spec:run('scripts.evaluate_qwen35',
                '--adapter-dir',plan['cpo']['output'],'--input',spec['path'],
                '--output',f'runs/qwen35-finetuned-{split}.jsonl')))
        for label,operation in steps:
            state.update(phase=label,updated_at=now())
            write_json(state_path,state)
            print({'phase':label},flush=True)
            operation()
            state['completed'].append(label)
        state.update(phase='individual_paired_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error_type=type(exc).__name__,error=str(exc),updated_at=now())
        raise
    finally:
        write_json(state_path,state)


if __name__=='__main__':
    main()
