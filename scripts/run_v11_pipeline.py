"""One sequential local GPU workflow; stop before semantic acceptance."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-benchmark-pid',type=int,required=True)
    args = parser.parse_args()
    state_path = Path('runs/v11-pipeline-progress.json')
    if state_path.exists():
        raise ValueError('Preserve existing pipeline evidence')
    summary_path = Path('runs/v7-short-compiled-eager-rounding.summary.json')
    state = {'at':now(),'phase':'waiting_for_existing_benchmark',
        'benchmark_pid':args.wait_benchmark_pid,'release_approved':False}
    def stage(phase):
        state.update(phase=phase,updated_at=now())
        write_json(state_path,state)
        print({'phase':phase,'at':now()},flush=True)
    try:
        try:
            process = psutil.Process(args.wait_benchmark_pid)
            if ('scripts.benchmark_short' not in process.cmdline()
                    or Path(process.cwd()).resolve()!=Path.cwd().resolve()):
                raise ValueError('Wait target is not the confirmed project benchmark')
            state['benchmark_created_at'] = process.create_time()
            stage('waiting_for_existing_benchmark')
            while process.is_running():
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    continue
                break
        except psutil.NoSuchProcess:
            if not summary_path.exists():
                raise ValueError('Benchmark stopped without complete evidence; do not restart it here')
        if not summary_path.exists():
            raise ValueError('Benchmark stopped without complete summary')
        # The comparison verifies all81 records, frozen inputs, prompt, weight,
        # normal JSON/EOS and the same quantization/generation protocol.
        stage('comparing_complete_runtime')
        subprocess.run([sys.executable,'-u','-m','scripts.compare_short_runtime',
            '--compiled','runs/v7-short-compiled-eager-rounding.jsonl',
            '--output','runs/v7-short-eager-rounding-comparison.json'],check=True)
        stage('training_v11')
        subprocess.run([sys.executable,'-u','-m','scripts.low_cpu_run','--module','scripts.train_v4',
            '--data-dir','data/prepared/v11-mixed','--output','models/witrans-4b-v11-sft',
            '--selection-output','runs/v11-selection.json'],check=True)
        stage('verifying_training_and_freezing_dev_protocol')
        subprocess.run([sys.executable,'-u','-m','scripts.prepare_v8_evaluation','--version','v11'],check=True)
        stage('generating_v11_known_dev')
        subprocess.run([sys.executable,'-u','-m','scripts.evaluate_v6',
            '--role','sft','--plan','runs/v11-evaluation-plan.json'],check=True)
        stage('generating_original_base_known_dev')
        plan = json.loads(Path('runs/v11-base-dev-plan.json').read_text(encoding='utf-8'))
        if fingerprint(read_jsonl(plan['input']))!=plan['development_hash']:
            raise ValueError('Frozen original-base DEV changed')
        subprocess.run([sys.executable,'-u','-m','scripts.low_cpu_run','evaluate','--baseline',
            '--quantization','nf4','--input',plan['input'],'--output',plan['output'],
            '--max-length','1024','--max-new-tokens','256'],check=True)
        stage('awaiting_individual_semantic_review')
    except Exception as exc:
        state.update(error_type=type(exc).__name__,error=str(exc))
        stage('stopped_preserving_evidence')
        raise


if __name__=='__main__':
    main()
