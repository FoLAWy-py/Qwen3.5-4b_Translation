"""Wait for the verified training process, then freeze and generate all matched arms."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from witrans_tools.common import fingerprint, now, read_jsonl, write_json

TRAIN_PLAN = Path('data/prepared/v12-factorial-v2/plan.json')
TRAIN_JOB = Path('runs/v12-factorial-job-2.json')
EVAL_PLAN = Path('runs/v12-factorial-evaluation-plan.json')
PROGRESS = Path('runs/v12-factorial-evaluation-progress.json')


def validate_selection(plan, arm):
    selection = json.loads(Path(arm['selection_output']).read_text(encoding='utf-8'))
    metrics = json.loads((Path(arm['output'])/'metrics.json').read_text(encoding='utf-8'))
    config = json.loads((Path(arm['output'])/'run_config.json').read_text(encoding='utf-8'))
    if (selection['plan_hash'] != fingerprint(plan) or config['plan_hash'] != fingerprint(plan)
            or selection['arm'] != arm['name'] or config['arm'] != arm
            or metrics['actual_training_input_tokens'] != plan['training_input_tokens_per_arm']
            or len(metrics['history']) != plan['updates']
            or metrics['history'][-1]['full_tokens'] != plan['training_input_tokens_per_arm']):
        raise ValueError('Training evidence differs from frozen arm')
    candidates = selection['candidates']
    if sorted(c['step'] for c in candidates) != plan['checkpoints']:
        raise ValueError('Missing planned checkpoint')
    expected = min(candidates, key=lambda row: row['nll']['balanced_mean'])
    if selection['selected'] != expected or metrics['selected'] != expected:
        raise ValueError('Checkpoint selection differs from frozen NLL rule')
    for candidate in candidates:
        scores = candidate['nll']
        if abs(scores['balanced_mean'] - (scores['dev']+scores['public_dev'])/2) > 1e-10:
            raise ValueError('Development weighting changed')
        path = Path(candidate['directory'])
        digest = hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()
        metadata = json.loads((path/'witrans_adapter.json').read_text(encoding='utf-8'))
        if (digest != candidate['adapter_sha256'] or metadata['adapter_sha256'] != digest
                or metadata['base_revision'] != plan['base']['revision']
                or metadata['prompt_hash'] != plan['prompt_hash']):
            raise ValueError('Frozen trained weights changed')
    return expected


def await_training(state):
    # Require an actual handle now; a state file alone cannot establish a live wait.
    job = json.loads(TRAIN_JOB.read_text(encoding='utf-8'))
    if job['status'] == 'running':
        process = psutil.Process(job['process_pid'])
        creation = process.create_time()
        if 'scripts.run_v12_factorial' not in process.cmdline():
            raise ValueError('Training PID does not name expected pipeline')
        state.update(phase='waiting_for_live_training', training_pid=process.pid,
                     training_process_created=creation)
        write_json(PROGRESS, state)
        while True:
            job = json.loads(TRAIN_JOB.read_text(encoding='utf-8'))
            if job['status'] != 'running':
                break
            if not process.is_running() or process.create_time() != creation:
                # Allow the supervisor to persist the terminal record, never relaunch.
                for _ in range(5):
                    time.sleep(1)
                    job = json.loads(TRAIN_JOB.read_text(encoding='utf-8'))
                    if job['status'] != 'running':
                        break
                if job['status'] == 'running':
                    raise ValueError('Training handle vanished without terminal evidence')
                break
            time.sleep(10)
    if job['status'] != 'finished' or job.get('exit_code') != 0:
        raise ValueError('Training did not complete successfully; preserve evidence')
    progress = json.loads(Path('runs/v12-factorial-progress.json').read_text(encoding='utf-8'))
    if progress['phase'] != 'training_complete_semantic_evaluation_pending':
        raise ValueError('Not all four arms finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if PROGRESS.exists() or EVAL_PLAN.exists():
        raise ValueError('Preserve existing evaluation evidence; no automatic restart')
    state = {'at': now(), 'phase': 'checking_training', 'completed': [], 'release_approved': False}
    write_json(PROGRESS, state)
    try:
        if args.wait_for_training:
            await_training(state)
        else:
            job = json.loads(TRAIN_JOB.read_text(encoding='utf-8'))
            if job.get('status') != 'finished' or job.get('exit_code') != 0:
                raise ValueError('Training still running or failed')
        plan = json.loads(TRAIN_PLAN.read_text(encoding='utf-8'))
        datasets = {'known': plan['dev'], 'public': plan['public_dev']}
        for spec in datasets.values():
            if fingerprint(read_jsonl(spec['path'])) != spec['hash']:
                raise ValueError('Frozen generation inputs changed')
        targets = []
        # Public controls were not generated before training; retain same input/protocol.
        for role in ('base', 'v7'):
            targets.append({'role': role, 'split': 'public', 'baseline': role == 'base',
                            'adapter': None if role == 'base' else plan['starting_adapter'],
                            'adapter_sha256': None if role == 'base' else plan['starting_adapter_sha256'],
                            'output': f'runs/v12-factorial-{role}-public.jsonl'})
        for arm in plan['arms']:
            selected = validate_selection(plan, arm)
            for split in datasets:
                targets.append({'role': arm['name'], 'split': split, 'baseline': False,
                                'adapter': selected['directory'], 'adapter_sha256': selected['adapter_sha256'],
                                'output': f"runs/v12-factorial-{arm['name']}-{split}.jsonl"})
        evaluation = {'at': now(), 'training_plan_hash': fingerprint(plan), 'datasets': datasets,
                      'base': plan['base'], 'prompt_hash': plan['prompt_hash'], 'targets': targets,
                      'quantization': 'nf4', 'max_length': 1024, 'max_new_tokens': 256,
                      'semantic_baselines': {'base': 'runs/v12-blind-base-semantic.jsonl',
                                             'v7': 'runs/v12-blind-v7-semantic.jsonl'},
                      'policy': 'Known DEV200 plus auxiliary public DEV116; frozen checkpoint choice precedes generation. No release test, promotion or causal claim.'}
        if any(Path(target['output']).exists() for target in targets):
            raise ValueError('Generation output already exists')
        write_json(EVAL_PLAN, evaluation)
        state.update(phase='generating', evaluation_plan_hash=fingerprint(evaluation))
        write_json(PROGRESS, state)
        for target in targets:
            state.update(current_target=target['role']+'/'+target['split'], updated_at=now())
            write_json(PROGRESS, state)
            command = [sys.executable, '-X', 'utf8', '-u', '-m', 'scripts.low_cpu_run', 'evaluate',
                       '--input', datasets[target['split']]['path'], '--output', target['output'],
                       '--quantization', 'nf4', '--max-length', '1024', '--max-new-tokens', '256']
            command += ['--baseline'] if target['baseline'] else ['--adapter-dir', target['adapter']]
            subprocess.run(command, check=True)
            summary = json.loads(Path(target['output']).with_suffix('.summary.json').read_text(encoding='utf-8'))
            if (summary['adapter_sha256'] != target['adapter_sha256']
                    or summary['data_hash'] != datasets[target['split']]['hash']
                    or summary['base'] != plan['base'] or summary['prompt_hash'] != plan['prompt_hash']):
                raise ValueError('Generated protocol or weight binding changed')
            state['completed'].append(state['current_target'])
            write_json(PROGRESS, state)
        state.update(phase='generation_complete_individual_semantic_review_pending', updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence', updated_at=now(), error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(PROGRESS, state)


if __name__ == '__main__':
    main()
