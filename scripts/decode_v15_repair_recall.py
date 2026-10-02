"""Wait for the owned DEV pipeline, then decode only the eleven TRAIN errors."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    state_path = Path('runs/v15-repair-recall-progress.json')
    if state_path.exists():
        raise ValueError('Preserve recall attempt')
    plan = load('data/prepared/v15-error-repair/plan.json')
    pairs = read_jsonl(plan['pairs_path'])
    assert len(pairs) == 11 and fingerprint(pairs) == plan['pairs_hash']
    selection = load('runs/v15-error-fixed-selection.json')
    assert selection['training_plan_hash'] == fingerprint(plan)
    weights = Path(selection['selected']['directory']) / 'adapter_model.safetensors'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == selection['selected']['adapter_sha256']
    owner_path = 'runs/v15-error-optimization-job.json'
    owner = load(owner_path)
    assert owner['command'][-1] == 'scripts.run_v15_error_optimization'
    assert Path(owner['cwd']).resolve() == Path.cwd().resolve()
    proc = psutil.Process(owner['process_pid']) if owner['status'] == 'running' else None
    if proc is not None:
        assert proc.cmdline() == owner['command']
    state = dict(at=now(), phase='waiting_for_owned_dev_pipeline', owner_pid=owner.get('process_pid'),
                 owner_created=proc.create_time() if proc else None, release_approved=False)
    write_json(state_path, state)
    try:
        while owner['status'] in ('launching', 'running'):
            if proc and proc.is_running():
                assert proc.create_time() == state['owner_created'], 'PID reused'
            time.sleep(5)
            owner = load(owner_path)
        assert owner['status'] == 'finished' and owner['exit_code'] == 0, 'DEV pipeline failed'
        assert load('runs/v15-error-optimization-progress.json')['phase'] == 'individual_semantic_review_pending'
        input_path = 'runs/v15-repair-recall-input.jsonl'
        output_path = 'runs/v15-repair-recall.jsonl'
        assert not Path(input_path).exists() and not Path(output_path).exists()
        write_jsonl(input_path, pairs)
        recall_plan = dict(at=now(), training_plan_hash=fingerprint(plan), input=input_path, output=output_path,
                           data_hash=fingerprint(pairs), selected=selection['selected'], base=plan['base'],
                           prompt_hash=plan['prompt_hash'], quantization='nf4', max_length=1024,
                           max_new_tokens=256, scope='TRAIN recall only; never heldout efficacy', release_approved=False)
        write_json('runs/v15-repair-recall-plan.json', recall_plan)
        state.update(phase='decoding_train_recall', updated_at=now())
        write_json(state_path, state)
        subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m', 'scripts.low_cpu_run', 'evaluate',
                        '--adapter-dir', plan['output'], '--input', input_path, '--output', output_path,
                        '--quantization', 'nf4', '--max-length', '1024', '--max-new-tokens', '256'], check=True)
        summary = load(Path(output_path).with_suffix('.summary.json'))
        assert summary['adapter_sha256'] == selection['selected']['adapter_sha256']
        assert summary['data_hash'] == fingerprint(pairs) and summary['prompt_hash'] == plan['prompt_hash']
        assert summary['base'] == plan['base']
        state.update(phase='individual_train_recall_review_pending', updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence', error=str(exc), updated_at=now())
        raise
    finally:
        write_json(state_path, state)


if __name__ == '__main__':
    main()
