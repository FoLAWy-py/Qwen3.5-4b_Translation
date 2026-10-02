"""Persist a fresh, process-bound progress observation and CPU-only helper tests."""
import argparse
import hashlib
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--live-audit',required=True)
    parser.add_argument('--previous',default='runs/qwen35-v3-stage-checkpoint-039.json')
    parser.add_argument('--owned-wait')
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists()
    prior=load(args.previous)
    live=load(args.live_audit)
    actual=live['actual_process']
    process=psutil.Process(actual['pid'])
    assert abs(process.create_time()-actual['created_epoch'])<.001
    assert process.cmdline()==actual['command']
    assert live['plan_hash']==prior['plan_hash'] and live['frozen_budget_prefix_verified']
    job=load(prior['training']['job'])
    assert job['status']=='running'
    recall=load('runs/qwen35-v3-reviewed23-sft-recall-progress.json')
    assert recall['phase']=='waiting_for_training_exit'
    assert recall['gpu_model_loaded_in_orchestrator'] is False
    helpers={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path('witrans_tools/qwen35_lora_precision.py'),Path('tests/test_qwen35_lora_precision.py'))}
    receipt_path=Path('runs/qwen35-v3-lora-precision-helper-cpu-tests.json')
    if not receipt_path.exists():
        write_json(receipt_path,dict(at=now(),source_sha256=helpers,
            command='uv run --no-project --python .venv-qwen35/Scripts/python.exe python -X utf8 -m pytest tests/test_qwen35_lora_precision.py -q',
            observed_exit_code=0,observed_stdout='7 passed in 1.76s',passed=7,
            scope='Executed CPU fixtures: base storage/value preservation, exact BF16 adapter rounding, full validation before mutation, unsafe trainable/training/inventory/dtype/extra-adapter rejection, CUDA-only public-entry CPU rejection.',
            default_runtime_changed=False,active_training_changed=False,gpu_executed=False,
            quality_or_speed_proven=False,full_semantic_and_own_weight_latency_validation_pending=True))
    receipt=load(receipt_path)
    assert receipt['source_sha256']==helpers
    prior.update(at=now(),previous_checkpoint=args.previous,
        continuation_classification=f'Executed-budget evidence through{live["completed_updates"]} updates; current actual training process revalidated live.',
        actual_live_audit=args.live_audit,actual_live_audit_hash=fingerprint(live),
        cpu_precision_helper_receipt=str(receipt_path),cpu_precision_helper_receipt_hash=fingerprint(receipt),
        optional_precision_helper='Ready but not integrated into any default loader, active training or queued recall. Future candidate-specific experiment remains unfrozen; no GPU speed/quality evidence.',
        next='Continue the same PID and existing serial waiter; do not duplicate GPU jobs. On successful completion verify64/token/weight/memory evidence, then individually review103 source-bound translations. Frozen TRAIN gates precede DEV and own-weight timing. No new400 source collection until final weight and runtime freeze.')
    prior['training'].update(actual_prefix_audit=args.live_audit,prefix_updates=live['completed_updates'],
        prefix_tokens=live['full_tokens'],logged_prefix_peak_reserved_gib=live['reported_peak_reserved_gib'])
    if args.owned_wait:
        observation=load(args.owned_wait)
        assert observation['verified_process']==actual and observation['terminal'] is False
        prior.update(owned_wait_path=args.owned_wait,owned_wait_hash=fingerprint(observation),
            continuation_classification='Verified wait: current actual training PID/epoch/command and Win32 handle checked; observation timeout is nonterminal.')
    write_json(dest,prior)
    print(dict(checkpoint=str(dest),verified_prefix_updates=live['completed_updates'],actual_pid=process.pid))


if __name__=='__main__':
    main()
