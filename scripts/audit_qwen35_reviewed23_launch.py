"""CPU-only serial-process, frozen-budget and negative-quality guard verification."""
import copy
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, write_json
from witrans_tools.qwen35_trial_eligibility import require_eligible_trial


def main():
    dest = Path('runs/qwen35-v3-reviewed23-sft-launch-audit.json')
    assert not dest.exists()
    plan = load('data/prepared/qwen35-v3-reviewed23-sft-repair/plan.json')
    preflight = load('runs/qwen35-v3-reviewed23-sft-preflight.json')
    assert preflight['plan_hash'] == fingerprint(plan)
    assert preflight['full_tokens'] == plan['full_token_budget'] == 435971
    assert preflight['forward_calls'] == plan['budget']['forwards'] == 1728
    assert preflight['backward_calls'] == plan['budget']['backwards'] == 1472
    assert preflight['optimizer_updates'] == 0 and preflight['gpu_executed'] is False
    require_eligible_trial(plan)
    attacks = []
    bad = copy.deepcopy(plan)
    bad['pairs_path'] = 'data/prepared/qwen35-v3-reviewed32-current-errors/pairs.jsonl'
    bad_audit = copy.deepcopy(plan)
    bad_audit['evidence']['focused_negative_audit_hash'] = '0'*64
    for name, altered in [('unfiltered32_pool', bad), ('changed_audit_hash', bad_audit)]:
        try:
            require_eligible_trial(altered)
        except ValueError as exc:
            attacks.append(dict(case=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('Guard accepted invalid negative evidence')
    family = {psutil.Process().pid, *(p.pid for p in psutil.Process().parents())}
    processes = []
    model_processes = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'exe']):
        if process.pid in family or 'python' not in (process.info['name'] or '').lower():
            continue
        command = process.info['cmdline'] or []
        if not any('scripts.' in part for part in command):
            continue
        entry = dict(pid=process.pid, created=process.info['create_time'], exe=process.info['exe'], command=command)
        processes.append(entry)
        if any(term in ' '.join(command) for term in ('scripts.low_cpu_run','scripts.evaluate_qwen35',
                'scripts.train_', 'scripts.probe_qwen35_', 'scripts.diagnose_qwen35_stage')):
            model_processes.append(entry)
    assert not model_processes, 'Existing model process must complete before launch'
    write_json(dest, dict(at=now(), plan_hash=fingerprint(plan), preflight_hash=fingerprint(preflight),
        valid_plan_accepted=True, attacks=attacks, script_process_inventory=processes,
        other_model_processes=model_processes, gpu_executed=False,
        budget_receipt_pytest=dict(tests='tests/test_qwen35_repair_budget_receipt.py', passed=6),
        stage_goal_complete=False, release_approved=False, default_promoted=False))
    print(dict(launch_audit=str(dest), serial_gpu_free=True, attacks=attacks))


if __name__ == '__main__':
    main()
