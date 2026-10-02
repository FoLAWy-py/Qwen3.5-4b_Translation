"""CPU receipt rejection tests; fixtures are not actual training evidence."""
import hashlib
from pathlib import Path

import pytest

from scripts.decode_qwen35_repair_recall import verify_training
from witrans_tools.common import fingerprint, write_json


def receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output=Path('models/fixture')
    output.mkdir(parents=True)
    payload=b'CPU fixture only, not a model'
    (output/'adapter_model.safetensors').write_bytes(payload)
    sha=hashlib.sha256(payload).hexdigest()
    plan=dict(output=str(output),base=dict(model_id='Qwen/Qwen3.5-4B',revision='fixture-revision'),
        prompt_hash='fixture-prompt',starting_adapter_sha256='fixture-parent',
        selected_checkpoint=64,updates=64,accumulation=4,full_token_budget=305739,memory_cap_gib=6.5,
        schedule=[dict(pair='fixture',replay=['p']*(3 if i%4<3 else 2)) for i in range(256)],
        budget=dict(forwards=1216,backwards=960))
    metrics=dict(plan_hash=fingerprint(plan),selected_checkpoint=64,history=[dict(step=i) for i in range(1,65)],
        full_tokens=305739,forward_calls=1216,backward_calls=960,cpu_parameter_count=0,
        peak_reserved_gib=5,adapter_sha256=sha)
    return plan,metrics,sha


def publish(plan,metrics,sha):
    output=Path(plan['output'])
    metrics['plan_hash']=fingerprint(plan)
    write_json(output/'metrics.json',metrics)
    write_json(output/'run_config.json',dict(plan=plan,plan_hash=fingerprint(plan),expected_full_tokens=305739))
    write_json(output/'witrans_adapter.json',dict(plan_hash=fingerprint(plan),adapter_sha256=sha,
        parent_adapter_sha256=plan['starting_adapter_sha256'],base_model_id=plan['base']['model_id'],
        base_revision=plan['base']['revision'],prompt_hash=plan['prompt_hash'],smoke_only=False))


def test_larger_predeclared_replay_receipt(tmp_path,monkeypatch):
    plan,metrics,sha=receipt(tmp_path,monkeypatch)
    publish(plan,metrics,sha)
    assert verify_training(plan)==(metrics,sha)


@pytest.mark.parametrize('attack',['old_calls','missing_visit','false_budget','cpu_offload','changed_weights'])
def test_consistent_hashes_do_not_excuse_invalid_execution(tmp_path,monkeypatch,attack):
    plan,metrics,sha=receipt(tmp_path,monkeypatch)
    if attack=='old_calls':
        metrics.update(forward_calls=1024,backward_calls=768)
    elif attack=='missing_visit':
        plan['schedule'].pop()
    elif attack=='false_budget':
        plan['budget'].update(forwards=1024,backwards=768)
        metrics.update(forward_calls=1024,backward_calls=768)
    elif attack=='cpu_offload':
        metrics['cpu_parameter_count']=1
    publish(plan,metrics,sha)
    if attack=='changed_weights':
        (Path(plan['output'])/'adapter_model.safetensors').write_bytes(b'changed fixture')
    with pytest.raises(AssertionError):
        verify_training(plan)
