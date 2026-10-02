import hashlib
import json

import pytest

from archive.qwen3.scripts.run_v12_factorial_evaluation import validate_selection
from witrans_tools.common import fingerprint


def fixture(tmp_path):
    output = tmp_path/'model'
    output.mkdir()
    arm = {'name':'base-old', 'output':str(output), 'selection_output':str(tmp_path/'selection.json')}
    plan = {'updates':2, 'training_input_tokens_per_arm':99, 'checkpoints':[1,2],
            'base':{'revision':'fixed'}, 'prompt_hash':'prompt'}
    candidates = []
    for step, score in ((1, .6),(2, .7)):
        checkpoint = output/f'checkpoint-{step}'
        checkpoint.mkdir()
        weights = bytes([step])*10
        (checkpoint/'adapter_model.safetensors').write_bytes(weights)
        digest = hashlib.sha256(weights).hexdigest()
        (checkpoint/'witrans_adapter.json').write_text(json.dumps({'adapter_sha256':digest,
            'base_revision':'fixed','prompt_hash':'prompt'}), encoding='utf-8')
        candidates.append({'step':step,'directory':str(checkpoint),'adapter_sha256':digest,
                            'nll':{'dev':score,'public_dev':score,'balanced_mean':score}})
    selection = {'arm':'base-old','plan_hash':fingerprint(plan), 'candidates':candidates,'selected':candidates[0]}
    metrics = {'actual_training_input_tokens':99,'history':[{'full_tokens':50},{'full_tokens':99}],
                'selected':candidates[0]}
    (output/'run_config.json').write_text(json.dumps({'arm':arm,'plan_hash':fingerprint(plan)}),encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps(metrics),encoding='utf-8')
    (tmp_path/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
    return plan,arm,selection,metrics


def test_cannot_select_worse_checkpoint_after_generation(tmp_path):
    plan,arm,selection,metrics = fixture(tmp_path)
    assert validate_selection(plan,arm)['step'] == 1
    selection['selected'] = selection['candidates'][1]
    metrics['selected'] = selection['selected']
    (tmp_path/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
    (tmp_path/'model'/'metrics.json').write_text(json.dumps(metrics),encoding='utf-8')
    with pytest.raises(ValueError,match='selection differs'):
        validate_selection(plan,arm)


def test_token_mismatch_cannot_enter_matched_evaluation(tmp_path):
    plan,arm,selection,metrics = fixture(tmp_path)
    metrics['actual_training_input_tokens'] = 98
    (tmp_path/'model'/'metrics.json').write_text(json.dumps(metrics),encoding='utf-8')
    with pytest.raises(ValueError,match='Training evidence differs'):
        validate_selection(plan,arm)


def test_changed_weights_do_not_inherit_checkpoint_selection(tmp_path):
    plan,arm,selection,metrics = fixture(tmp_path)
    (tmp_path/'model'/'checkpoint-1'/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='weights changed'):
        validate_selection(plan,arm)
