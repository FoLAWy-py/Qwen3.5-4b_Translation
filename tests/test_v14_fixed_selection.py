import copy
import hashlib
import pytest
from scripts.run_v14_evaluation import validate_fact_selection
from witrans_tools.common import fingerprint,write_json


def fixture(tmp_path):
    output = tmp_path/'model'
    plan = {'arms':{'matching':str(output)},'updates':16,'checkpoints':[8,16],
            'selected_checkpoint':16,'expected_training_forward_calls':640,'expected_training_backward_calls':384,
            'starting_adapter_sha256':'parent','base':{'revision':'revision'},'prompt_hash':'prompt'}
    checkpoints = []
    for step in (8,16):
        directory = output/f'checkpoint-{step}'
        directory.mkdir(parents=True)
        value = f'checkpoint {step}'.encode()
        (directory/'adapter_model.safetensors').write_bytes(value)
        digest = hashlib.sha256(value).hexdigest()
        write_json(directory/'witrans_adapter.json',{'adapter_sha256':digest,'parent_adapter_sha256':'parent',
                                                     'base_revision':'revision','prompt_hash':'prompt'})
        checkpoints.append({'step':step,'directory':str(directory),'adapter_sha256':digest})
    config = {'mode':'matching','plan_hash':fingerprint(plan),'expected_full_tokens_including_recomputation':1600}
    metrics = {'mode':'matching','plan_hash':fingerprint(plan),'full_tokens':1600,'forward_calls':640,'backward_calls':384,
               'peak_reserved_gib':4.75,'history':[{'step':i,'forward_calls':40*i,'backward_calls':24*i,'full_tokens':100*i}
                                                for i in range(1,17)],'checkpoints':checkpoints,'selected':checkpoints[-1]}
    write_json(output/'run_config.json',config)
    write_json(output/'metrics.json',metrics)
    return plan,output,metrics


def test_intermediate_checkpoint_cannot_replace_frozen_final(tmp_path):
    plan,output,metrics = fixture(tmp_path)
    assert validate_fact_selection(plan,'matching')['step']==16
    wrong = copy.deepcopy(metrics)
    wrong['selected'] = wrong['checkpoints'][0]
    write_json(output/'metrics.json',wrong)
    with pytest.raises(ValueError,match='Fixed final'):
        validate_fact_selection(plan,'matching')


def test_changed_training_call_count_rejected(tmp_path):
    plan,output,metrics = fixture(tmp_path)
    metrics['history'][5]['backward_calls']-=1
    write_json(output/'metrics.json',metrics)
    with pytest.raises(ValueError,match='counters'):
        validate_fact_selection(plan,'matching')


def test_changed_selected_weights_rejected(tmp_path):
    plan,_,metrics = fixture(tmp_path)
    from pathlib import Path
    (Path(metrics['selected']['directory'])/'adapter_model.safetensors').write_bytes(b'replaced')
    with pytest.raises(ValueError,match='weights'):
        validate_fact_selection(plan,'matching')
