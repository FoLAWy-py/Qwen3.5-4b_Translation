"""Runtime identity/placement/budget failures remain active with python -O."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from witrans_tools.qwen35 import MODEL_ID, REVISION, PROMPT_HASH, validate_identity, validate_cuda, validate_placement, validate_runtime_files


def fixture(tmp_path):
    base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text(json.dumps({'model_type':'qwen3_5','text_config':{'model_type':'qwen3_5_text'}}))
    receipt={'model_id':MODEL_ID,'revision':REVISION,'status':'complete'}
    (base/'witrans_base.json').write_text(json.dumps(receipt))
    adapter=tmp_path/'adapter';adapter.mkdir()
    metadata={'base_model_id':MODEL_ID,'base_revision':REVISION,'prompt_hash':PROMPT_HASH,'smoke_only':False,'adapter_sha256':'wrong'}
    (adapter/'witrans_adapter.json').write_text(json.dumps(metadata))
    (adapter/'adapter_config.json').write_text('{}')
    (adapter/'adapter_model.safetensors').write_bytes(b'wrong weights')
    return base,adapter,receipt,metadata


@pytest.mark.parametrize('field,value',[('model_id','Qwen/Qwen3-4B'),('revision','main'),('status','downloading')])
def test_wrong_base_rejected(tmp_path,field,value):
    base,adapter,receipt,_=fixture(tmp_path)
    receipt[field]=value;(base/'witrans_base.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Base model'):validate_identity(base,adapter,1024)


@pytest.mark.parametrize('field,value',[('base_revision','main'),('prompt_hash','changed'),('smoke_only',True),('smoke_only',None)])
def test_wrong_adapter_rejected(tmp_path,field,value):
    base,adapter,_,metadata=fixture(tmp_path)
    metadata[field]=value;(adapter/'witrans_adapter.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):validate_identity(base,adapter,1024)


def test_hash_budget_cuda_and_offload_checks(tmp_path):
    base,adapter,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='SHA256'):validate_identity(base,adapter,1024)
    for budget in (True,1024.0,512,2048):
        with pytest.raises(ValueError,match='budget'):validate_identity(base,adapter,budget)
    with pytest.raises(RuntimeError,match='CUDA'):
        validate_cuda(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    with pytest.raises(RuntimeError,match='offload'):
        validate_placement(SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=SimpleNamespace(type='cpu'))])))
    with pytest.raises(ValueError,match='config/tokenizer/template'):
        validate_runtime_files(base,adapter)


def test_optimized_python_preserves_runtime_guards(tmp_path):
    base,adapter,_,_=fixture(tmp_path)
    code='''
import sys
from types import SimpleNamespace as S
from witrans_tools.qwen35 import validate_identity, validate_cuda, validate_placement, validate_runtime_files
from witrans_tools.runtime import GenerationRuntime
def rejected(call):
    try: call()
    except (ValueError,RuntimeError): return
    raise RuntimeError('Guard disappeared under optimized Python')
base,adapter=sys.argv[1:]
rejected(lambda:validate_identity(base,adapter,1024))
rejected(lambda:validate_identity(base,None,2048))
rejected(lambda:validate_runtime_files(base,adapter))
rejected(lambda:validate_cuda(S(cuda=S(is_available=lambda:False))))
rejected(lambda:validate_placement(S(parameters=lambda:iter([S(device=S(type='cpu'))]))))
t=GenerationRuntime();t.max_length=1024
rejected(lambda:t.generate_raw('x','en',max_new_tokens=1024))
print('optimized guards active')
'''
    result=subprocess.run([sys.executable,'-O','-c',code,str(base),str(adapter)],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert 'optimized guards active' in result.stdout
