import copy
import pytest
from archive.qwen3.scripts.review_v12_factorial import binding, collect
from witrans_tools.common import fingerprint


def fixture():
    row = {'id':'case','input':{'text':'Do not leave.','target_lang':'zh-CN'},
           'raw':'{"translation":"不要离开。"}','reference':{'translation':'别离开。'},
           'ended':True,'prediction':{'translation':'不要离开。'}}
    ref = {'id':'case','group_id':'family','category':'daily','input':row['input'],'output':row['reference']}
    decision = {'id':'case','verdict':'pass','note':'否定正确。','language_correct':True,'binding_hash':binding(row)}
    return row,ref,decision


def test_changed_negation_never_inherits_previous_reading():
    row,ref,decision = fixture()
    changed = copy.deepcopy(row)
    changed['raw'] = '{"translation":"请离开。"}'
    changed['prediction']['translation'] = '请离开。'
    assert collect(changed,ref,{},[('old',decision)]) is None
    assert collect(row,ref,{},[('old',decision)])['verdict'] == 'pass'


def test_partial_or_reformatted_output_is_not_exact_cache_match():
    row,ref,decision = fixture()
    row['ended'] = False
    assert collect(row,ref,{},[('old',decision)]) is None


def test_stale_manual_reading_is_rejected():
    row,ref,decision = fixture()
    manual = {**decision,'reviewer':'Codex','generation_hash':fingerprint(row)}
    row['raw'] = 'A replaced output.'
    with pytest.raises(ValueError,match='stale'):
        collect(row,ref,{'case':manual},[])
