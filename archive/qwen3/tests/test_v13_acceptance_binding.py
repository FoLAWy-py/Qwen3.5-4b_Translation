import copy
import pytest
from archive.qwen3.scripts.finalize_v13_optimization import complete_review
from archive.qwen3.scripts.review_v12_factorial import binding
from witrans_tools.common import fingerprint, write_jsonl


def evidence(tmp_path):
    row = {'id':'test-en','input':{'text':'不要删除备份。','target_lang':'en'},
           'reference':{'translation':'Do not delete the backup.'},
           'raw':'{"translation":"Do not delete the backup."}',
           'prediction':{'translation':'Do not delete the backup.'},'ended':True,'seconds':2.0}
    ref = {'id':row['id'],'group_id':'family','category':'work','input':row['input'],'output':row['reference']}
    decision = {'id':row['id'],'group_id':'family','category':'work','target_lang':'en',
                'binding_hash':binding(row),'generation_hash':fingerprint(row),
                'output_hash':fingerprint({k:row[k] for k in ('input','raw','reference')}),
                'reviewer':'Codex','verdict':'pass','note':'否定和备份对象正确。',
                'format_valid':True,'ended':True,'language_correct':True}
    stem = str(tmp_path/'trial')
    refs = str(tmp_path/'refs.jsonl')
    write_jsonl(stem+'.jsonl',[row])
    write_jsonl(stem+'-semantic.jsonl',[decision])
    write_jsonl(refs,[ref])
    return stem,{'path':refs,'hash':fingerprint([ref])},row,decision


def test_changed_generation_metadata_requires_rebound_reading(tmp_path):
    stem,spec,row,_ = evidence(tmp_path)
    assert complete_review(stem,spec)[0] == [row]
    changed = copy.deepcopy(row)
    changed['seconds'] = 3.0
    write_jsonl(stem+'.jsonl',[changed])
    with pytest.raises(ValueError,match='bound semantic'):
        complete_review(stem,spec)


def test_incomplete_and_stale_semantic_export_rejected(tmp_path):
    stem,spec,_,decision = evidence(tmp_path)
    changed = {**decision,'output_hash':'stale'}
    write_jsonl(stem+'-semantic.jsonl',[changed])
    with pytest.raises(ValueError,match='bound semantic'):
        complete_review(stem,spec)
    write_jsonl(stem+'-semantic.jsonl',[])
    with pytest.raises(ValueError,match='Complete individual'):
        complete_review(stem,spec)
