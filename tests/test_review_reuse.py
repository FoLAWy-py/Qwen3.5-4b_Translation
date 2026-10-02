from copy import deepcopy
import pytest
from witrans_tools.review_reuse import content_hash, reuse_identical_reviews

def evidence():
    row = {'id':'one', 'category':'daily', 'input':{'text':'Keep it.', 'target_lang':'zh-CN'},
        'raw':'{"translation":"保留它。"}', 'reference':{'translation':'留着它。'},
        'prediction':{'translation':'保留它。'}, 'ended':True}
    review = {'id':'one', 'reviewer':'Codex', 'at':'original', 'verdict':'pass', 'ended':True,
        'format_valid':True, 'output_hash':content_hash(row)}
    return row, review

def test_identical_content_with_new_timing_retains_actual_review():
    row, decision = evidence()
    current = {**row, 'seconds':9.0}
    result = reuse_identical_reviews([row], [decision], [current])[0]
    assert result['at'] == 'original'
    assert result['verdict'] == 'pass'
    assert 'reused_at' in result

@pytest.mark.parametrize('field,value', [('raw','{"translation":"丢掉它。"}'), ('input',{'text':'Discard it.', 'target_lang':'zh-CN'}),
    ('ended',False), ('prediction',{'translation':'丢掉它。'})])
def test_changed_evidence_requires_new_review(field, value):
    row, decision = evidence()
    current = deepcopy(row)
    current[field] = value
    with pytest.raises(ValueError):
        reuse_identical_reviews([row], [decision], [current])

def test_stale_or_incomplete_reviews_rejected():
    row, decision = evidence()
    with pytest.raises(ValueError):
        reuse_identical_reviews([row], [{**decision, 'output_hash':'stale'}], [row])
    with pytest.raises(ValueError):
        reuse_identical_reviews([row], [], [row])
    with pytest.raises(ValueError):
        reuse_identical_reviews([row, row], [decision], [row])
