import pytest
from witrans_tools.critical_spans import encode_critical_spans

class CharacterTokenizer:
    is_fast = True
    eos_token_id = 999
    def apply_chat_template(self, *args, **kwargs):
        return [7,8,9]
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def __call__(self, text, **kwargs):
        return {'input_ids':self.encode(text), 'offset_mapping':[(i,i+1) for i in range(len(text))]}

def record(translation):
    return {'id':'span-test', 'input':{'text':'Test.', 'target_lang':'zh-CN','context':'','glossary':{}},
        'output':{'translation':translation}}

def test_json_escaping_keeps_weights_inside_reviewed_translation_span():
    encoded = encode_critical_spans(CharacterTokenizer(), record('他说"不借给我"。'), ['"不借给我"'])
    focused = ''.join(chr(token) for token,weight in zip(encoded['input_ids'],encoded['token_weights']) if weight>1)
    assert focused == '\\"不借给我\\"'
    assert encoded['token_weights'][:3] == [0.,0.,0.]
    assert encoded['labels'][-1] == 999 and encoded['token_weights'][-1] == 1.

def test_ambiguous_repeated_span_cannot_silently_select_one_occurrence():
    with pytest.raises(ValueError, match='exactly once'):
        encode_critical_spans(CharacterTokenizer(), record('指标与指标不同。'), ['指标'])

def test_offsets_that_do_not_match_training_token_ids_are_rejected():
    class WrongOffsetTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result['input_ids'][0] += 1
            return result
    with pytest.raises(ValueError, match='tokenization changed'):
        encode_critical_spans(WrongOffsetTokenizer(), record('不要透露。'), ['不要'])

def test_span_review_cannot_be_reused_after_chosen_answer_changes():
    from witrans_tools.common import fingerprint
    from witrans_tools.critical_spans import validated_annotations
    row = {**record('只有他来才开始。'), 'rejected':{'translation':'等我来开始。'}, 'preference_issue':'Role changed.'}
    content = {k:row[k] for k in ('input','output','rejected','preference_issue')}
    annotation = {'id':row['id'], 'critical_spans':['只有他来'], 'multiplier':3.,
        'reviewer':'Codex','at':'recorded','note':'Individually read.',
        'content_hash':fingerprint({**content,'critical_spans':['只有他来'],'multiplier':3.})}
    report = {'train_hash':fingerprint([row]),'annotations':[annotation],'annotation_hash':fingerprint([annotation])}
    assert validated_annotations([row],report)[row['id']] == annotation
    changed = {**row, 'output':{'translation':'只有我来才开始。'}}
    # Even rehashing the outer training manifest must not validate stale approval.
    with pytest.raises(ValueError, match='individual content-bound review'):
        validated_annotations([changed], {**report,'train_hash':fingerprint([changed])})
