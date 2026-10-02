import pytest
from witrans_tools.data import encode_example


class TemplateTokenizer:
    eos_token_id=99
    def apply_chat_template(self,*args,**kwargs):
        assert kwargs['return_dict'] is False
        return {'input_ids':[10,20,30],'attention_mask':[1,1,1]}
    def encode(self,*args,**kwargs):
        return [40,50]


def test_new_template_mapping_preserves_real_prompt_tokens_and_mask():
    encoded=encode_example(TemplateTokenizer(),dict(input=dict(text='Test.',target_lang='en'),output=dict(translation='Test.')))
    assert encoded['input_ids']==[10,20,30,40,50,99]
    assert encoded['labels']==[-100,-100,-100,40,50,99]


def test_nested_or_noninteger_template_ids_are_rejected_before_training():
    class InvalidTokenizer(TemplateTokenizer):
        def apply_chat_template(self,*args,**kwargs): return {'input_ids':[[10,20,30]]}
    with pytest.raises(ValueError,match='flat integer'):
        encode_example(InvalidTokenizer(),dict(input=dict(text='Test.',target_lang='en'),output=dict(translation='Test.')))
