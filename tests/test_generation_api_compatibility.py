"""Shared decoder keeps explicit greedy settings across Transformers APIs."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
from witrans import _LocalTranslator


class Inputs(dict):
    def to(self, device):
        assert device == 'cuda:0'
        return self


class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False and kwargs['tokenize'] is False
        return 'prompt'
    def __call__(self, prompt, **kwargs):
        return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))
    def decode(self, ids, **kwargs):
        assert ids == [7] and kwargs['skip_special_tokens'] is False
        return '{"translation":"你好"}'


class Output:
    def __getitem__(self, index):
        assert index == (0, slice(3, None))
        return SimpleNamespace(tolist=lambda: [7, 99])


@pytest.mark.parametrize('supports_defaults', [True, False])
def test_generation_defaults_flag_is_api_specific(supports_defaults):
    translator = _LocalTranslator.__new__(_LocalTranslator)
    translator.tokenizer = Tokenizer()
    translator.max_length = 1024
    translator._supports_use_model_defaults = supports_defaults
    translator._torch = SimpleNamespace(inference_mode=nullcontext)
    translator._generation_config = lambda **kwargs: SimpleNamespace(**kwargs)
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        return Output()
    translator.model = SimpleNamespace(generate=generate)
    assert translator.generate_raw('Hello', 'zh-CN', max_new_tokens=256) == ('{"translation":"你好"}', True)
    assert ('use_model_defaults' in captured) is supports_defaults
    if supports_defaults: assert captured['use_model_defaults'] is False
    config = captured['generation_config']
    assert config.do_sample is False and config.num_beams == 1
    assert config.max_new_tokens == 256 and config.eos_token_id == 99
    assert translator.last_generation_stats == dict(prompt_tokens=3, generated_tokens_including_eos=2)
