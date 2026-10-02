"""Shared generation implementation: explicit budget, greedy decoding and EOS."""
import json
from collections.abc import Mapping
from .protocol import make_messages, parse_translation


class GenerationRuntime:
    def translate(
        self, text: str, target_lang: str = "zh-CN", *,
        context: str = "", glossary: Mapping[str, str] | None = None,
        max_new_tokens: int = 256,
    ) -> dict[str, str]:
        raw, ended = self.generate_raw(text, target_lang, context=context,
                                       glossary=glossary, max_new_tokens=max_new_tokens)
        if not ended:
            raise RuntimeError("输出未正常结束；缩短输入或增加预算后重试")
        return parse_translation(raw)

    def generate_raw(
        self, text: str, target_lang: str = "zh-CN", *, context: str = "",
        glossary: Mapping[str, str] | None = None, max_new_tokens: int = 256,
    ) -> tuple[str, bool]:
        """评测保留未修复的原始生成与 EOS 状态。"""
        self.last_generation_stats = None
        messages = make_messages(text, target_lang, context, glossary)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < self.max_length:
            raise ValueError("max_new_tokens 必须在 1 与 max_length-1 之间")
        if not text.strip():
            return json.dumps({"translation": text}, ensure_ascii=False), True
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tokenizer(prompt, add_special_tokens=False,
                                return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[-1]
        if prompt_length + max_new_tokens > self.max_length:
            raise ValueError("超出总 token 预算；缩短原文、上下文或术语，不静默截断")
        inputs = inputs.to("cuda:0")
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer 缺少 EOS，请核对基础模型")
        config = self._generation_config(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            eos_token_id=eos, pad_token_id=eos, use_cache=True)
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=config,
                **({'use_model_defaults':False} if getattr(self,'_supports_use_model_defaults',True) else {}))
        new_ids = outputs[0, prompt_length:].tolist()
        self.last_generation_stats = {"prompt_tokens": prompt_length, "generated_tokens_including_eos": len(new_ids)}
        ended = bool(new_ids and new_ids[-1] == eos)
        raw = self.tokenizer.decode(new_ids[:-1] if ended else new_ids, skip_special_tokens=False)
        return raw, ended
