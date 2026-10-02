"""Compatibility imports; the installed API is witrans_tools.Qwen35Translator."""
from witrans_tools.protocol import SYSTEM_PROMPT, make_messages, parse_translation
from witrans_tools.runtime import GenerationRuntime as _LocalTranslator
from witrans_tools.qwen35 import Qwen35Translator

WiTrans = Qwen35Translator
