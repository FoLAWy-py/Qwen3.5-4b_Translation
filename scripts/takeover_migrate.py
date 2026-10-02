"""One-time mechanical migration; preserves prompt/decoder bytes and history."""
from pathlib import Path

root=Path(__file__).resolve().parent.parent
old=(root/'witrans.py').read_text(encoding='utf-8')
protocol=old[old.index('from __future__'):old.index('class _LocalTranslator:')]
methods=old[old.index('    def translate('):old.index('\n\ndef _place_bf16_cpu_vocab')]
archive=root/'archive/qwen3'
archive.mkdir(parents=True,exist_ok=True)
(archive/'__init__.py').write_text('"""Historical Qwen3 experiments; excluded from default installation/tests."""\n',encoding='utf-8')
legacy=old[:old.index('SYSTEM_PROMPT =')]
legacy+='from witrans_tools.protocol import SYSTEM_PROMPT, make_messages, parse_translation\n\n'
legacy+=old[old.index('class _LocalTranslator:'):old.index('    def translate(')]
legacy+='\n'+old[old.index('\n\ndef _place_bf16_cpu_vocab'):]
legacy=legacy.replace('class _LocalTranslator:', 'from witrans_tools.runtime import GenerationRuntime\n\n\nclass _LocalTranslator(GenerationRuntime):')
(archive/'runtime.py').write_text(legacy,encoding='utf-8')
(root/'witrans_tools/protocol.py').write_text('"""Frozen model-independent translation protocol shared by all tools."""\n'+protocol,encoding='utf-8')
(root/'witrans_tools/runtime.py').write_text('"""Shared generation implementation: explicit budget, greedy decoding and EOS."""\nimport json\nfrom collections.abc import Mapping\nfrom .protocol import make_messages, parse_translation\n\n\nclass GenerationRuntime:\n'+methods,encoding='utf-8')
(root/'witrans.py').write_text('"""Compatibility imports; the installed API is witrans_tools.Qwen35Translator."""\nfrom witrans_tools.protocol import SYSTEM_PROMPT, make_messages, parse_translation\nfrom witrans_tools.runtime import GenerationRuntime as _LocalTranslator\nfrom witrans_tools.qwen35 import Qwen35Translator\n\nWiTrans = Qwen35Translator\n',encoding='utf-8')
(root/'witrans-4b-model-spec-v1.3.md').rename(archive/'witrans-4b-model-spec-v1.3.md')
for name in ('__main__','download','evaluate','train','doctor'):
    path=root/f'witrans_tools/{name}.py'
    text=path.read_text(encoding='utf-8')
    text=text.replace('from .common import','from witrans_tools.common import').replace('from .data import','from witrans_tools.data import')
    text=text.replace('from witrans import WiTrans, SYSTEM_PROMPT, _LocalTranslator, parse_translation','from .runtime import WiTrans, SYSTEM_PROMPT, _LocalTranslator, parse_translation')
    (archive/f'{name}.py').write_text(text,encoding='utf-8')
    if name not in ('__main__','download'):path.unlink()
history=root/'archive/qwen3/tests';history.mkdir(parents=True,exist_ok=True)
historical=['test_adapter_integrity.py','test_inference_placement.py']
historical += [p.name for p in (root/'tests').glob('test_v*.py')]
for name in historical:
    path=root/'tests'/name
    text=path.read_text(encoding='utf-8').replace('from witrans import','from archive.qwen3.runtime import')
    (history/name).write_text(text,encoding='utf-8');path.unlink()
print('Migrated shared protocol/generation once; archived Qwen3 loaders/spec/tests.')
