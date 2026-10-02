"""Archive model-specific historical scripts and rebind imports, without copies."""
import re
from pathlib import Path

root=Path(__file__).resolve().parent.parent
destination=root/'archive/qwen3/scripts';destination.mkdir(exist_ok=True)
(destination/'__init__.py').write_text('"""Historical experiment scripts; not an installed CLI."""\n',encoding='utf-8')
explicit={'benchmark_short','build_bf16_test','build_precision_test','check_bf16_budget','compare_short_runtime',
    'compile_nf4_probe','compile_smoke','nf4_budget_probe','profile_nf4','select_adapter','select_bf16','select_precision',
    'summarize_inference_speed','prepare_base_dev_comparison','prepare_public_short_pilot','prepare_public_fact_pairs',
    'review_public_short','review_short_changes','review_short_rounding','audit_public_short','train_preference'}
paths=[p for p in (root/'scripts').glob('*.py') if 'qwen35' not in p.stem and (re.search(r'(?:^|_)v\d+(?:_|$)',p.stem) or p.stem in explicit)]
names={p.stem for p in paths}
for path in paths:path.rename(destination/path.name)
for folder in ('scripts','tests','archive'):
    for path in (root/folder).rglob('*.py'):
        text=path.read_text(encoding='utf-8')
        revised=text
        for name in names:
            revised=revised.replace(f'from scripts.{name} import',f'from archive.qwen3.scripts.{name} import')
            revised=revised.replace(f'import scripts.{name}',f'import archive.qwen3.scripts.{name}')
        if destination in path.parents:
            revised=revised.replace('from witrans import','from archive.qwen3.runtime import')
        if revised!=text:path.write_text(revised,encoding='utf-8')
print(f'Archived {len(paths)} Qwen3-specific scripts; shared data/review/stats helpers retained.')
