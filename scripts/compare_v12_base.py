"""Require complete content-bound matched evidence; do not approve release."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.paired_stats import paired_pass_interval


def main():
    refs = {r['id']: r for r in read_jsonl('data/prepared/v4/dev.jsonl')}
    evidence, counts, strata, summaries = {}, {}, {}, {}
    for role, stem in [('base','v12-base'),('v7','v7-critical')]:
        raw = {r['id']: r for r in read_jsonl(f'runs/{stem}-dev.jsonl')}
        decisions = {r['id']: r for r in read_jsonl(f'runs/{stem}-dev-semantic.jsonl')}
        summary = json.loads(Path(f'runs/{stem}-dev.summary.json').read_text(encoding='utf-8'))
        if set(refs) != set(raw) or set(refs) != set(decisions) or len(refs) != 200:
            raise ValueError('Incomplete matched200-row evidence')
        for key, row in raw.items():
            decision, ref = decisions[key], refs[key]
            if (row['input'] != ref['input'] or row['reference'] != ref['output']
                    or decision['output_hash'] != fingerprint({k:row[k] for k in ('input','raw','reference')})
                    or decision.get('reviewer') != 'Codex' or not decision.get('note')
                    or decision['format_valid'] != ('prediction' in row) or decision['ended'] != row['ended']
                    or decision['group_id'] != ref['group_id'] or decision['category'] != ref['category']
                    or decision['target_lang'] != ref['input']['target_lang']
                    or decision['verdict'] not in ('pass','minor','major','critical')):
                raise ValueError('Stale or mismatched semantic evidence')
        if summary['data_hash'] != fingerprint(list(refs.values())):
            raise ValueError('Input fingerprint changed')
        evidence[role] = list(decisions.values())
        counts[role] = dict(Counter(d['verdict'] for d in evidence[role]))
        strata[role] = defaultdict(Counter)
        for decision in evidence[role]:
            strata[role][f"{decision['category']}/{decision['target_lang']}"][decision['verdict']] += 1
        summaries[role] = summary
    for key in ('base','prompt_hash','data_hash','quantization','decoding'):
        if summaries['base'][key] != summaries['v7'][key]:
            raise ValueError(f'Matched protocol differs: {key}')
    if summaries['base']['adapter_sha256'] is not None or summaries['base']['adapter_dir'] is not None:
        raise ValueError('Base control has adapter')
    if summaries['v7']['adapter_sha256'] != '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424':
        raise ValueError('v7 weight identity differs')
    stats = paired_pass_interval(evidence['v7'], evidence['base'])
    report = {'at': now(), 'rows': 200, 'groups': len({r['group_id'] for r in refs.values()}),
              'counts': counts, 'v7_vs_base': stats,
              'strata': {role:{key:dict(value) for key,value in values.items()} for role,values in strata.items()},
              'format_valid': {role:sum(d['format_valid'] for d in rows) for role,rows in evidence.items()},
              'language_correct': {role:sum(d['language_correct'] for d in rows) for role,rows in evidence.items()},
              'eos': {role:sum(d['ended'] for d in rows) for role,rows in evidence.items()},
              'release_approved': False, 'default_promoted': False,
              'limits': 'Repeated known DEV; evaluator knew baseline identity.34 base rows reuse exact previous inspected output and166 newly read. Not a fully independent blinded review, new release evaluation or causal algorithm benefit; matched timing is not informative when format/output lengths differ.'}
    write_json('runs/v12-base-v7-comparison.json', report)
    progress_path = Path('runs/v12-base-progress.json')
    progress = json.loads(progress_path.read_text(encoding='utf-8'))
    progress.update(phase='matched_semantic_screening_complete_blind_recheck_pending', updated_at=now(),
                    comparison='runs/v12-base-v7-comparison.json')
    write_json(progress_path, progress)
    text = '# 原始NF4基模与v7同集200条开发对照\n\n'
    text += '| 模型 | pass | minor | major含critical | JSON |\n|---|---:|---:|---:|---:|\n'
    for role in ('base','v7'):
        c = counts[role]
        text += f"| {role} | {c.get('pass',0)} | {c.get('minor',0)} | {c.get('major',0)+c.get('critical',0)} | {report['format_valid'][role]}/200 |\n"
    text += '\n来源组配对统计：`' + json.dumps(stats, ensure_ascii=False) + '`。\n\n'
    text += '两个模型的revision、提示词、NF4、输入与greedy1024/256预算相同。语义独立于JSON检查，纯译文即使格式错误也按实际内容判断。\n\n'
    text += '审核者知道基模身份；34条严格同内容复用、166条新读，尚非完全独立盲审。已知DEV被反复用于研究，不能证明全新发布质量或当前算法的因果收益。默认权重未替换，Goal未通过；后续须完成来源审核、盲化复审、公开数据对照及新的600条发布验收。\n'
    Path('runs/v12-base-v7-comparison.md').write_text(text, encoding='utf-8')
    print({key:value for key,value in report.items() if key != 'strata'})


if __name__ == '__main__':
    main()
