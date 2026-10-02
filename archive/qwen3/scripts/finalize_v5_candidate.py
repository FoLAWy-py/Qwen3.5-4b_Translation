"""Verify all candidate development reviews without claiming release acceptance."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    plan = load('runs/v5-evaluation-plan.json')
    selected = plan['selected']
    directory = Path(selected['directory'])
    if hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest() != selected['adapter_sha256']:
        raise ValueError('Chosen weights changed')
    selection = load('runs/v5-selection.json')
    if selection['selected'] != selected:
        raise ValueError('Post-generation model selection changed')
    for name, key in (('train', 'training_hash'), ('dev', 'development_hash')):
        if fingerprint(read_jsonl(f'models/witrans-4b-v5-sft/{name}_snapshot.jsonl')) != plan[key]:
            raise ValueError('Actual training snapshot changed')
    summary = load('runs/v5-selected-dev.summary.json')
    if any(summary[k] != plan[p] for k,p in (('data_hash','development_hash'), ('prompt_hash','prompt_hash'))):
        raise ValueError('Generation protocol or data mismatch')
    if summary['adapter_sha256'] != selected['adapter_sha256'] or summary['quantization'] != 'nf4' or summary['decoding'] != plan['decoding']:
        raise ValueError('Generation model or settings mismatch')
    raw = read_jsonl('runs/v5-selected-dev.jsonl')
    decisions = read_jsonl('runs/v5-selected-dev-semantic.jsonl')
    indexed = {r['id']:r for r in raw}
    if len(raw) != 200 or len(indexed) != 200 or len(decisions) != 200 or {d['id'] for d in decisions} != set(indexed):
        raise ValueError('Incomplete individually reviewed evidence')
    for decision in decisions:
        row = indexed[decision['id']]
        if decision['output_hash'] != fingerprint({k:row[k] for k in ('input','raw','reference')}):
            raise ValueError('Review no longer matches output')
    counts = dict(Counter(d['verdict'] for d in decisions))
    buckets = defaultdict(Counter)
    for d in decisions:
        buckets[f"{d['category']}/{d['target_lang']}"][d['verdict']] += 1
    report = {'at':now(), 'stage':'candidate_development_review_complete_start_comparison_pending', 'selected':selected,
        'reviewed_rows':200, 'reviewed_groups':100, 'verdicts':counts, 'pass_fraction':counts.get('pass',0)/200,
        'strata':dict(buckets), 'generation':summary, 'default_promoted':False, 'release_approved':False,
        'source_caveats':['028English does not explicitly specify train', '044Chinese legume term does not specify lentils',
            '047Chinese beater does not specify manual whisk; mixer is allowed as an electric beater reading'],
        'limitations':['Known200-row development set, not new600-row release test', 'Fresh starting-model comparison not finished',
            'One-pass development timings, not three-round fixed short-input performance gate',
            'No release context/glossary coverage or1024-token stress evidence yet for selected checkpoint']}
    write_json('runs/v5-candidate-acceptance.json', report)
    metadata = load(directory / 'witrans_adapter.json')
    metadata.update(quality_status='Experimental; reviewed known development pass69.5%; release goal unproven', acceptance_report='runs/v5-candidate-acceptance.json')
    write_json(directory / 'witrans_adapter.json', metadata)
    text = ('# v5候选开发审核\n\n678条/306来源组，NF4继续SFT85次更新，按开发答案NLL选第85步。'
        '起点NLL0.85450，所选0.66957；训练含评估与保存1483秒，峰值保留显存4.76 GiB。\n\n'
        f"已逐条阅读全部200条候选译文：{counts.get('pass',0)}通过、{counts.get('minor',0)}轻微、{counts.get('major',0)}严重；"
        'JSON200/200，正常结束200/200。默认权重未替换，尚不能通过发布验收。\n\n'
        '单轮开发生成平均3.26秒、P95 4.38秒、5.22实际输出token/s；峰值保留显存3.17 GiB，CPU参数为0。'
        '这不是三轮固定短句性能验收。\n\n'
        '起点正重新生成同一200条开发输入，核对提示词与模型指纹后进行配对比较。'
        '600条新发布测试、分项约束和长预算验收仍未完成，Goal保持active。\n\n'
        '判定按实际原文而非参考字符串。028/044参考限定有歧义；047中文打蛋器未限定手动，允许电动搅打器读法。\n')
    Path('runs/v5-candidate-acceptance.md').write_text(text, encoding='utf-8')
    print({'verdicts':counts, 'release_approved':False})

if __name__ == '__main__':
    main()
