"""Verify completed candidate evidence; leave comparison/release unproven."""
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from witrans_tools.common import fingerprint, now, read_jsonl, write_json

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    selection = load('runs/v4-selection.json')
    selected = selection['selected']
    directory = Path(selected['directory'])
    config = load('models/witrans-4b-v4-sft/run_config.json')
    for name in ('train', 'dev'):
        if fingerprint(read_jsonl(f'models/witrans-4b-v4-sft/{name}_snapshot.jsonl')) != config[f'{name}_hash']:
            raise ValueError('Actual training snapshot changed')
    if hashlib.sha256((directory/'adapter_model.safetensors').read_bytes()).hexdigest() != selected['adapter_sha256']:
        raise ValueError('Selected weights changed')
    raw = {r['id']:r for r in read_jsonl('runs/v4-selected-dev.jsonl')}
    decisions = read_jsonl('runs/v4-selected-dev-semantic.jsonl')
    if len(decisions) != 200 or len(raw) != 200 or {r['id'] for r in decisions} != set(raw):
        raise ValueError('Incomplete candidate semantic inspection')
    for decision in decisions:
        row = raw[decision['id']]
        if decision['output_hash'] != fingerprint({k: row[k] for k in ('input', 'raw', 'reference')}):
            raise ValueError('Semantic decision no longer matches generation')
    buckets = defaultdict(Counter)
    for d in decisions:
        buckets[f"{d['category']}/{d['target_lang']}"][d['verdict']] += 1
    counts = dict(Counter(d['verdict'] for d in decisions))
    record = {'at': now(), 'stage': 'candidate_development_review_complete; starting_comparison_pending',
        'selected': selected, 'verdicts': counts, 'pass_fraction': counts.get('pass',0)/200,
        'buckets': dict(buckets), 'generation': load('runs/v4-selected-dev.summary.json'),
        'snapshots_and_selected_weights_verified': True, 'default_promoted': False, 'release_approved': False,
        'limitations': ['200 development rows/100 source groups, not the600-row release test',
            'Starting comparison incomplete; cannot claim relative improvement',
            'Source028 reference over-specifies train and044 Chinese legume name is ambiguous; judged actual source, not exact reference',
            'Single generation pass on this dev set is not the fixed three-run performance gate']}
    write_json('runs/v4-candidate-acceptance.json', record)
    metadata = load(directory/'witrans_adapter.json')
    metadata.update(quality_status='Experimental; development semantic quality insufficient for release', acceptance_report='runs/v4-candidate-acceptance.json')
    write_json(directory/'witrans_adapter.json', metadata)
    Path('runs/v4-candidate-acceptance.md').write_text(
        '# v4候选开发审核\n\n200条/100组新开发译文全部经Codex逐条阅读，139通过、27轻微、34严重；JSON200/200。通过率69.5%，尚不能进入发布验收。\n\n'
        '起点对照运行中断后已保留单条输出与诊断，重新进行完整对照。配对差异未完成，不宣称质量提升。\n\n'
        '候选平均3.60秒、P95 5.24秒、4.77实际输出token/s；峰值已分配2.87 GiB、保留3.17 GiB。这是单轮开发集性能，不是三轮固定短句门槛。\n\n'
        '实际训练快照、所选权重与审核输出绑定已核对。默认权重未替换；600条新发布测试、分项门槛、基线比较、置信区间与长预算/三轮性能仍未全部完成，Goal保持active。\n\n'
        '参考质量问题：028的英文connection未明确列车，044中文扁豆未明确lentils；逐条判定依据实际原文，不要求与参考字符串一致。旧冻结文件保留，不因候选输出改写试题。\n', encoding='utf-8')
    print({'verdicts': counts, 'pass_fraction': record['pass_fraction'], 'release_approved': False})

if __name__ == '__main__':
    main()
