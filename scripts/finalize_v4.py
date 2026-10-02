"""Complete paired development evidence; no promotion or release claim."""
import json
from pathlib import Path
from collections import Counter, defaultdict
from scripts.finalize_v4_candidate import main as verify_candidate
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.paired_stats import paired_pass_interval

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    verify_candidate()
    reviews = {role: read_jsonl(f'runs/v4-{role}-dev-semantic.jsonl') for role in ('selected','start')}
    summaries = {role: load(f'runs/v4-{role}-dev.summary.json') for role in reviews}
    for role, decisions in reviews.items():
        raw = {r['id']:r for r in read_jsonl(f'runs/v4-{role}-dev.jsonl')}
        if len(decisions) != 200 or len(raw) != 200 or {d['id'] for d in decisions} != set(raw):
            raise ValueError('Incomplete paired evidence')
        for d in decisions:
            if d['output_hash'] != fingerprint({k:raw[d['id']][k] for k in ('input','raw','reference')}):
                raise ValueError('Output/review mismatch')
    if summaries['selected']['data_hash'] != summaries['start']['data_hash'] or summaries['start']['adapter_sha256'] != 'a2dfb3bc414146d9048fcb27231d4c052c1760513290394d77017defe1b73c68':
        raise ValueError('Starting model or paired corpus mismatch')
    counts = {role: dict(Counter(d['verdict'] for d in rows)) for role, rows in reviews.items()}
    stats = paired_pass_interval(reviews['selected'], reviews['start'])
    buckets = {role: defaultdict(Counter) for role in reviews}
    for role, rows in reviews.items():
        for d in rows:
            buckets[role][f"{d['category']}/{d['target_lang']}"][d['verdict']] += 1
    regression = {key: (buckets['selected'][key]['pass'] - buckets['start'][key]['pass'])/20 for key in buckets['start']}
    gate = stats['pass_difference'] >= .03 and counts['selected'].get('major',0) <= counts['start'].get('major',0) and summaries['selected']['buckets']['all']['valid_json'] >= 198 and all(v >= -.02 for v in regression.values())
    record = {'at': now(), 'status': 'experimental_rejected' if not gate else 'development_screen_passed', 'development_promotion_gate': gate,
        'counts': counts, 'paired_stats': stats, 'stratum_pass_changes': regression, 'performance': summaries,
        'default_promoted': False, 'release_approved': False, 'release_test_generated': False,
        'source_review_caveats': '028reference over-specifies train;044Chinese legume name ambiguous. Both roles judged actual input, unchanged frozen references.',
        'conclusion': 'Lower development answer NLL did not establish semantic improvement; broader independent training coverage needed'}
    write_json('runs/v4-model-acceptance.json', record)
    candidate_record = load('runs/v4-candidate-acceptance.json')
    candidate_record['stage'] = 'candidate_and_starting_development_review_complete'
    candidate_record['limitations'][1] = 'Complete starting comparison shows no development promotion; paired evidence in runs/v4-model-acceptance.json'
    write_json('runs/v4-candidate-acceptance.json', candidate_record)
    candidate_path = Path('runs/v4-candidate-acceptance.md')
    candidate_path.write_text(candidate_path.read_text(encoding='utf-8').replace(
        '配对差异未完成，不宣称质量提升。',
        '完整对照现已完成，详见[v4完整验收](v4-model-acceptance.md)，没有建立质量提升证据。'), encoding='utf-8')
    interval = stats['group_bootstrap_95_interval']
    lines = ['# v4完整开发对照验收', '', '新开发200条/100来源组，两个模型各200条译文全部由Codex逐条阅读。', '',
        '| 模型 | 通过 | 轻微 | 严重 | JSON |', '|---|---:|---:|---:|---:|']
    for role, label in (('start','NF4 v2起点'),('selected','v4所选checkpoint31')):
        c = counts[role]
        lines.append(f"| {label} | {c.get('pass',0)} | {c.get('minor',0)} | {c.get('major',0)} | 200/200 |")
    lines += ['', f"配对通过率变化{stats['pass_difference']*100:+.1f}个百分点；按100来源组重采样5000次，95%百分位区间[{interval[0]*100:.1f}, {interval[1]*100:.1f}]个百分点。不能宣称改善。",
        '', '开发loss从0.855降至0.674，但语义通过数略降，未满足提高至少3个百分点等晋级要求。默认权重未替换。', '',
        '这是开发筛选，600条新发布测试与完整Goal验收仍未完成；未把测试范围缩小到本集合。', '',
        '中断的首轮起点单条输出保存在v4-start-dev.interrupted-20261001.jsonl；重新完整生成200条，不混合两次性能。', '',
        '逐条判定依据实际原文与语境，而非参考字符串。028/044参考措辞问题记录保留，未因模型输出改写冻结数据。', '']
    Path('runs/v4-model-acceptance.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'counts': counts, 'paired_stats': stats, 'development_promotion_gate': gate})

if __name__ == '__main__':
    main()
