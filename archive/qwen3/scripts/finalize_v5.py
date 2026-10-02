"""Complete matched fresh-generation development comparison, never promotion."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from archive.qwen3.scripts.finalize_v5_candidate import main as verify_candidate
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.paired_stats import paired_pass_interval

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    verify_candidate()
    plan = load('runs/v5-evaluation-plan.json')
    reviews, summaries, raw = {}, {}, {}
    for role in ('selected','start'):
        reviews[role] = read_jsonl(f'runs/v5-{role}-dev-semantic.jsonl')
        outputs = read_jsonl(f'runs/v5-{role}-dev.jsonl')
        raw[role] = {r['id']:r for r in outputs}
        summaries[role] = load(f'runs/v5-{role}-dev.summary.json')
        if len(reviews[role]) != 200 or len(outputs) != 200 or len(raw[role]) != 200 or {d['id'] for d in reviews[role]} != set(raw[role]):
            raise ValueError('Incomplete matched evidence')
        for d in reviews[role]:
            row = raw[role][d['id']]
            if d['output_hash'] != fingerprint({k:row[k] for k in ('input','raw','reference')}):
                raise ValueError('Review/output content mismatch')
        summary = summaries[role]
        expected_sha = plan['selected']['adapter_sha256'] if role == 'selected' else plan['starting_adapter_sha256']
        if summary['adapter_sha256'] != expected_sha or summary['prompt_hash'] != plan['prompt_hash'] or summary['data_hash'] != plan['development_hash'] or summary['decoding'] != plan['decoding'] or summary['quantization'] != 'nf4':
            raise ValueError('Different model/protocol/settings')
    counts = {role:dict(Counter(d['verdict'] for d in decisions)) for role,decisions in reviews.items()}
    stats = paired_pass_interval(reviews['selected'], reviews['start'])
    strata = {role:defaultdict(Counter) for role in reviews}
    for role, decisions in reviews.items():
        for d in decisions:
            strata[role][f"{d['category']}/{d['target_lang']}"][d['verdict']] += 1
    changes = {key:(strata['selected'][key]['pass']-strata['start'][key]['pass'])/sum(strata['start'][key].values()) for key in strata['start']}
    major = {role:counts[role].get('major',0)+counts[role].get('critical',0) for role in counts}
    gate = stats['pass_difference'] >= .03 and major['selected'] <= major['start'] and all(value >= -.02 for value in changes.values()) and summaries['selected']['buckets']['all']['valid_json'] >= 198
    reuse = load('runs/v5-start-dev-semantic.summary.json')
    report = {'at':now(), 'status':'development_screen_passed' if gate else 'experimental_rejected',
        'development_promotion_gate':gate, 'counts':counts, 'paired_stats':stats, 'stratum_pass_changes':changes,
        'performance':summaries, 'starting_review_reuse':reuse, 'default_promoted':False, 'release_approved':False,
        'release_test_generated':False, 'conclusion':'No release claim. Known development answer NLL cannot replace semantic gates.'}
    write_json('runs/v5-model-acceptance.json', report)
    partial = load('runs/v5-candidate-acceptance.json')
    partial['stage'] = 'candidate_and_starting_development_comparison_complete'
    partial['limitations'][1] = 'Complete paired evidence in runs/v5-model-acceptance.json; development promotion passed' if gate else 'Complete paired evidence in runs/v5-model-acceptance.json; development promotion failed'
    write_json('runs/v5-candidate-acceptance.json', partial)
    candidate_md = Path('runs/v5-candidate-acceptance.md')
    candidate_md.write_text(candidate_md.read_text(encoding='utf-8').replace(
        '起点正重新生成同一200条开发输入，核对提示词与模型指纹后进行配对比较。',
        '完整起点配对比较已完成，见[v5完整开发验收](v5-model-acceptance.md)。当前两个模型通过数相同，未通过开发晋级。'), encoding='utf-8')
    interval = stats['group_bootstrap_95_interval']
    lines = ['# v5完整开发对照验收', '', 'NF4，200条已知开发输入、100来源组，候选逐条新读；起点重新生成，并仅对与已读译文完全一致的内容复用审核。', '',
        '| 模型 | 通过 | 轻微 | 严重 | JSON |', '|---|---:|---:|---:|---:|']
    for role,label in (('start','v2起点重新生成'), ('selected','v5所选checkpoint85')):
        c = counts[role]
        lines.append(f"| {label} | {c.get('pass',0)} | {c.get('minor',0)} | {major[role]} | {summaries[role]['buckets']['all']['valid_json']}/200 |")
    lines += ['', f"配对通过率变化{stats['pass_difference']*100:+.1f}个百分点，按来源组bootstrap的95%区间[{interval[0]*100:.1f}, {interval[1]*100:.1f}]个百分点。",
        '', f"开发晋级：{'通过' if gate else '未通过'}。默认权重未替换，600条新发布测试和完整Goal仍未完成。",
        '', '当前两个模型同样的输出使用同样的判定。对相同旧输出补充发现的判定差异记录在JSON，历史v4报告保持原记录，不混合两次性能。',
        '', '开发NLL降低至0.66957，尚不能证明发布质量合格。单轮开发时延不能代替固定27条、预热后三轮的性能门槛。', '']
    Path('runs/v5-model-acceptance.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'counts':counts, 'paired_stats':stats, 'development_promotion_gate':gate, 'same_output_judgment_changes':reuse['current_same_output_judgment_changes']})

if __name__ == '__main__':
    main()
