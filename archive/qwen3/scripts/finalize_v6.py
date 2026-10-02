"""Require complete content-bound semantic evidence for both trials; never release."""
import hashlib
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.paired_stats import paired_pass_interval
from witrans_tools.review_reuse import indexed

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-only', choices=('cpo','sft_control','critical_cpo','sft'))
    parser.add_argument('--plan', default='runs/v6-evaluation-plan.json')
    args = parser.parse_args()
    plan = load(args.plan)
    expected_roles = plan.get('candidate_roles', ['cpo','sft_control'])
    candidate_roles = [args.candidate_only] if args.candidate_only else expected_roles
    refs = indexed(read_jsonl(plan['input']))
    if fingerprint(list(refs.values())) != plan['development_hash']:
        raise ValueError('Frozen development changed')
    reviews, summaries = {}, {}
    evidence = plan.get('evidence', {'start':'v5-selected','cpo':'v6-cpo','sft_control':'v6-sft-control'})
    controls = plan.get('comparison_controls',{})
    evidence = {**evidence,**{role:spec['stem'] for role,spec in controls.items()}}
    for role, stem in evidence.items():
        if role != 'start' and role not in candidate_roles and role not in controls:
            continue
        raw = indexed(read_jsonl(f'runs/{stem}-dev.jsonl'))
        reviewed = indexed(read_jsonl(f'runs/{stem}-dev-semantic.jsonl'))
        if set(raw) != set(refs) or set(reviewed) != set(refs):
            raise ValueError('Incomplete individual semantic evidence')
        for key, row in raw.items():
            decision = reviewed[key]
            if (row['input'] != refs[key]['input'] or row['reference'] != refs[key]['output']
                    or decision.get('output_hash') != fingerprint({k:row[k] for k in ('input','raw','reference')})
                    or decision.get('verdict') not in ('pass','minor','major','critical')
                    or decision.get('reviewer') != 'Codex' or not decision.get('note')
                    or decision.get('format_valid') != ('prediction' in row) or decision.get('ended') != row.get('ended')
                    or decision.get('group_id') != refs[key]['group_id'] or decision.get('category') != refs[key]['category']
                    or decision.get('target_lang') != refs[key]['input']['target_lang']
                    or type(decision.get('language_correct')) is not bool):
                raise ValueError('Semantic judgment does not bind current output')
        summary = load(f'runs/{stem}-dev.summary.json')
        sha = (plan['starting_adapter_sha256'] if role == 'start' else
               controls[role]['adapter_sha256'] if role in controls else plan['candidates'][role]['adapter_sha256'])
        if (summary['adapter_sha256'] != sha or summary['data_hash'] != plan['development_hash']
                or summary['quantization'] != plan['quantization'] or summary['decoding'] != plan['decoding']
                or summary['prompt_hash'] != plan['prompt_hash']):
            raise ValueError('Generation protocol differs')
        if role in controls:
            if sha is not None or summary['adapter_dir'] is not None or summary['model']!='Qwen3-4B baseline':
                raise ValueError('Original-base control unexpectedly has adapter weights')
            if summary['base']!=summaries['start']['base']:
                raise ValueError('Original-base revision differs from starting model')
        if role in candidate_roles:
            directory = Path(plan['candidates'][role]['directory'])
            if hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest() != sha:
                raise ValueError('Candidate weights changed')
        summaries[role], reviews[role] = summary, list(reviewed.values())
    counts = {role:dict(Counter(d['verdict'] for d in rows)) for role,rows in reviews.items()}
    strata = {role:defaultdict(Counter) for role in reviews}
    for role, rows in reviews.items():
        for d in rows:
            strata[role][f"{d['category']}/{d['target_lang']}"][d['verdict']] += 1
    major = {role:c.get('major',0)+c.get('critical',0) for role,c in counts.items()}
    comparisons, gates = {}, {}
    for role in candidate_roles:
        stats = paired_pass_interval(reviews[role], reviews['start'])
        changes = {key:(strata[role][key]['pass']-baseline['pass'])/sum(baseline.values())
            for key,baseline in strata['start'].items()}
        comparisons[role] = {'paired_stats':stats, 'stratum_pass_changes':changes}
        gates[role] = (stats['pass_difference'] >= .03 and major[role] <= major['start']
            and all(change >= -.02 for change in changes.values())
            and summaries[role]['buckets']['all']['valid_json'] >= 198)
        for control in controls:
            control_stats = paired_pass_interval(reviews[role],reviews[control])
            control_changes = {key:(strata[role][key]['pass']-baseline['pass'])/sum(baseline.values())
                for key,baseline in strata[control].items()}
            control_gate = (control_stats['pass_difference']>=0 and major[role]<=major[control]
                            and all(change>=-.02 for change in control_changes.values()))
            comparisons[role+'_vs_'+control] = {'paired_stats':control_stats,
                'stratum_pass_changes':control_changes,'not_worse_than_control':control_gate}
            gates[role] = gates[role] and control_gate
    if 'cpo' in reviews and 'sft_control' in reviews:
        comparisons['cpo_vs_sft_control'] = {'paired_stats':paired_pass_interval(reviews['cpo'], reviews['sft_control'])}
    if 'critical_cpo' in reviews:
        for control in ('cpo','sft_control'):
            if control in reviews:
                comparisons['critical_cpo_vs_'+control] = {'paired_stats':paired_pass_interval(reviews['critical_cpo'],reviews[control])}
    report = {'at':now(), 'counts':counts, 'major_including_critical':major,
        'comparison_complete':set(candidate_roles)==set(expected_roles),
        'comparisons':comparisons, 'development_promotion_gates':gates,
        'performance':summaries, 'default_promoted':False, 'release_approved':False,
        'release_test_generated':False,
        'scope':plan.get('acceptance_scope','Repeatedly used200-row/100-group development only. Cached v5 start generation has matching weights and protocol, no fresh matched timing claim. No efficacy or innovation claim from one seed.'),
        'next':plan.get('report_next','New release test only after semantic development screening succeeds; otherwise improve training. Full goal requires600 new test rows, source audit, three-model comparisons, budget and3-round performance.')}
    report_stem = f"{plan.get('report_prefix','runs/v6')}-{args.candidate_only}-candidate-acceptance" if args.candidate_only else plan.get('report_stem','runs/v6-model-acceptance')
    write_json(report_stem+'.json', report)
    for role in candidate_roles:
        if role not in plan.get('metadata_update_roles',candidate_roles):
            continue
        metadata_path = Path(plan['candidates'][role]['directory']) / 'witrans_adapter.json'
        metadata = load(metadata_path)
        metadata.update(quality_status=f"Experimental; known-development pass{counts[role].get('pass',0)/len(refs):.1%}; development gate {'passed' if gates[role] else 'rejected'}; release unproven",
            acceptance_report=report_stem+'.json')
        write_json(metadata_path, metadata)
    lines = [plan.get('report_title','# v6偏好训练与SFT完整开发验收'), '',
        plan.get('report_training_description','85个训练偏好对：60个作者构造语义对照、25个真实模型负例；同一v5起点、NF4、22次更新。'), '',
        '| 模型 | pass | minor | major含critical | JSON |', '|---|---:|---:|---:|---:|']
    for role in ['start',*controls,*candidate_roles]:
        c = counts[role]
        lines.append(f"| {role} | {c.get('pass',0)} | {c.get('minor',0)} | {major[role]} | {summaries[role]['buckets']['all']['valid_json']}/200 |")
    for role in candidate_roles:
        s = comparisons[role]['paired_stats']
        interval = s['group_bootstrap_95_interval']
        label = plan.get('starting_label','v5起点')
        lines += ['', f"{role}相对{label}：通过率{s['pass_difference']*100:+.1f}个百分点，组级95%区间[{interval[0]*100:.1f}, {interval[1]*100:.1f}]；开发晋级{'通过' if gates[role] else '未通过'}。"]
    lines += ['', '已知开发集不能证明发布质量。默认权重未替换，完整Goal未完成；未运行600条新发布测试。', '']
    if args.candidate_only:
        lines[0] = f'# v6 {args.candidate_only}单候选开发验收'
        lines += ['此文件只包含一个候选相对起点，CPO与SFT完整对照仍待完成。', '']
    Path(report_stem+'.md').write_text('\n'.join(lines), encoding='utf-8')
    print({'counts':counts, 'development_promotion_gates':gates, 'comparisons':comparisons})

if __name__ == '__main__':
    main()
