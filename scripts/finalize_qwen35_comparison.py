"""Accept only full source-bound reviews of real matched new-model outputs."""
import hashlib
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path
from archive.qwen3.scripts.finalize_v13_optimization import complete_review
from witrans_tools.common import fingerprint,now,read_jsonl,write_json
from witrans_tools.paired_stats import paired_pass_interval


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    dest=Path('runs/qwen35-v2-model-comparison.json')
    assert not dest.exists(), 'Preserve semantic conclusion'
    plan=load('data/prepared/qwen35-v2/plan.json')
    candidate=Path(plan['cpo']['output'])
    with (candidate/'adapter_model.safetensors').open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    metadata=load(candidate/'witrans_adapter.json')
    assert sha==metadata['adapter_sha256'] and metadata['plan_hash']==fingerprint(plan)
    sections={}
    for split,spec in plan['datasets'].items():
        reviewed={};summaries={}
        for role in ('base','finetuned'):
            stem=f'runs/qwen35-base-v2-{split}' if role=='base' else f'runs/qwen35-finetuned-{split}'
            rows,decisions=complete_review(stem,spec)
            summary=load(stem+'.summary.json')
            assert summary['base']==plan['base'] and summary['prompt_hash']==plan['prompt_hash']
            assert summary['quantization']=='nf4' and summary['data_hash']==spec['hash']
            assert summary['generation_hash']==fingerprint(rows)
            assert summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
            assert summary['adapter_sha256']==(sha if role=='finetuned' else None)
            reviewed[role]=decisions;summaries[role]=summary
        counts={role:dict(Counter(d['verdict'] for d in decisions)) for role,decisions in reviewed.items()}
        strata={role:defaultdict(Counter) for role in reviewed}
        for role,decisions in reviewed.items():
            for d in decisions:
                strata[role][d['category']+'/'+d['target_lang']][d['verdict']]+=1
        changes={key:(strata['finetuned'][key]['pass']-c['pass'])/sum(c.values()) for key,c in strata['base'].items()}
        metrics={role:dict(pass_count=c.get('pass',0),major_including_critical=c.get('major',0)+c.get('critical',0),
            critical=c.get('critical',0),rows=len(reviewed[role]),pass_rate=c.get('pass',0)/len(reviewed[role]))
            for role,c in counts.items()}
        transitions=Counter((b['verdict'],c['verdict']) for b,c in zip(
            sorted(reviewed['base'],key=lambda r:r['id']),sorted(reviewed['finetuned'],key=lambda r:r['id'])))
        structure={role:all(d['format_valid'] and d['ended'] and d['language_correct'] for d in decisions)
                   for role,decisions in reviewed.items()}
        structure_counts={role:{key:sum(d[key] for d in decisions)
            for key in ('format_valid','ended','language_correct')} for role,decisions in reviewed.items()}
        for role,decisions in reviewed.items():
            metrics[role]['pass_with_all_protocol_constraints']=sum(d['verdict']=='pass' and
                d['format_valid'] and d['ended'] and d['language_correct'] for d in decisions)
        stats=paired_pass_interval(reviewed['finetuned'],reviewed['base'])
        paired_examples=[]
        raw_by_role={}
        for role in ('base','finetuned'):
            stem=f'runs/qwen35-base-v2-{split}' if role=='base' else f'runs/qwen35-finetuned-{split}'
            raw_by_role[role]={r['id']:r for r in read_jsonl(stem+'.jsonl')}
        notes_by_role={role:{d['id']:d for d in decisions} for role,decisions in reviewed.items()}
        for rid,b in notes_by_role['base'].items():
            c=notes_by_role['finetuned'][rid]
            if b['verdict']==c['verdict']: continue
            rb,rc=raw_by_role['base'][rid],raw_by_role['finetuned'][rid]
            paired_examples.append(dict(id=rid,category=b['category'],target_lang=b['target_lang'],
                input=rb['input'],reference=rb['reference'],before=b['verdict'],after=c['verdict'],
                base=rb.get('prediction',{}).get('translation',rb.get('raw')),
                finetuned=rc.get('prediction',{}).get('translation',rc.get('raw')),
                base_note=b['note'],finetuned_note=c['note']))
        timings={role:dict(mean_seconds=statistics.mean(r['seconds'] for r in rows.values()),
            median_seconds=statistics.median(r['seconds'] for r in rows.values())) for role,rows in raw_by_role.items()}
        sections[split]=dict(counts=counts,metrics=metrics,source_group_paired_stats=stats,
            strata={role:{k:dict(v) for k,v in values.items()} for role,values in strata.items()},
            stratum_rows={k:sum(v.values()) for k,v in strata['base'].items()},stratum_pass_changes=changes,
            transitions=[dict(before=b,after=a,count=n) for (b,a),n in transitions.items()],
            structure_direction_eos=structure,protocol_counts=structure_counts,runtime=summaries,development_generation_timings=timings,
            changed_verdict_examples=paired_examples)
        if split=='known':
            sensitivity=load('runs/qwen35-source-ambiguity-sensitivity-plan.json')
            excluded=set(sensitivity['excluded_known_groups'])
            filtered={role:[d for d in decisions if d['group_id'] not in excluded]
                      for role,decisions in reviewed.items()}
            sections[split]['source_ambiguity_sensitivity']=dict(plan=sensitivity,
                counts={role:dict(Counter(d['verdict'] for d in ds)) for role,ds in filtered.items()},
                paired_stats=paired_pass_interval(filtered['finetuned'],filtered['base']))
    known,public=sections['known'],sections['public']
    gates=dict(known_pass_gain_at_least3pp=known['source_group_paired_stats']['pass_difference']>=.03,
        known_major_nonincrease=known['metrics']['finetuned']['major_including_critical']<=known['metrics']['base']['major_including_critical'],
        category_direction_regression_at_most2pp=all(v>=-.02 for v in known['stratum_pass_changes'].values()),
        public_pass_regression_at_most2pp=public['source_group_paired_stats']['pass_difference']>=-.02,
        public_major_nonincrease=public['metrics']['finetuned']['major_including_critical']<=public['metrics']['base']['major_including_critical'],
        all_json_direction_eos=all(all(s['structure_direction_eos'].values()) for s in sections.values()),
        candidate_critical_zero=all(s['metrics']['finetuned']['critical']==0 for s in sections.values()))
    report=dict(at=now(),plan_hash=fingerprint(plan),base_revision=plan['base']['revision'],candidate_sha256=sha,
        recipe=plan['method'],sections=sections,development_gates_vs_own_base=gates,
        fine_tuning_screen_passed=all(gates.values()),
        legacy_v7_context=dict(known=dict(pass_count=140,minor=31,major=29,rows=200),
            public=dict(pass_count=95,minor=18,major=3,rows=116),
            scope='Historical current-best same input/prompt/NF4 reference; different model architecture/tokenizer/training lineage, not an equal-total-training method ablation.'),
        excluded_failed_generation_evidence=['runs/qwen35-base-known.jsonl','runs/qwen35-base-public.jsonl'],
        exclusions_reason='All initial base rows failed removed Transformers generation API before any raw output. Preserved as interface failure, not semantic model accuracy.',
        limitations=plan['limitations'],scope='Repeated known200 and auxiliary116. Authorized AI source review, not independent confirmation or human professional validation. One concrete candidate; no stable method/algorithm claim.',
        default_promoted=False,stage_goal_complete=False,release_approved=False)
    benchmark=load('runs/qwen35-v2-short-benchmark.json')
    assert benchmark['plan_hash']==fingerprint(plan)
    assert benchmark['summaries']['finetuned']['adapter_sha256']==sha
    assert benchmark['summaries']['base']['adapter_sha256'] is None
    report['own_weight_short_benchmark']=benchmark
    write_json(dest,report)
    lines=['# Qwen3.5-4B：项目微调前后语义比较','',
        '“base”为用户链接中的官方后训练检查点，尚未做本项目微调。两边NF4、关闭thinking、同提示词、1024总token预算、贪心解码。','',
        '| 集合 | 模型 | pass | minor | major含critical | critical | JSON/方向/EOS全正确 |',
        '|---|---|---:|---:|---:|---:|---|']
    for split,section in sections.items():
        for role in ('base','finetuned'):
            c=section['counts'][role];m=section['metrics'][role]
            lines.append(f"| {split} | {role} | {m['pass_count']} | {c.get('minor',0)} | {m['major_including_critical']} | {m['critical']} | {section['structure_direction_eos'][role]} |")
    for split,section in sections.items():
        stats=section['source_group_paired_stats'];ci=stats['group_bootstrap_95_interval']
        lines.extend(['',f"{split} 微调相对base：pass变化{stats['pass_difference']*100:+.2f}个百分点，来源组配对95%区间[{ci[0]*100:.2f}, {ci[1]*100:.2f}]；{stats['rows']}条/{stats['source_groups']}组。",''])
        for role in ('base','finetuned'):
            protocol=section['protocol_counts'][role];m=section['metrics'][role]
            lines.append(f"{split}/{role}：合法JSON {protocol['format_valid']}/{m['rows']}，语言方向 {protocol['language_correct']}/{m['rows']}，EOS {protocol['ended']}/{m['rows']}；语义pass且同时满足全部协议 {m['pass_with_all_protocol_constraints']}/{m['rows']}。")
        lines.append('')
    lines.extend(['pass/minor/major仅评翻译语义；格式独立严格检查。非法JSON中的可读译文可语义合格，但不能直接被接口消费；未修复或重写原始模型输出。','',
        '| 已知集类别×方向 | 样本数 | base pass | 微调pass | pass变化 |',
        '|---|---:|---:|---:|---:|'])
    for key,count in sorted(known['stratum_rows'].items()):
        before=known['strata']['base'][key].get('pass',0)
        after=known['strata']['finetuned'][key].get('pass',0)
        lines.append(f'| {key} | {count} | {before} | {after} | {100*(after-before)/count:+.1f}个百分点 |')
    sensitivity=known['source_ambiguity_sensitivity']['paired_stats']
    interval=sensitivity['group_bootstrap_95_interval']
    lines.extend(['',f"剔除预先标注的3个歧义来源组（双向6条）后：{sensitivity['rows']}条/{sensitivity['source_groups']}组，pass变化{sensitivity['pass_difference']*100:+.2f}个百分点，95%区间[{interval[0]*100:.2f}, {interval[1]*100:.2f}]。此分析不替代主评分，也不用于调参。",''])
    lines.extend([f"开发筛选{'通过' if all(gates.values()) else '未通过'}。各类别×方向样本数及退步、具体转移和真实生成路径见JSON报告。",'',
        '数据及预算在生成前冻结；微调为678条监督翻译SFT85更新，然后85偏好对的关键片段加权CPO22更新，固定最终检查点。',
        '旧v7 adapter不可跨架构迁移，此次未复现全部历史前置训练。开发集被重复使用；尚无全新400条独立确认或95%发布质量结论。',
        '初次base接口失败记录被完整保留并排除语义计分；本报告只使用修正API后的真实译文。',''])
    for role,summary in benchmark['summaries'].items():
        lines.append(f"{role} 自身权重短句三轮（18次，预热后）：平均{summary['mean_seconds']:.2f}秒，P95 {summary['p95_seconds']:.2f}秒，峰值保留显存{summary['peak_reserved_gib']:.2f} GiB，CPU参数{summary['cpu_parameter_count']}。")
    lines.extend(['',
        '本次判断：微调有具体工程收益，尤其是单字段JSON遵循及部分语义修复；已知集配对区间仍包含零，公开辅助集pass无净增，尚不能证明跨来源稳定语义提升。',
        '原始模型与微调模型均有锉刀句的普通英文词未译；当前微调候选短句平均耗时未达到4秒门槛。保留为下一轮优化候选，不自动替换现有默认模型，不视为阶段Goal完成或发布通过。',
        '所有结果限定于本项目提示词、NF4、关闭thinking和贪心解码，不能推广为官方模型在全部推理配置上的质量排名。',
        '完整真实输出、内容绑定评分、逐项变化、权重指纹、运行配置和失败证据路径均在JSON报告中。',''])
    dest.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'counts':{k:s['counts'] for k,s in sections.items()},'gates':gates,'fine_tuning_screen_passed':all(gates.values())})


if __name__=='__main__': main()
