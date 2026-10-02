"""Require complete candidate/v7 semantic readings; no original-base generation."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from archive.qwen3.scripts.compare_v12_blind import DECISION_HASH
from archive.qwen3.scripts.review_v12_factorial import binding
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.paired_stats import paired_pass_interval


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def complete_review(stem, spec):
    rows = read_jsonl(stem+'.jsonl')
    decisions = read_jsonl(stem+'-semantic.jsonl')
    refs = {row['id']:row for row in read_jsonl(spec['path'])}
    if fingerprint(list(refs.values()))!=spec['hash']:
        raise ValueError('Frozen review references changed')
    indexed = {row['id']:row for row in rows}
    reviewed = {row['id']:row for row in decisions}
    if len(indexed)!=len(rows) or len(reviewed)!=len(decisions) or set(indexed)!=set(refs) or set(reviewed)!=set(refs):
        raise ValueError('Complete individual semantic reading required')
    for key, row in indexed.items():
        decision = reviewed[key]
        if (row['input']!=refs[key]['input'] or row['reference']!=refs[key]['output']
                or decision['binding_hash']!=binding(row) or decision['reviewer']!='Codex'
                or decision['generation_hash']!=fingerprint(row)
                or decision['output_hash']!=fingerprint({k:row[k] for k in ('input','raw','reference')})
                or decision['verdict'] not in ('pass','minor','major','critical') or not decision['note']
                or decision['group_id']!=refs[key]['group_id'] or decision['category']!=refs[key]['category']
                or decision['target_lang']!=refs[key]['input']['target_lang']
                or any(type(decision[k]) is not bool for k in ('format_valid','ended','language_correct'))
                or decision['format_valid']!=('prediction' in row) or decision['ended']!=row.get('ended')):
            raise ValueError('Current output lacks bound semantic acceptance')
    return rows,decisions


def finalize_qwen35_dev(args,plan):
    from collections import Counter
    from scripts.decode_qwen35_repair_recall import verify_training
    from archive.qwen3.scripts.review_v12_factorial import qwen35_review_caches
    from witrans_tools.qwen35_stage_gates import evaluate_stage
    assert args.training_plan, 'Explicit frozen training plan required'
    training=load_json(args.training_plan)
    assert fingerprint(training)==plan['training_plan_hash']
    metrics,sha=verify_training(training)
    selection=load_json(args.selection)
    assert selection['training_plan_hash']==fingerprint(training)
    assert selection['selected']['adapter_sha256']==sha
    screening=load_json(f'runs/{args.prefix}-recall-screening.json')
    assert screening['recall_first_gate_passed'] and screening['adapter_sha256']==sha
    assert screening['plan_hash']==fingerprint(training)
    assert not any(target['baseline'] for target in plan['targets'])
    candidates=[target for target in plan['targets'] if not target.get('cached')]
    assert len(candidates)==2 and {target['split'] for target in candidates}=={'known','public'}
    assert {target['role'] for target in candidates}=={args.prefix}
    statistics=plan['statistics']
    assert statistics['unit']=='paired_source_group' and statistics['iterations']>=5000
    candidate={};raw={};controls={};comparisons={}
    for target in candidates:
        split=target['split'];stem=str(Path(target['output']).with_suffix(''))
        raw[split],candidate[split]=complete_review(stem,plan['datasets'][split])
        summary=load_json(stem+'.summary.json')
        assert summary['generation_hash']==fingerprint(raw[split])
        assert summary['adapter_sha256']==sha==target['adapter_sha256']
        assert summary['base']==training['base'] and summary['prompt_hash']==training['prompt_hash']
        assert summary['quantization']=='nf4' and summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
        assert summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
        caches=qwen35_review_caches(plan,split)
        controls[split]={role:[note for label,note in caches if label==role+'/'+split]
            for role in ('frozen_qwen35','v7')}
        comparisons[split]={}
        for role,baseline in controls[split].items():
            baseline_ids={row['id']:row for row in baseline}
            transitions=[]
            for row in candidate[split]:
                prior=baseline_ids[row['id']]
                assert row['group_id']==prior['group_id'] and row['category']==prior['category']
                assert row['target_lang']==prior['target_lang']
                transitions.append(dict(id=row['id'],group_id=row['group_id'],category=row['category'],
                    target_lang=row['target_lang'],baseline_verdict=prior['verdict'],candidate_verdict=row['verdict'],
                    baseline_direction=prior['language_correct'],candidate_direction=row['language_correct'],
                    same_input_raw_reference_eos_prediction=row['binding_hash']==prior['binding_hash']))
            strata={}
            for category in ('daily','travel','food','academic','hard'):
                for direction in ('en','zh-CN'):
                    filtered=lambda rows:[r for r in rows if r['category']==category and r['target_lang']==direction]
                    current,prior=filtered(candidate[split]),filtered(baseline)
                    if not current:
                        continue
                    strata[category+'/'+direction]=dict(rows=len(current),
                        candidate_counts=dict(Counter(r['verdict'] for r in current)),
                        baseline_counts=dict(Counter(r['verdict'] for r in prior)),
                        interval=paired_pass_interval(current,prior,iterations=statistics['iterations'],seed=statistics['seed']),
                        transitions=dict(Counter(t['baseline_verdict']+' -> '+t['candidate_verdict'] for t in transitions
                            if t['category']==category and t['target_lang']==direction)))
            comparisons[split][role]=dict(interval=paired_pass_interval(candidate[split],baseline,
                iterations=statistics['iterations'],seed=statistics['seed']),
                baseline_counts=dict(Counter(r['verdict'] for r in baseline)),
                candidate_counts=dict(Counter(r['verdict'] for r in candidate[split])),
                transition_counts=dict(Counter(t['baseline_verdict']+' -> '+t['candidate_verdict'] for t in transitions)),
                individual_transitions=transitions,strata=strata)
    gates=evaluate_stage(candidate['known'],candidate['public'],controls['known']['frozen_qwen35'],controls['known']['v7'])
    report=dict(at=now(),training_plan_hash=fingerprint(training),evaluation_plan_hash=fingerprint(plan),
        adapter_sha256=sha,parent_adapter_sha256=training['starting_adapter_sha256'],
        comparisons=comparisons,stage_development=gates,optimization_screen_passed=gates['passed'],
        reviewed_generation_hashes={split:fingerprint(rows) for split,rows in raw.items()},
        reviewed_decision_hashes={split:fingerprint(rows) for split,rows in candidate.items()},
        statistics=statistics,reviewer='Codex AI, user-authorized; no independent professional human review claimed',
        limitations='Repeated known200/public116 development sources. Each known category/direction has20 rows: one pass is5pp. Paired source-group intervals are descriptive and do not establish stable capability or algorithm superiority. One seed; no causal method claim.',
        next='If development passes, still require final own-weight full warmup/three-round timing and resource checks, then candidate/config freeze before new400/200-group independent three-way confirmation. If failed, preserve evidence and distinguish TRAIN learning from unseen-source transfer and category/direction regressions.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(f'runs/{args.prefix}-model-acceptance.json',report)
    print(dict(development_passed=gates['passed'],counts=gates['counts'],gates=gates['gates']),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',default='v13')
    parser.add_argument('--plan',default='runs/v13-evaluation-plan.json')
    parser.add_argument('--selection',default='runs/v13-public-sft-selection.json')
    parser.add_argument('--candidate-role',default='sft',choices=('sft','matching','cpo'))
    parser.add_argument('--training-plan')
    args = parser.parse_args()
    prefix,role = args.prefix,args.candidate_role
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',prefix):
        raise ValueError('Simple workspace report prefix required')
    if Path(f'runs/{prefix}-model-acceptance.json').exists():
        raise ValueError('Preserve finalized semantic evidence')
    plan = load_json(args.plan)
    if plan['base']['model_id']=='Qwen/Qwen3.5-4B':
        finalize_qwen35_dev(args,plan)
        return
    if any(target['baseline'] for target in plan['targets']):
        raise ValueError('User revoked further original-base comparisons')
    candidate_targets = [target for target in plan['targets'] if target['role']!='v7']
    if len(candidate_targets)!=2 or len({target['role'] for target in candidate_targets})!=1:
        raise ValueError('One candidate on known and public splits required')
    stems = {target['split']:str(Path(target['output']).with_suffix('')) for target in candidate_targets}
    control = next(target for target in plan['targets'] if target['role']=='v7' and target['split']=='public')
    public_start_stem = str(Path(control['output']).with_suffix(''))
    known_raw,known_decisions = complete_review(stems['known'],plan['datasets']['known'])
    _,public_candidate = complete_review(stems['public'],plan['datasets']['public'])
    _,public_start = complete_review(public_start_stem,plan['datasets']['public'])
    masked = read_jsonl('runs/v12-blind-decisions.jsonl')
    if fingerprint(masked)!=DECISION_HASH:
        raise ValueError('Frozen v7 masked review changed')
    masked_index = {(row['packet_id'],row['label']):row for row in masked}
    original = read_jsonl('runs/v7-critical-dev.jsonl')
    original_by_id = {row['id']:row for row in original}
    direction_reviews = {row['id']:row for row in read_jsonl('runs/v7-critical-dev-semantic.jsonl')}
    start_decisions = []
    for decision in read_jsonl('runs/v12-blind-v7-semantic.jsonl'):
        frozen = masked_index[(decision['packet_id'],decision['label'])]
        raw = original_by_id[decision['id']]
        direction = direction_reviews[decision['id']]
        digest = fingerprint({k:raw[k] for k in ('input','raw','reference')})
        if (decision['verdict']!=frozen['grade'] or decision['note']!=frozen['note']
                or decision['output_hash']!=digest or direction['output_hash']!=digest):
            raise ValueError('Masked starting review no longer binds original output')
        start_decisions.append({**decision,'reviewer':'Codex','language_correct':direction['language_correct'],
                                'review_source':'frozen_identity_hidden_v7_rereading'})
    selection_record = load_json(args.selection)
    if selection_record.get('training_plan_hash',selection_record.get('plan_hash'))!=plan['training_plan_hash']:
        raise ValueError('Candidate selection belongs to another training plan')
    selection = selection_record['selected']
    start_summary = load_json('runs/v7-critical-dev.summary.json')
    candidate_summary = load_json(stems['known']+'.summary.json')
    public_summaries = [load_json(stem+'.summary.json') for stem in (public_start_stem,stems['public'])]
    if start_summary['adapter_sha256']!=control['adapter_sha256']:
        raise ValueError('Cached parent differs from public control')
    for target in plan['targets']:
        summary = load_json(str(Path(target['output']).with_suffix('.summary.json')))
        if (summary['adapter_sha256']!=target['adapter_sha256']
                or summary['data_hash']!=plan['datasets'][target['split']]['hash']):
            raise ValueError('Reviewed generation weights or data differ from frozen plan')
    for summary in [start_summary,candidate_summary,*public_summaries]:
        if (summary['base']!=plan['base'] or summary['prompt_hash']!=plan['prompt_hash']
                or summary['quantization']!='nf4' or summary['decoding']!={'do_sample':False,'max_length':1024,'max_new_tokens':256}):
            raise ValueError('Frozen comparison protocol differs')
    # Aliases preserve raw content for the existing tested200-row numeric gate.
    write_jsonl(f'runs/{prefix}-start-dev.jsonl',original)
    write_jsonl(f'runs/{prefix}-start-dev-semantic.jsonl',start_decisions)
    write_json(f'runs/{prefix}-start-dev.summary.json',start_summary)
    write_jsonl(f'runs/{prefix}-selected-dev.jsonl',known_raw)
    write_jsonl(f'runs/{prefix}-selected-dev-semantic.jsonl',known_decisions)
    write_json(f'runs/{prefix}-selected-dev.summary.json',candidate_summary)
    acceptance = {'input':plan['datasets']['known']['path'],'development_hash':plan['datasets']['known']['hash'],
                  'candidate_roles':[role],'evidence':{'start':prefix+'-start',role:prefix+'-selected'},
                  'comparison_controls':{},'starting_adapter_sha256':start_summary['adapter_sha256'],
                  'candidates':{role:selection},'quantization':'nf4','prompt_hash':plan['prompt_hash'],
                  'decoding':candidate_summary['decoding'],'report_stem':f'runs/{prefix}-known-screening',
                  'report_title':f'# {prefix}候选优化开发验收','starting_label':'当前最佳v7',
                  'report_training_description':('从v7继续SFT，已审核旧/公开混合数据、lr5e-6、64更新、262144完整输入token；不新跑原始base。'
                    if prefix=='v13' else
                    '从v7继续真实TRAIN错误CPO；11条已审核错误、64偏好访问、128正例回放、lr2e-6、16更新，固定最终检查点。训练修复率另报；不新跑原始base。'
                    if prefix=='v15-error' else
                    '从v7继续事实小试；14既有训练家族/28矩阵、128回放、lr2e-6、16更新，固定最终检查点。完整训练配置见冻结计划；不新跑原始base。'),
                  'acceptance_scope':'Known DEV200 screening only; parent frozen masked rereading, evaluator prior knowledge disclosed. Auxiliary public116 additionally checked. No release or one-seed efficacy claim.',
                  'report_next':'New release test only after semantic screening succeeds; otherwise improve training facts. No further original-base comparison.'}
    if args.training_plan:
        training = load_json(args.training_plan)
        if fingerprint(training) != plan['training_plan_hash']:
            raise ValueError('Training description must bind actual frozen plan')
        acceptance['report_training_description'] = (
            f"从v7继续{training['method']}；{len(read_jsonl(training['pairs_path']))}条真实TRAIN错误、"
            f"{len(read_jsonl(training['replay_path']))}条已审核正例回放、lr{training['learning_rate']}、"
            f"{training['updates']}更新、{training['full_token_budget']}完整输入token；固定最终检查点。"
            "训练修复率另报；未生成原始base对照。")
    write_json(f'runs/{prefix}-acceptance-plan.json',acceptance)
    subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
                    'scripts.finalize_v6','--plan',f'runs/{prefix}-acceptance-plan.json'],check=True)
    report = load_json(f'runs/{prefix}-known-screening.json')
    stats = paired_pass_interval(public_candidate,public_start)
    major = lambda rows:sum(row['verdict'] in ('major','critical') for row in rows)
    auxiliary_ok = stats['pass_difference']>=-.02 and major(public_candidate)<=major(public_start)
    candidate_flags_ok = all(row['format_valid'] and row['ended'] and row['language_correct']
                            for row in known_decisions+public_candidate)
    start_strata, candidate_strata = {}, {}
    for destination, decisions in ((start_strata,start_decisions),(candidate_strata,known_decisions)):
        for row in decisions:
            key = row['category']+'/'+row['target_lang']
            bucket = destination.setdefault(key,dict(rows=0,pass_count=0,major_including_critical=0))
            bucket['rows'] += 1
            bucket['pass_count'] += row['verdict']=='pass'
            bucket['major_including_critical'] += row['verdict'] in ('major','critical')
    stratum_changes = {key:(candidate_strata[key]['pass_count']-s['pass_count'])/s['rows']
                       for key,s in start_strata.items()}
    stage_gates = dict(known_pass_at_least146=sum(d['verdict']=='pass' for d in known_decisions)>=146,
        known_major_at_most29=major(known_decisions)<=29,
        category_direction_regression_at_most2pp=all(v>=-.02 for v in stratum_changes.values()),
        public_regression_at_most2pp=stats['pass_difference']>=-.02,
        public_major_at_most3=major(public_candidate)<=3,
        all_json_direction_eos=candidate_flags_ok)
    report.update(at=now(),public_auxiliary={'paired_stats':stats,'candidate_major':major(public_candidate),
                  'v7_major':major(public_start),'not_materially_worse':auxiliary_ok},
                  optimization_screen_passed=all(stage_gates.values()),stage_development_gates=stage_gates,
                  known_strata={'v7':start_strata,'candidate':candidate_strata,'pass_changes':stratum_changes},
                  public_counts={'v7':{v:sum(d['verdict']==v for d in public_start) for v in ('pass','minor','major','critical')},
                                 'candidate':{v:sum(d['verdict']==v for d in public_candidate) for v in ('pass','minor','major','critical')}},
                  candidate_direction_and_eos_valid=candidate_flags_ok,
                  default_promoted=False,release_approved=False,
                  next='If semantic screening succeeds, verify best candidate on fresh release protocol and runtime; otherwise optimize training facts. No further original-base comparison.')
    write_json(f'runs/{prefix}-model-acceptance.json',report)
    print({'counts':report['counts'],'public_pair_stats':stats,'optimization_screen_passed':report['optimization_screen_passed'],
           'release_approved':False},flush=True)


if __name__ == '__main__':
    main()
