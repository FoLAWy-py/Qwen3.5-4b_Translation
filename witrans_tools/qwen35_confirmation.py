"""Numeric independent-confirmation gates; never proof of source or freeze integrity."""
from collections import Counter

from witrans_tools.paired_stats import paired_pass_interval
from witrans_tools.qwen35_stage_gates import counts


def evaluate_confirmation(candidate, v7, frozen_qwen35, *, iterations=10000, seed=20261002):
    """Caller must separately prove frozen weights, source isolation and bound reviews."""
    if iterations<5000:
        raise ValueError('At least5000 predeclared source-group bootstrap iterations required')
    reviews=(candidate,v7,frozen_qwen35)
    indices=[]
    for rows in reviews:
        index={row['id']:row for row in rows}
        if not rows or len(index)!=len(rows):raise ValueError('Empty or duplicate review IDs')
        for row in rows:
            if row['verdict'] not in ('pass','minor','major','critical'):
                raise ValueError('Unreviewed verdict')
            if row['category'] not in ('daily','travel','food','academic','hard') or row['target_lang'] not in ('en','zh-CN'):
                raise ValueError('Unknown category/direction')
            if not isinstance(row['group_id'],str) or not row['group_id']:
                raise ValueError('Source group required')
            if any(type(row[key]) is not bool for key in ('format_valid','language_correct','ended')):
                raise ValueError('Explicit protocol review required')
        indices.append(index)
    if not (set(indices[0])==set(indices[1])==set(indices[2])):
        raise ValueError('Three-way review IDs differ')
    for rid,current in indices[0].items():
        for prior in (indices[1][rid],indices[2][rid]):
            if any(current[key]!=prior[key] for key in ('group_id','category','target_lang')):
                raise ValueError('Paired source/category/direction identity mismatch')
    # One source family cannot masquerade as several categories to meet coverage.
    categories_by_group={}
    for row in candidate:
        categories_by_group.setdefault(row['group_id'],set()).add(row['category'])
    if any(len(value)!=1 for value in categories_by_group.values()):
        raise ValueError('A source family must keep one category')
    cc,vc,fc=[counts(rows) for rows in reviews]
    comparisons={};strata={}
    for role,baseline in (('v7',v7),('frozen_qwen35',frozen_qwen35)):
        paired=paired_pass_interval(candidate,baseline,iterations=iterations,seed=seed)
        before={r['id']:r for r in baseline}
        comparisons[role]=dict(interval=paired,baseline_counts=counts(baseline),
            transition_counts=dict(Counter(before[r['id']]['verdict']+' -> '+r['verdict'] for r in candidate)))
    for category in ('daily','travel','food','academic','hard'):
        for direction in ('en','zh-CN'):
            subset=lambda rows:[r for r in rows if r['category']==category and r['target_lang']==direction]
            current=subset(candidate)
            strata[category+'/'+direction]=dict(rows=len(current),source_groups=len({r['group_id'] for r in current}),
                candidate_counts=counts(current),comparisons={})
            if current:
                for role,baseline in (('v7',v7),('frozen_qwen35',frozen_qwen35)):
                    prior=subset(baseline);before={r['id']:r for r in prior}
                    strata[category+'/'+direction]['comparisons'][role]=dict(baseline_counts=counts(prior),
                        interval=paired_pass_interval(current,prior,iterations=iterations,seed=seed),
                        transition_counts=dict(Counter(before[r['id']]['verdict']+' -> '+r['verdict'] for r in current)))
    delta=cc['pass_count']-vc['pass_count']
    gates=dict(rows_at_least400=len(candidate)>=400,source_groups_at_least200=len(categories_by_group)>=200,
        all_category_direction_rows_at_least40=all(s['rows']>=40 for s in strata.values()),
        all_category_direction_source_groups_at_least40=all(s['source_groups']>=40 for s in strata.values()),
        # Integer comparison avoids rounding a3pp boundary down by floating point.
        pass_gain_vs_v7_at_least3pp=100*delta>=3*len(candidate),
        paired95_lower_vs_v7_above_zero=comparisons['v7']['interval']['group_bootstrap_95_interval'][0]>0,
        major_nonincrease_vs_v7=cc['major_including_critical']<=vc['major_including_critical'],
        pass_nondecrease_vs_qwen35=cc['pass_count']>=fc['pass_count'],
        major_nonincrease_vs_qwen35=cc['major_including_critical']<=fc['major_including_critical'],
        candidate_critical_zero=cc['critical']==0,
        candidate_all_json_direction_eos=all(all(r[key] for key in ('format_valid','language_correct','ended')) for r in candidate))
    return dict(counts=dict(candidate=cc,v7=vc,frozen_qwen35=fc),gates=gates,
        numeric_confirmation_passed=all(gates.values()),comparisons=comparisons,strata=strata,
        baseline_protocol_counts={role:{key:sum(r[key] for r in rows) for key in ('format_valid','language_correct','ended')}
            for role,rows in (('v7',v7),('frozen_qwen35',frozen_qwen35))},
        statistics=dict(unit='paired_source_group',iterations=iterations,seed=seed),
        scope='Numeric confirmation only. Caller must validate source/license/context/ambiguity before generation, isolation from all prior sources, frozen runtime/weights, content-bound AI review and own-weight resource/performance gates. This calculation does not authorize release or default promotion.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
