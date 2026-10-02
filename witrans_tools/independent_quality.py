"""Absolute independent-test gates and source-group statistics; no v7 dependency."""
from collections import Counter, defaultdict
import math
import random

from .common import fingerprint


CATEGORIES=('daily','travel','food','academic','hard')
DIRECTIONS=('en','zh-CN')


def counts(rows):
    c=Counter(r['verdict'] for r in rows)
    n=len(rows)
    return dict(rows=n,source_groups=len({r['group_id'] for r in rows}),pass_count=c['pass'],minor=c['minor'],
        major=c['major']+c['critical'],critical=c['critical'],pass_rate=c['pass']/n if n else 0,
        major_rate=(c['major']+c['critical'])/n if n else 1,
        json_valid=sum(r['json_valid'] for r in rows),language_correct=sum(r['language_correct'] for r in rows),
        ended=sum(r['ended'] for r in rows))


def group_intervals(rows,iterations=10000,seed=20261002):
    groups=defaultdict(list)
    for row in rows:groups[row['group_id']].append(row)
    packed=[(len(g),sum(r['verdict']=='pass' for r in g),sum(r['verdict'] in ('major','critical') for r in g)) for g in groups.values()]
    if not packed:return None
    rng=random.Random(seed);passed=[];major=[]
    for _ in range(iterations):
        sample=rng.choices(packed,k=len(packed))
        n=sum(g[0] for g in sample)
        passed.append(sum(g[1] for g in sample)/n);major.append(sum(g[2] for g in sample)/n)
    def interval(values):
        values.sort();return [values[math.floor(.025*iterations)],values[min(iterations-1,math.ceil(.975*iterations)-1)]]
    event_groups=sum(g[2]>0 for g in packed)
    return dict(unit='source_group',groups=len(packed),resamples=iterations,seed=seed,
        pass_bootstrap_95=interval(passed),major_bootstrap_95=interval(major),major_event_groups=event_groups,
        zero_major_group_event_one_sided95_upper=1-.05**(1/len(packed)) if event_groups==0 else None,
        limitation='Balanced audited test scope only; AI review and synthetic source limitations reported separately. Zero-event bootstrap bounds are degenerate; separate group-event upper bound supplied.')


def evaluate(rows,*,stage,performance):
    if stage not in ('confirmation','release'):raise ValueError('Unknown independent stage')
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate reviews')
    groups=defaultdict(set)
    for row in rows:
        if row['verdict'] not in ('pass','minor','major','critical'):raise ValueError('Pending semantic review')
        if row['category'] not in CATEGORIES or row['target_lang'] not in DIRECTIONS:raise ValueError('Unknown stratum')
        for key in ('json_valid','language_correct','ended','has_constraint','multisentence'):
            if type(row[key]) is not bool:raise ValueError('Explicit independent boolean required: '+key)
        if row['has_constraint'] and type(row.get('constraint_preserved')) is not bool:
            raise ValueError('Explicit context/term constraint review required')
        groups[row['group_id']].add(row['category'])
    if any(len(c)!=1 for c in groups.values()):raise ValueError('Source family split across categories')
    minimum=400 if stage=='confirmation' else 600
    context=[r for r in rows if r['has_constraint']]
    multi=[r for r in rows if r['multisentence']]
    strata={f'{c}/{d}':[r for r in rows if r['category']==c and r['target_lang']==d] for c in CATEGORIES for d in DIRECTIONS}
    total=counts(rows)
    constraint_rate=sum(r['constraint_preserved'] for r in context)/len(context) if context else 0
    gates=dict(rows=len(rows)>=minimum,source_groups=len(groups)>=minimum//2,
        stratum_coverage=all(len(s)>=minimum//10 for s in strata.values()),
        constraint_coverage=len(context)>=minimum//5,multisentence_coverage=len(multi)>=minimum//5,
        pass_rate=total['pass_rate']>=.95,major_rate=total['major_rate']<=.01,critical=total['critical']==0,
        json=total['json_valid']==len(rows),direction=total['language_correct']==len(rows),eos=total['ended']==len(rows),
        stratum_pass=all(counts(s)['pass_rate']>=.90 for s in strata.values()),
        stratum_major=all(counts(s)['major_rate']<=.03 for s in strata.values()),constraint_retention=constraint_rate>=.98,
        performance=bool(performance.get('performance_passed')) and performance.get('peak_reserved_gib',float('inf'))<=6.5
            and performance.get('cpu_parameter_count')==0 and bool(performance.get('all_json_valid')) and bool(performance.get('all_eos')))
    return dict(stage=stage,review_hash=fingerprint(rows),counts=total,intervals=group_intervals(rows),
        strata={k:dict(counts=counts(v),intervals=group_intervals(v)) for k,v in strata.items()},
        context=dict(counts=counts(context),retention=constraint_rate,intervals=group_intervals(context)),
        multisentence=dict(counts=counts(multi),intervals=group_intervals(multi)),
        source_composition=dict(Counter(r['source_kind'] for r in rows)),gates=gates,
        release_entry_passed=all(gates.values()),release_approved=False)
