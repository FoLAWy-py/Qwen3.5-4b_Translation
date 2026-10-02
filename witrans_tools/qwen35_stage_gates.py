"""Exact stage gates against both frozen starts; no release or independent claim."""
from collections import Counter


def counts(decisions):
    c=Counter(d['verdict'] for d in decisions)
    return dict(rows=len(decisions),pass_count=c['pass'],minor=c['minor'],
                major_including_critical=c['major']+c['critical'],critical=c['critical'])


def evaluate_stage(known,public,frozen_known,v7_known):
    assert len(known)==len(frozen_known)==len(v7_known)==200 and len(public)==116
    assert len({d['id'] for d in known})==200 and len({d['id'] for d in public})==116
    assert {d['id'] for d in known}=={d['id'] for d in frozen_known}=={d['id'] for d in v7_known}
    assert {d['verdict'] for d in known+public+frozen_known+v7_known} <= {'pass','minor','major','critical'}
    strata={}
    for category in ('daily','travel','food','academic','hard'):
        for direction in ('en','zh-CN'):
            filter_rows=lambda ds:[d for d in ds if d['category']==category and d['target_lang']==direction]
            current,starting,legacy=[filter_rows(ds) for ds in (known,frozen_known,v7_known)]
            assert len(current)==len(starting)==len(legacy)==20
            c,s,l=[sum(d['verdict']=='pass' for d in ds) for ds in (current,starting,legacy)]
            # At n20, one changed pass is5pp; a2pp floor permits no whole-row loss.
            strata[category+'/'+direction]=dict(rows=20,candidate_pass=c,frozen_qwen35_pass=s,v7_pass=l,
                 required_pass=max(s,l),change_vs_qwen35=(c-s)/20,change_vs_v7=(c-l)/20,
                 passed=(c-s)/20>=-.02 and (c-l)/20>=-.02)
    kc,pc=counts(known),counts(public)
    gates=dict(known_pass_at_least164=kc['pass_count']>=164,known_major_at_most17=kc['major_including_critical']<=17,
               public_pass_at_least105=pc['pass_count']>=105,public_major_at_most3=pc['major_including_critical']<=3,
               category_direction_regression_vs_both_at_most2pp=all(s['passed'] for s in strata.values()),
               hard_en_to_zh_pass_at_least17=strata['hard/zh-CN']['candidate_pass']>=17,
               both_critical_zero=kc['critical']==pc['critical']==0,
               all_json_direction_eos=all(all(d[k] is True for k in ('format_valid','language_correct','ended')) for d in known+public))
    return dict(counts={'known':kc,'public':pc},gates=gates,passed=all(gates.values()),strata=strata,
                effective_known_pass_minimum=sum(s['required_pass'] for s in strata.values()),
                scope='Development screen only. Separate final own-weight performance and fresh400/200-group confirmation remain required.',
                stage_goal_complete=False,release_approved=False)
