"""Adversarial stage selection cases; CPU-only synthetic decisions, no model inference."""
import copy

from witrans_tools.qwen35_stage_gates import evaluate_stage


def fixture_decisions():
    baseline_pass={'daily/en':16,'daily/zh-CN':15,'travel/en':17,'travel/zh-CN':13,
                   'food/en':17,'food/zh-CN':12,'academic/en':19,'academic/zh-CN':20,
                   'hard/en':19,'hard/zh-CN':16}
    baseline=[]
    for stratum,n in baseline_pass.items():
        category,direction=stratum.split('/')
        for index in range(20):
            baseline.append(dict(id=stratum+str(index),category=category,target_lang=direction,
                                verdict='pass' if index<n else 'minor',format_valid=True,
                                language_correct=True,ended=True))
    legacy=copy.deepcopy(baseline)
    for stratum in ('travel/en','hard/zh-CN'):
        next(d for d in legacy if d['id']==stratum+str(baseline_pass[stratum]))['verdict']='pass'
    public=[dict(id=str(i),category='daily',target_lang='en',verdict='pass' if i<105 else 'minor',
                 format_valid=True,language_correct=True,ended=True) for i in range(116)]
    return baseline,legacy,public


def repaired_decisions():
    baseline,legacy,public=fixture_decisions()
    candidate=copy.deepcopy(legacy)
    return candidate,baseline,legacy,public


def test_aggregate164_cannot_hide_dual_baseline_stratum_failure():
    baseline,legacy,public=fixture_decisions()
    result=evaluate_stage(baseline,public,baseline,legacy)
    assert result['gates']['known_pass_at_least164']
    assert result['effective_known_pass_minimum']==166
    assert not result['passed']
    assert {k for k,v in result['strata'].items() if not v['passed']}=={'travel/en','hard/zh-CN'}


def test166_does_not_allow_a_tradeoff_in_another_stratum():
    candidate,baseline,legacy,public=repaired_decisions()
    assert evaluate_stage(candidate,public,baseline,legacy)['passed']
    next(d for d in candidate if d['id']=='academic/en0')['verdict']='minor'
    result=evaluate_stage(candidate,public,baseline,legacy)
    assert result['counts']['known']['pass_count']==165
    assert not result['passed']
    assert not result['strata']['academic/en']['passed']


def test_semantics_cannot_compensate_for_direction_or_missing_eos():
    candidate,baseline,legacy,public=repaired_decisions()
    candidate[0]['language_correct']=False
    assert not evaluate_stage(candidate,public,baseline,legacy)['passed']
    candidate[0]['language_correct']=True
    public[0]['ended']=False
    assert not evaluate_stage(candidate,public,baseline,legacy)['passed']


def test_critical_counts_as_major_but_never_passes_zero_critical_gate():
    candidate,baseline,legacy,public=repaired_decisions()
    next(d for d in candidate if d['verdict']=='minor')['verdict']='critical'
    result=evaluate_stage(candidate,public,baseline,legacy)
    assert result['counts']['known']['major_including_critical']==1
    assert result['gates']['known_major_at_most17']
    assert not result['gates']['both_critical_zero'] and not result['passed']
