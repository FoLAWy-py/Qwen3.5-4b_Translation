"""Synthetic gate attacks; these are not confirmation examples or model results."""
import copy

import pytest

from witrans_tools.qwen35_confirmation import evaluate_confirmation


def fixture():
    candidate=[]
    for category in ('daily','travel','food','academic','hard'):
        for group in range(40):
            for direction in ('en','zh-CN'):
                candidate.append(dict(id=f'{category}/{group}/{direction}',group_id=f'{category}/{group}',
                    category=category,target_lang=direction,verdict='pass',format_valid=True,language_correct=True,ended=True))
    v7=copy.deepcopy(candidate)
    # Distributed bilateral losses:6 groups/12 rows=exactly3pp, not12 independent rows.
    for row in v7:
        if row['group_id'] in ('daily/0','daily/1','travel/0','food/0','academic/0','hard/0'):
            row['verdict']='minor'
    return candidate,v7,copy.deepcopy(candidate)


def test_exact_three_pp_boundary_and_grouped_pairs():
    c,v,f=fixture();result=evaluate_confirmation(c,v,f,iterations=5000)
    assert result['numeric_confirmation_passed']
    assert result['comparisons']['v7']['interval']['source_groups']==200
    assert result['comparisons']['v7']['interval']['pass_difference']==.03
    assert result['strata']['daily/en']['rows']==40
    assert not result['stage_goal_complete'] and not result['release_approved']


def test_gain_does_not_compensate_for_critical_direction_or_qwen35_loss():
    c,v,f=fixture();c[0]['verdict']='critical';c[1]['language_correct']=False
    result=evaluate_confirmation(c,v,f,iterations=5000)
    assert result['counts']['candidate']['major_including_critical']==1
    assert not result['gates']['candidate_critical_zero']
    assert not result['gates']['candidate_all_json_direction_eos']
    assert not result['gates']['pass_nondecrease_vs_qwen35']
    assert not result['numeric_confirmation_passed']


def test_pair_identity_and_bootstrap_budget_are_required():
    c,v,f=fixture();f[0]['target_lang']='zh-CN'
    with pytest.raises(ValueError,match='identity mismatch'):
        evaluate_confirmation(c,v,f,iterations=5000)
    with pytest.raises(ValueError,match='5000'):
        evaluate_confirmation(*fixture(),iterations=4999)
