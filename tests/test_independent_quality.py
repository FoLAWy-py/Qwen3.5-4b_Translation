import pytest
from witrans_tools.independent_quality import counts, group_intervals, evaluate


def row(rid,verdict='pass'):
    return dict(id=rid,group_id='shared',category='food',target_lang='en',verdict=verdict,
        json_valid=True,language_correct=True,ended=True,has_constraint=True,constraint_preserved=True,
        multisentence=True,source_kind='synthetic_Codex')


def test_critical_is_also_major_and_source_pairs_are_one_unit():
    rows=[row('a'),row('b','critical')]
    c=counts(rows)
    assert c['critical']==1 and c['major']==1 and c['source_groups']==1
    interval=group_intervals(rows)
    assert interval['groups']==1 and interval['pass_bootstrap_95']==[.5,.5]
    assert interval['major_bootstrap_95']==[.5,.5]


def test_incomplete_or_cross_category_family_cannot_pass():
    rows=[row('a'),row('b')]
    report=evaluate(rows,stage='confirmation',performance={})
    assert report['release_entry_passed'] is False
    assert report['gates']['performance'] is False
    rows[1]['category']='hard'
    with pytest.raises(ValueError,match='family'):evaluate(rows,stage='confirmation',performance={})
    rows[1]['category']='food';rows[1]['constraint_preserved']=None
    with pytest.raises(ValueError,match='constraint'):evaluate(rows,stage='confirmation',performance={})
