import pytest
from witrans_tools.paired_stats import paired_pass_interval

def reviews(verdict):
    return [{'id': f'{g}-{direction}', 'group_id': str(g), 'verdict': verdict} for g in range(10) for direction in ('en', 'zh-CN')]

def test_identical_outputs_have_zero_paired_difference_and_group_count():
    result = paired_pass_interval(reviews('pass'), reviews('pass'), iterations=100)
    assert result['group_bootstrap_95_interval'] == [0, 0]
    assert result['source_groups'] == 10 and result['rows'] == 20

def test_clear_change_is_paired_and_unreviewed_or_unmatched_data_is_rejected():
    assert paired_pass_interval(reviews('pass'), reviews('major'), iterations=100)['group_bootstrap_95_interval'] == [1, 1]
    with pytest.raises(ValueError):
        paired_pass_interval(reviews('pass'), reviews('major')[:-1])
    with pytest.raises(ValueError):
        paired_pass_interval(reviews('pending'), reviews('major'))
