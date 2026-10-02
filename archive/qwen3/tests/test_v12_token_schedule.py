import pytest
from archive.qwen3.scripts.prepare_v12_factorial import exact_schedule


def test_exact_schedule_keeps_whole_records_and_matches_budget():
    rows = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    encoded = [{'input_ids': [0]*size} for size in (5,7,11)]
    schedule = exact_schedule(rows, encoded, 37, 8, 42)
    lengths = {'a':5, 'b':7, 'c':11}
    assert len(schedule) == 8 and all(schedule)
    assert sum(lengths[key] for batch in schedule for key in batch) == 37*8
    assert schedule == exact_schedule(rows, encoded, 37, 8, 42)


def test_impossible_budget_is_rejected_without_truncation():
    with pytest.raises(ValueError, match='infeasible'):
        exact_schedule([{'id':'a'}], [{'input_ids':[0]*6}], 19, 3, 42)
