import copy
from pathlib import Path
import pytest
from scripts import compare_v12_blind, record_v12_blind_reviews
from witrans_tools.common import read_jsonl


def test_raw_change_invalidates_masked_decisions():
    packets = copy.deepcopy(read_jsonl('runs/v12-blind-packets.jsonl'))
    packets[0]['outputs']['A'] = 'A replaced output.'
    with pytest.raises(ValueError, match='Frozen masked evidence changed'):
        compare_v12_blind.validate_decisions(packets, read_jsonl('runs/v12-blind-decisions.jsonl'))


def test_unmasking_locks_annotation_export():
    assert Path('runs/v12-blind-unmask-receipt.json').exists()
    with pytest.raises(ValueError, match='already unmasked'):
        record_v12_blind_reviews.main()
