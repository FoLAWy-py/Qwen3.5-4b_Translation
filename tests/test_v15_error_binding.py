import copy
import pytest
from scripts.prepare_v15_error_repair import reviewed_error_pairs
from witrans_tools.common import read_jsonl


def evidence():
    return (read_jsonl('runs/v15-training-hardness/review-selected-records.jsonl'),
            read_jsonl('runs/v15-training-triage-generations.jsonl'),
            read_jsonl('runs/v15-training-triage-manual.jsonl'))


def test_valid_paraphrases_never_become_negatives():
    refs, generations, notes = evidence()
    pairs = reviewed_error_pairs(refs,generations,notes)
    expected = {r['id'] for r in notes if r['verdict'] in ('major','critical')}
    assert {r['id'] for r in pairs}==expected
    assert len(pairs)==11


def test_changed_generation_cannot_reuse_judgment():
    refs, generations, notes = evidence()
    generations = copy.deepcopy(generations)
    generations[0]['raw']+=' '
    with pytest.raises(ValueError):
        reviewed_error_pairs(refs,generations,notes)


def test_heldout_source_is_not_training_authorized():
    refs, generations, notes = evidence()
    refs = copy.deepcopy(refs)
    refs[0]['source']['training_allowed']=False
    with pytest.raises(ValueError,match='授权'):
        reviewed_error_pairs(refs,generations,notes)
