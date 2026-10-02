import copy
import pytest
from witrans_tools.common import read_jsonl
from witrans_tools.fact_pairs import validate_fact_packet


def test_changed_fact_cannot_inherit_source_acceptance():
    packet = copy.deepcopy(read_jsonl('data/prepared/v14-public-fact-pairs/pairs.jsonl')[0])
    validate_fact_packet(packet)
    packet['outputs'][1]['translation'] = packet['outputs'][0]['translation']+'审核后新增事实。'
    with pytest.raises(ValueError,match='bound individual acceptance'):
        validate_fact_packet(packet)


def test_heldout_parent_cannot_be_used_as_training_pair():
    packet = copy.deepcopy(read_jsonl('data/prepared/v14-public-fact-pairs/pairs.jsonl')[0])
    packet['source']['training_allowed'] = False
    with pytest.raises(ValueError,match='授权'):
        validate_fact_packet(packet)


def test_all_variants_and_directions_keep_parent_family():
    pairs = read_jsonl('data/prepared/v14-public-fact-pairs/pairs.jsonl')
    parents = {}
    for packet in pairs:
        validate_fact_packet(packet)
        key = packet['source']['parent_record_id']
        assert parents.setdefault(key,packet['group_id'])==packet['group_id']
    assert len(parents)==14
