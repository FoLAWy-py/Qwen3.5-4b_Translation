from witrans_tools.training_triage import select_training_triage


def test_two_high_loss_variants_do_not_count_as_two_source_families():
    rows = [{'id':f'r{i}','group_id':f'g{i}','category':'daily','target_lang':'en',
             'short_decode_eligible':True,'answer_eos_nll':10.-i} for i in range(5)]
    rows.insert(1,{**rows[0],'id':'same-family-variant','answer_eos_nll':9.9})
    selected = select_training_triage(rows,limit=4,quota=2)
    assert len({row['group_id'] for row in selected})==4
    assert 'same-family-variant' not in {row['id'] for row in selected}
