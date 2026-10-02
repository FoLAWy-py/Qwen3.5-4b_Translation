from witrans_tools.common import fingerprint, read_jsonl, write_json, write_jsonl
import pytest

@pytest.mark.parametrize('role,current_stem,first_cache,prior_stems',[
    ('sft_control','v6-sft-control','v6-cpo',('v6-cpo','v5-selected','v5-start','v4-selected','v4-start')),
    ('sft','v8-selected','v7-critical',('v7-critical','v6-cpo','v6-sft-control','v5-selected','v5-start','v4-selected','v4-start')),
])
def test_cache_role_name_cannot_override_current_review_output(monkeypatch, tmp_path,role,current_stem,first_cache,prior_stems):
    from scripts.review_v6 import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv',['review_v6','--role',role])
    inp = {'text':'Test.', 'target_lang':'zh-CN','context':'','glossary':{}}
    ref = {'id':'one','group_id':'group','category':'daily','input':inp,'output':{'translation':'测试。'}}
    row = {'id':'one','category':'daily','input':inp,'reference':ref['output'],
        'raw':'{"translation":"测试。"}', 'prediction':ref['output'], 'ended':True}
    decision = {'id':'one','group_id':'group','category':'daily','target_lang':'zh-CN',
        'verdict':'pass','reviewer':'Codex','format_valid':True,'ended':True,'language_correct':True,
        'note':'Already individually read.', 'output_hash':fingerprint({k:row[k] for k in ('input','raw','reference')})}
    write_jsonl('refs.jsonl',[ref])
    write_json('runs/v6-evaluation-plan.json',{'input':'refs.jsonl',
        'outputs':{role:f'runs/{current_stem}-dev.jsonl'}})
    write_jsonl(f'runs/{current_stem}-dev.jsonl',[row])
    for stem in prior_stems:
        write_jsonl(f'runs/{stem}-dev.jsonl',[row])
        write_jsonl(f'runs/{stem}-dev-semantic.jsonl',[decision])
    main()
    assert read_jsonl(f'runs/{current_stem}-dev-semantic.jsonl')[0]['review_source'] == first_cache
    assert read_jsonl(f'runs/{current_stem}-dev-unmatched.jsonl') == []
    for stem in prior_stems:
        assert read_jsonl(f'runs/{stem}-dev-semantic.jsonl') == [decision]
        assert not (tmp_path / (stem+'-semantic.jsonl')).exists()
