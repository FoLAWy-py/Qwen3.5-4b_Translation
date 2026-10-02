"""Promotion must not hide a regression against the original base model."""
import hashlib
import json
import pytest
from witrans_tools.common import fingerprint, write_json, write_jsonl


@pytest.mark.parametrize('baseline_pass_groups,expected_gate',[(95,False),(85,True)])
def test_original_base_control_gates_promotion(monkeypatch,tmp_path,baseline_pass_groups,expected_gate):
    from archive.qwen3.scripts.finalize_v6 import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv',['finalize_v6','--plan','plan.json'])
    refs = []
    categories = ('daily','travel','food','academic','hard')
    for group in range(100):
        for lang in ('zh-CN','en'):
            refs.append({'id':f'{group}-{lang}','group_id':f'group-{group}',
                'category':categories[group%5],
                'input':{'text':f'Synthetic source {group} {lang}.','target_lang':lang,'context':'','glossary':{}},
                'output':{'translation':'正确译文' if lang=='zh-CN' else 'Correct translation'}})
    write_jsonl('refs.jsonl',refs)
    weights = b'synthetic fixture, never load as model'
    candidate = tmp_path/'candidate'
    candidate.mkdir()
    (candidate/'adapter_model.safetensors').write_bytes(weights)
    sha = hashlib.sha256(weights).hexdigest()
    write_json(candidate/'witrans_adapter.json',{'adapter_sha256':sha})
    base = {'model_id':'Qwen/Qwen3-4B','revision':'fixed-test-revision','status':'complete'}
    decoding = {'do_sample':False,'max_length':1024,'max_new_tokens':256}
    specs = [('start','parent',80,'parent-sha'),('sft','candidate',90,sha),
             ('baseline','original-base',baseline_pass_groups,None)]
    for role,stem,pass_groups,digest in specs:
        raw,decisions = [],[]
        for ref in refs:
            verdict = 'pass' if int(ref['id'].split('-')[0])<pass_groups else 'minor'
            prediction = ref['output'] if verdict=='pass' else {'translation':'Synthetic awkward wording'}
            row = {'id':ref['id'],'input':ref['input'],'reference':ref['output'],
                'raw':json.dumps(prediction,ensure_ascii=False),'prediction':prediction,'ended':True}
            raw.append(row)
            decisions.append({'id':ref['id'],'group_id':ref['group_id'],'category':ref['category'],
                'target_lang':ref['input']['target_lang'],'verdict':verdict,'reviewer':'Codex',
                'note':'Synthetic test fixture judgement.','format_valid':True,'ended':True,
                'language_correct':True,'output_hash':fingerprint({k:row[k] for k in ('input','raw','reference')})})
        write_jsonl(f'runs/{stem}-dev.jsonl',raw)
        write_jsonl(f'runs/{stem}-dev-semantic.jsonl',decisions)
        write_json(f'runs/{stem}-dev.summary.json',{'adapter_sha256':digest,
            'data_hash':fingerprint(refs),'quantization':'nf4','decoding':decoding,'prompt_hash':'fixed-prompt',
            'base':base,'adapter_dir':None if role=='baseline' else 'fixture-adapter',
            'model':'Qwen3-4B baseline' if role=='baseline' else 'witrans-4b',
            'buckets':{'all':{'valid_json':200}}})
    write_json('plan.json',{'input':'refs.jsonl','development_hash':fingerprint(refs),
        'candidate_roles':['sft'],'candidates':{'sft':{'directory':str(candidate),'adapter_sha256':sha}},
        'evidence':{'start':'parent','sft':'candidate'},
        'comparison_controls':{'baseline':{'stem':'original-base','adapter_sha256':None}},
        'starting_adapter_sha256':'parent-sha','quantization':'nf4','decoding':decoding,
        'prompt_hash':'fixed-prompt','report_stem':'runs/test-report'})
    main()
    report = json.loads((tmp_path/'runs/test-report.json').read_text(encoding='utf-8'))
    assert report['comparisons']['sft']['paired_stats']['pass_difference']==pytest.approx(.10)
    assert report['development_promotion_gates']['sft'] is expected_gate
    assert report['comparisons']['sft_vs_baseline']['not_worse_than_control'] is expected_gate
    assert report['release_approved'] is False
    assert report['default_promoted'] is False
