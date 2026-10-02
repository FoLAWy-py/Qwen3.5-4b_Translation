"""Decode one frozen TRAIN-screened repair candidate, with cached v7 control."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def qwen35_controls(training):
    from scripts.audit_qwen35_stage import legacy_known
    from archive.qwen3.scripts.finalize_v13_optimization import complete_review
    datasets=load('data/prepared/qwen35-v2/plan.json')['datasets']
    controls=[]
    for role,adapter,sha,stems in (
        ('frozen_qwen35',training['starting_adapter'],training['starting_adapter_sha256'],
         dict(known='runs/qwen35-finetuned-known',public='runs/qwen35-finetuned-public')),
        ('v7','models/witrans-4b-v7-critical-cpo',
         '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424',
         dict(known='runs/v15-error-start-dev',public='runs/v13-v7-public'))):
        actual=hashlib.sha256((Path(adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()
        assert actual==sha
        for split,stem in stems.items():
            generated,reviewed=(legacy_known if role=='v7' and split=='known' else complete_review)(stem,datasets[split])
            summary=load(stem+'.summary.json')
            assert summary['adapter_sha256']==sha and summary['data_hash']==datasets[split]['hash']
            assert summary['prompt_hash']==training['prompt_hash'] and summary['quantization']=='nf4'
            assert summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
            controls.append(dict(role=role,split=split,baseline=False,adapter=adapter,adapter_sha256=sha,
                output=stem+'.jsonl',cached=True,generation_hash=fingerprint(generated),
                decisions_hash=fingerprint(reviewed),summary_hash=fingerprint(summary)))
    return datasets,controls


def decode_qwen35(args,training,state_path):
    import psutil
    from scripts.decode_qwen35_repair_recall import verify_training
    owner=load(args.owner_job)
    assert owner['status']=='finished' and owner['exit_code']==0
    screening=load(f'runs/{args.prefix}-recall-screening.json')
    assert screening['recall_first_gate_passed'] and screening['plan_hash']==fingerprint(training)
    metrics,sha=verify_training(training)
    selection=load(f'runs/{args.prefix}-fixed-selection.json')
    assert selection['training_plan_hash']==fingerprint(training)
    assert selection['selected']['adapter_sha256']==sha==screening['adapter_sha256']
    family={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid in family or 'python' not in (process.info['name'] or '').lower():
            continue
        command=' '.join(process.info['cmdline'] or [])
        assert not any(term in command for term in ('scripts.train_','scripts.evaluate_qwen35',
            'scripts.probe_qwen35_','scripts.diagnose_qwen35_stage')), 'Another GPU process remains'
    datasets,controls=qwen35_controls(training)
    targets=controls+[dict(role=args.prefix,split=split,baseline=False,adapter=training['output'],
        adapter_sha256=sha,output=f'runs/{args.prefix}-candidate-{split}.jsonl') for split in datasets]
    assert all(not Path(target['output']).exists() for target in targets if not target.get('cached'))
    evaluation=dict(at=now(),training_plan_hash=fingerprint(training),datasets=datasets,
        semantic_cache_roles=['frozen_qwen35','v7'],base=training['base'],prompt_hash=training['prompt_hash'],
        targets=targets,quantization='nf4',max_length=1024,max_new_tokens=256,
        statistics=dict(unit='paired_source_group',iterations=10000,seed=20261002,scope='Diagnostic DEV uncertainty; repeatedly used sources, not independent confirmation.'),
        policy='One final TRAIN-screened Qwen3.5 candidate versus both frozenQwen35 and actualV7. No raw-base generation; repeatedly used DEV, not independent confirmation.')
    plan_path=Path(f'runs/{args.prefix}-evaluation-plan.json');assert not plan_path.exists()
    write_json(plan_path,evaluation)
    state=dict(at=now(),phase='candidate_frozen',plan_hash=fingerprint(evaluation),adapter_sha256=sha,
        completed_generations=[],stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(state_path,state)
    try:
        for split,spec in datasets.items():
            state.update(phase='generating',current=split,updated_at=now());write_json(state_path,state)
            output=f'runs/{args.prefix}-candidate-{split}.jsonl'
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
                'scripts.evaluate_qwen35','--adapter-dir',training['output'],'--input',spec['path'],
                '--output',output,'--max-new-tokens','256'],check=True)
            summary=load(Path(output).with_suffix('.summary.json'))
            assert summary['adapter_sha256']==sha and summary['data_hash']==spec['hash']
            assert summary['prompt_hash']==training['prompt_hash'] and summary['base']==training['base']
            assert summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
            assert summary['quantization']=='nf4' and summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
            state['completed_generations'].append(split)
        state.update(phase='individual_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error=str(exc),updated_at=now())
        raise
    finally:
        write_json(state_path,state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--owner-job', required=True)
    args = parser.parse_args()
    prefix = args.prefix
    state_path = Path(f'runs/{prefix}-evaluation-progress.json')
    assert not state_path.exists(), 'Preserve every previous evaluation attempt'
    training = load(args.plan)
    if training['base']['model_id']=='Qwen/Qwen3.5-4B':
        decode_qwen35(args,training,state_path)
        return
    owner = load(args.owner_job)
    assert owner['status'] == 'finished' and owner['exit_code'] == 0
    screening = load(f'runs/{prefix}-recall-screening.json')
    assert screening['recall_first_gate_passed'] and screening['plan_hash'] == fingerprint(training)
    selection = load(f'runs/{prefix}-fixed-selection.json')
    selected = selection['selected']
    assert selection['training_plan_hash'] == fingerprint(training)
    sha = hashlib.sha256((Path(selected['directory'])/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert sha == selected['adapter_sha256'] == screening['adapter_sha256']
    original = load('data/prepared/v13-public-optimization/plan.json')
    datasets = {'known':original['dev'],'public':original['public_dev']}
    for spec in datasets.values():
        assert fingerprint(read_jsonl(spec['path'])) == spec['hash']
    public_summary = load('runs/v13-v7-public.summary.json')
    assert public_summary['adapter_sha256'] == training['starting_adapter_sha256']
    assert public_summary['data_hash'] == datasets['public']['hash']
    assert public_summary['prompt_hash'] == training['prompt_hash']
    targets = [dict(role='v7',split='public',baseline=False,adapter=training['starting_adapter'],
        adapter_sha256=training['starting_adapter_sha256'],output='runs/v13-v7-public.jsonl',cached=True)]
    targets += [dict(role=prefix,split=split,baseline=False,adapter=selected['directory'],
        adapter_sha256=sha,output=f'runs/{prefix}-candidate-{split}.jsonl') for split in datasets]
    assert all(not Path(t['output']).exists() for t in targets if not t.get('cached'))
    evaluation = dict(at=now(),training_plan_hash=fingerprint(training),datasets=datasets,
        semantic_cache_roles=['v7'],base=training['base'],prompt_hash=training['prompt_hash'],targets=targets,
        quantization='nf4',max_length=1024,max_new_tokens=256,
        policy='Cached current best v7 versus one fixed repair candidate; no original-base comparison. No independent confirmation claim.')
    write_json(f'runs/{prefix}-evaluation-plan.json',evaluation)
    state = dict(at=now(),phase='candidate_frozen',plan_hash=fingerprint(evaluation),
        adapter_sha256=sha,completed_generations=[],release_approved=False)
    write_json(state_path,state)
    try:
        for target in targets:
            if target.get('cached'):
                continue
            state.update(phase='generating',current=target['split'],updated_at=now())
            write_json(state_path,state)
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                '--adapter-dir',target['adapter'],'--input',datasets[target['split']]['path'],
                '--output',target['output'],'--quantization','nf4','--max-length','1024',
                '--max-new-tokens','256'],check=True)
            summary = load(Path(target['output']).with_suffix('.summary.json'))
            assert summary['adapter_sha256'] == sha and summary['base'] == training['base']
            assert summary['prompt_hash'] == training['prompt_hash']
            assert summary['data_hash'] == datasets[target['split']]['hash']
            assert summary['quantization'] == 'nf4'
            assert summary['decoding'] == dict(do_sample=False,max_length=1024,max_new_tokens=256)
            state['completed_generations'].append(target['split'])
        state.update(phase='individual_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error=str(exc),updated_at=now())
        raise
    finally:
        write_json(state_path,state)


if __name__ == '__main__':
    main()
