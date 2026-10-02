"""CPU-only serial handoff: wait for actual training exit, verify64, decode TRAIN121."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from witrans_tools.common import fingerprint,now,read_jsonl,write_json
from witrans_tools.qwen35_trial_eligibility import require_eligible_trial


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_training(plan):
    require_eligible_trial(plan)
    correction=Path('runs/qwen35-v3-borderline-negative-correction.json')
    if correction.exists() and plan['output']=='models/witrans-qwen35-v3-targeted-cpo':
        assert load(correction)['old_trial_eligible'] is not False, 'Quarantined trial includes a corrected reasonable-translation negative; preserve artifacts but do not accept candidate'
    output=Path(plan['output'])
    metrics=load(output/'metrics.json')
    config=load(output/'run_config.json')
    metadata=load(output/'witrans_adapter.json')
    assert config['plan']==plan and config['plan_hash']==metrics['plan_hash']==metadata['plan_hash']==fingerprint(plan)
    assert metrics['selected_checkpoint']==plan['selected_checkpoint']==plan['updates']==64
    assert len(metrics['history'])==64 and [r['step'] for r in metrics['history']]==list(range(1,65))
    assert metrics['full_tokens']==plan['full_token_budget']==config['expected_full_tokens']
    assert len(plan['schedule'])==plan['updates']*plan['accumulation']
    assert metrics['forward_calls']==plan['budget']['forwards']==sum(2+len(s['replay']) for s in plan['schedule'])
    assert metrics['backward_calls']==plan['budget']['backwards']==sum(1+len(s['replay']) for s in plan['schedule'])
    assert metrics['cpu_parameter_count']==0 and metrics['peak_reserved_gib']<=plan['memory_cap_gib']==6.5
    with (output/'adapter_model.safetensors').open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha==metrics['adapter_sha256']==metadata['adapter_sha256']
    assert metadata['parent_adapter_sha256']==plan['starting_adapter_sha256']
    assert metadata['base_model_id']==plan['base']['model_id']=='Qwen/Qwen3.5-4B'
    assert metadata['base_revision']==plan['base']['revision'] and metadata['prompt_hash']==plan['prompt_hash']
    assert metadata['smoke_only'] is False
    return metrics,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--owner-job',required=True)
    parser.add_argument('--owner-pid',type=int,required=True)
    parser.add_argument('--owner-created',type=float,required=True)
    parser.add_argument('--state',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    dest=Path(args.state);out=Path(args.output)
    assert not dest.exists() and not out.exists() and not out.with_suffix('.summary.json').exists()
    plan=load(args.plan);owner=load(args.owner_job)
    state=dict(at=now(),phase='waiting_for_training_exit',status='running',plan_hash=fingerprint(plan),
        owner_job=args.owner_job,owner_pid=args.owner_pid,owner_created=args.owner_created,
        gpu_model_loaded_in_orchestrator=False,stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(dest,state)
    handle=None
    try:
        if owner['status']=='running':
            process=psutil.Process(args.owner_pid)
            assert abs(process.create_time()-args.owner_created)<0.001
            assert 'scripts.train_v15_error_repair' in process.cmdline() and args.plan in process.cmdline()
            import _winapi
            handle=_winapi.OpenProcess(0x00100000|0x1000,False,args.owner_pid)
            state['verified_owner_command']=process.cmdline();write_json(dest,state)
            while _winapi.WaitForSingleObject(handle,20000)==258:
                state['updated_at']=now();write_json(dest,state)
            state['actual_training_exit_code']=_winapi.GetExitCodeProcess(handle)
            assert state['actual_training_exit_code']==0
            # The uv supervisor records completion only after its real child
            # exits. Wait for this durable completion instead of racing it.
            while load(args.owner_job)['status']=='running':
                time.sleep(.5)
        owner=load(args.owner_job)
        assert owner['status']=='finished' and owner['exit_code']==0
        family={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid==args.owner_pid:
                raise RuntimeError('Original GPU process still exists; preserve state')
            if process.pid in family or 'python' not in (process.info['name'] or '').lower():
                continue
            command=' '.join(process.info['cmdline'] or [])
            assert not any(term in command for term in ('scripts.train_','scripts.evaluate_qwen35',
                'scripts.probe_qwen35_','scripts.diagnose_qwen35_stage')), 'Another model process remains'
        metrics,sha=verify_training(plan)
        refs=read_jsonl(plan['recall_path'])
        expected=sum(len(plan['recall_sets'][key]) for key in ('actual_error_ids','preservation_ids','supplement_ids'))
        assert len(refs)==expected and fingerprint(refs)==plan['recall_hash']
        selection=out.with_name(out.stem.removesuffix('-recall')+'-fixed-selection.json')
        assert not selection.exists()
        write_json(selection,dict(at=now(),training_plan_hash=fingerprint(plan),
            selected=dict(directory=plan['output'],step=64,adapter_sha256=sha),
            selection=plan['selection'],metrics_hash=fingerprint(metrics),release_approved=False))
        state.update(phase='generating_train_recall',adapter_sha256=sha,updated_at=now());write_json(dest,state)
        # This process never loads CUDA. Only the subprocess owns model memory;
        # it starts after the held original training handle has signalled exit.
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
            'scripts.evaluate_qwen35','--adapter-dir',plan['output'],'--input',plan['recall_path'],
            '--output',str(out).replace('\\','/'),'--max-new-tokens','256'],check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        summary=load(out.with_suffix('.summary.json'));rows=read_jsonl(out)
        assert summary['adapter_sha256']==sha and summary['data_hash']==plan['recall_hash']
        assert summary['generation_hash']==fingerprint(rows) and summary['rows']==len(rows)==expected
        assert summary['prompt_hash']==plan['prompt_hash'] and summary['base']==plan['base']
        assert summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
        assert summary['quantization']=='nf4' and summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
        state.update(status='finished',phase='individual_train_readings_pending',expected_rows=expected,summary_hash=fingerprint(summary),
            note='No automatic semantic grade or DEV decoding. Original-error repair, preservation, and supplement gates require individual actual readings.')
    except Exception as exc:
        state.update(status='failed',phase='failed_preserving_evidence',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if handle is not None:
            import _winapi
            _winapi.CloseHandle(handle)
        state['updated_at']=now();write_json(dest,state)


if __name__=='__main__':
    main()
