"""Full113 diagnostic accounting for quarantined reviewed33; cannot accept a candidate."""
import hashlib
from collections import Counter
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load, verify_training
from scripts.review_v12_factorial import accept_manual, binding
from witrans import parse_translation
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    prefix='qwen35-v3-reviewed33'
    dest=Path(f'runs/{prefix}-quarantined-recall-diagnosis.json')
    assert not dest.exists()
    plan=load(f'data/prepared/{prefix}-repair/plan.json')
    correction=load('runs/qwen35-v3-sweetener-negative-correction.json')
    assert correction['old_trial_eligible'] is False and fingerprint(plan) in correction['invalidated_plan_hashes']
    try:
        verify_training(plan)
    except ValueError as exc:
        assert 'Quarantined trial' in str(exc)
    else:
        raise AssertionError('Quarantine not enforced')
    metrics=load(Path(plan['output'])/'metrics.json')
    completed=load(f'runs/{prefix}-training-completion-audit.json')
    assert completed['metrics_hash']==fingerprint(metrics) and metrics['plan_hash']==fingerprint(plan)
    with (Path(plan['output'])/'adapter_model.safetensors').open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha==completed['adapter_sha256']==correction['quarantined_adapter_sha256']
    job=load(f'runs/{prefix}-recall-job.json');state=load(f'runs/{prefix}-recall-progress.json')
    assert job['status']=='finished' and job['exit_code']==0 and state['status']=='finished'
    assert not psutil.pid_exists(68280) and not psutil.pid_exists(66108) and not psutil.pid_exists(63504)
    refs=read_jsonl(plan['recall_path']);rows=read_jsonl(f'runs/{prefix}-recall.jsonl')
    notes=read_jsonl(f'runs/{prefix}-recall-manual.jsonl');summary=load(f'runs/{prefix}-recall.summary.json')
    indexed={r['id']:r for r in refs};generated={r['id']:r for r in rows};decisions={n['id']:n for n in notes}
    assert len(indexed)==len(generated)==len(decisions)==len(refs)==len(rows)==len(notes)==113
    assert set(indexed)==set(generated)==set(decisions)
    assert summary['data_hash']==fingerprint(refs)==plan['recall_hash']
    assert summary['generation_hash']==fingerprint(rows) and summary['adapter_sha256']==sha
    assert summary['base']==plan['base'] and summary['prompt_hash']==plan['prompt_hash']
    assert summary['quantization']=='nf4' and summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
    assert summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
    for rid,ref in indexed.items():
        raw,note=generated[rid],decisions[rid]
        accept_manual(raw,note)
        assert raw['input']==ref['input'] and raw['reference']==ref['output']
        assert parse_translation(raw['raw'])==raw['prediction']
        assert note['binding_hash']==binding(raw) and note['generation_hash']==fingerprint(raw)
        assert note['chosen_content_hash']==ref['review']['content_hash']
        assert (note['group_id'],note['category'],note['target_lang'])==(ref['group_id'],ref['category'],ref['input']['target_lang'])
        assert note['format_valid'] is True and note['language_correct'] is True and note['ended'] is True
    sets=plan['recall_sets'];rule=plan['screening_rule']
    counts={key:Counter(decisions[rid]['verdict'] for rid in sets[key])
            for key in ('actual_error_ids','preservation_ids','supplement_ids')}
    hard_en=[rid for rid in sets['preservation_ids'] if indexed[rid]['category']=='hard' and indexed[rid]['input']['target_lang']=='en']
    assert len(hard_en)==4
    overall=Counter(n['verdict'] for n in notes)
    gates=dict(error_pass_minimum=counts['actual_error_ids']['pass']>=rule['error_pass_at_least'],
        error_major_maximum=counts['actual_error_ids']['major']+counts['actual_error_ids']['critical']<=rule['error_major_at_most'],
        preservation_pass_minimum=counts['preservation_ids']['pass']>=rule['preservation_pass_at_least'],
        hard_en_preservation_all_pass=sum(decisions[rid]['verdict']=='pass' for rid in hard_en)==4,
        supplement_pass_minimum=counts['supplement_ids']['pass']>=rule['supplement_pass_at_least'],
        supplement_major_maximum=counts['supplement_ids']['major']+counts['supplement_ids']['critical']<=rule['supplement_major_at_most'],
        critical_zero=overall['critical']==0,all_format_direction_eos=True,
        three_old_direction_errors_correct=all(decisions[rid]['verdict']=='pass' for rid in sets['direction_error_ids']))
    family={psutil.Process().pid,*(p.pid for p in psutil.Process().parents())}
    model_processes=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        if p.pid in family or 'python' not in (p.info['name'] or '').lower():
            continue
        command=p.info['cmdline'] or []
        if any(term in ' '.join(command) for term in ('scripts.train_','scripts.evaluate_qwen35','scripts.probe_qwen35_',
                                                     'scripts.diagnose_qwen35_stage','scripts.benchmark_qwen35')):
            model_processes.append(dict(pid=p.pid,created=p.info['create_time'],command=command))
    assert not model_processes
    write_json(dest,dict(at=now(),adapter_sha256=sha,plan_hash=fingerprint(plan),correction_report_hash=fingerprint(correction),
        rows=113,source_groups=len({r['group_id'] for r in refs}),counts=dict(overall),
        sets={key:dict(value) for key,value in counts.items()},diagnostic_frozen_rule_gates=gates,
        diagnostic_frozen_rule_passed=all(gates.values()),candidate_eligible=False,recall_first_gate_passed=False,
        generation_hash=fingerprint(rows),decisions_hash=fingerprint(notes),summary_hash=fingerprint(summary),
        completion_audit_hash=fingerprint(completed),recall_job_hash=fingerprint(job),recall_actual_exit_code=0,
        training_peak_reserved_gib=metrics['peak_reserved_gib'],recall_peak_reserved_gib=summary['peak_reserved_gib'],
        old_training_and_recall_processes_absent=True,other_model_processes=model_processes,
        analysis=dict(A='Five of the original33 designated error items stillmajor; twentyfour pass. Set includes corrected/nonmajor negatives so not a valid repair success claim.',
                      C='Previously correct preservation40 now36pass3minor1major, includinghard/en3/4; retention gate failed.',
                      context_supervision='Forty context positives now24pass13minor3major despite minimum3 positive visits; target32pass not reached.',
                      B='No new DEV or independent confirmation generated, so unseen-source improvement untested.'),
        next='Full focused32 draft-negative reread and conservative exclusions before another frozen hypothesis; retain original416 positive references. No DEV, runtime acceptance or finalfreeze for quarantined candidate.',
        scope='Complete source-bound TRAIN diagnosis of an ineligible historical experiment. Actual execution is preserved; no accepted optimization round, generalization, latency, algorithm superiority or release claim.',
        stage_goal_complete=False,release_approved=False,default_promoted=False))
    print(dict(diagnosis=str(dest),counts=dict(overall),sets={key:dict(value) for key,value in counts.items()},
               candidate_eligible=False,diagnostic_rule_passed=all(gates.values())),flush=True)


if __name__=='__main__':
    main()
