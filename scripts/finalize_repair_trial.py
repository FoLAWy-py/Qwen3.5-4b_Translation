"""Require content-bound manual recall decisions before the frozen recall-first gate."""
import argparse
import json
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def finalize_qwen35(plan,prefix,path):
    from scripts.decode_qwen35_repair_recall import verify_training
    from archive.qwen3.scripts.review_v12_factorial import accept_manual,binding
    from witrans import parse_translation
    metrics,sha=verify_training(plan)
    refs=read_jsonl(plan['recall_path'])
    assert fingerprint(refs)==plan['recall_hash']
    rows=read_jsonl(f'runs/{prefix}-recall.jsonl')
    notes=read_jsonl(f'runs/{prefix}-recall-manual.jsonl')
    summary=json.loads(Path(f'runs/{prefix}-recall.summary.json').read_text(encoding='utf-8'))
    indexed={r['id']:r for r in refs};by_id={r['id']:r for r in rows};decisions={n['id']:n for n in notes}
    expected=len(refs)
    assert len(rows)==len(notes)==len(indexed)==len(by_id)==len(decisions)==expected
    assert set(indexed)==set(by_id)==set(decisions)
    assert summary['data_hash']==fingerprint(refs) and summary['generation_hash']==fingerprint(rows)
    assert summary['adapter_sha256']==sha and summary['rows']==expected and summary['base']==plan['base']
    assert summary['prompt_hash']==plan['prompt_hash'] and summary['quantization']=='nf4'
    assert summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
    assert summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
    for rid,ref in indexed.items():
        row,note=by_id[rid],decisions[rid]
        accept_manual(row,note)
        assert row['input']==ref['input'] and row['reference']==ref['output']
        assert note['binding_hash']==binding(row) and note['chosen_content_hash']==ref['review']['content_hash']
        assert note['group_id']==ref['group_id'] and note['category']==ref['category']
        assert note['target_lang']==ref['input']['target_lang']
        assert note['verdict'] in ('pass','minor','major','critical') and type(note['language_correct']) is bool
        assert note['format_valid']==('prediction' in row) and note['ended']==row.get('ended')
        if 'prediction' in row:
            assert parse_translation(row['raw'])==row['prediction']
    sets=plan['recall_sets'];rule=plan['screening_rule']
    core=sets['actual_error_ids']+sets['preservation_ids']+sets['supplement_ids']
    assert len(core)==len(set(core))==expected and set(core)==set(indexed)
    counts={name:Counter(decisions[rid]['verdict'] for rid in sets[name])
        for name in ('actual_error_ids','preservation_ids','supplement_ids')}
    errors,preserved,added=[counts[name] for name in ('actual_error_ids','preservation_ids','supplement_ids')]
    hard_en=[rid for rid in sets['preservation_ids'] if indexed[rid]['category']=='hard' and indexed[rid]['input']['target_lang']=='en']
    assert len(hard_en)==4
    overall=Counter(note['verdict'] for note in notes)
    gates=dict(error_pass_minimum=errors['pass']>=rule['error_pass_at_least'],
        error_major_maximum=errors['major']+errors['critical']<=rule['error_major_at_most'],
        preservation_pass_minimum=preserved['pass']>=rule['preservation_pass_at_least'],
        hard_en_preservation_all_pass=sum(decisions[rid]['verdict']=='pass' for rid in hard_en)==rule['hard_en_preservation_pass'],
        supplement_pass_minimum=added['pass']>=rule['supplement_pass_at_least'],
        supplement_major_maximum=added['major']+added['critical']<=rule['supplement_major_at_most'],
        critical_zero=overall['critical']==rule['critical'],
        all_format_direction_eos=all(note['format_valid'] is True and note['language_correct'] is True
            and note['ended'] is True for note in notes),
        three_old_direction_errors_correct=all(decisions[rid]['language_correct'] is True for rid in sets['direction_error_ids']))
    passed=all(gates.values())
    report=dict(at=now(),plan_hash=fingerprint(plan),adapter_sha256=sha,hypothesis=plan['hypothesis'],
        rows=expected,source_groups=len({r['group_id'] for r in refs}),counts=dict(overall),
        sets={key:dict(value) for key,value in counts.items()},gates=gates,recall_first_gate_passed=passed,
        screening_rule=rule,next=rule['if_passed'] if passed else rule['if_failed'],
        generation_hash=fingerprint(rows),decisions_hash=fingerprint(notes),summary_hash=fingerprint(summary),
        full_tokens=metrics['full_tokens'],peak_reserved_gib=metrics['peak_reserved_gib'],
        scope='TRAIN-only semantic screen at predeclared final64; no DEV/generalization/performance or release claim. Method comparison and stable algorithm claims are absent.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    write_json(path,report)
    print(dict(counts=report['sets'],gates=gates,recall_first_gate_passed=passed),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--plan',required=True)
    args=parser.parse_args()
    path=Path(f'runs/{args.prefix}-recall-screening.json')
    assert not path.exists(), 'Preserve screening'
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if plan['base']['model_id']=='Qwen/Qwen3.5-4B':
        finalize_qwen35(plan,args.prefix,path)
        return
    refs=read_jsonl(plan['pairs_path'])
    rows=read_jsonl(f'runs/{args.prefix}-recall.jsonl')
    notes=read_jsonl(f'runs/{args.prefix}-recall-manual.jsonl')
    expected = len(refs)
    assert len(rows)==len(notes)==expected
    by_id={r['id']:r for r in rows}
    indexed={r['id']:r for r in refs}
    assert len(by_id)==expected and {r['id'] for r in notes}==set(by_id)==set(indexed)
    for note in notes:
        row=by_id[note['id']]
        assert note['reviewer']=='Codex' and note['generation_hash']==fingerprint(row) and note['note']
        assert row['input']==indexed[row['id']]['input'] and row['reference']==indexed[row['id']]['output']
    counts=Counter(n['verdict'] for n in notes)
    structure=all(r.get('ended') and r.get('prediction') and n['language_correct'] for r,n in
                  [(by_id[n['id']],n) for n in notes])
    rule=plan['screening_rule']
    original_ids = {r['id'] for r in read_jsonl('data/prepared/v15-error-repair/pairs.jsonl')}
    original_counts = Counter(n['verdict'] for n in notes if n['id'] in original_ids)
    new_counts = Counter(n['verdict'] for n in notes if n['id'] not in original_ids)
    passed=(counts['pass']>=rule['train_recall_pass_at_least']
        and counts['critical']==rule['train_recall_critical'] and structure
        and original_counts['pass']>=rule.get('original11_pass_at_least',0)
        and new_counts['pass']>=rule.get('new9_pass_at_least',0))
    metrics=json.loads((Path(plan['output'])/'metrics.json').read_text(encoding='utf-8'))
    summary=json.loads(Path(f'runs/{args.prefix}-recall.summary.json').read_text(encoding='utf-8'))
    assert summary['adapter_sha256']==metrics['adapter_sha256'] and metrics['plan_hash']==fingerprint(plan)
    prior={r['id']:r for r in read_jsonl('runs/v15-repair-recall.jsonl')}
    output_identical=sum(all(row.get(k)==prior[row['id']].get(k) for k in ('input','raw','prediction','reference','ended')) for row in rows if row['id'] in prior)
    report=dict(at=now(),plan_hash=fingerprint(plan),hypothesis=plan['hypothesis'],rows=expected,
        source_groups=len({r['group_id'] for r in refs}), original11_counts=dict(original_counts),new_error_counts=dict(new_counts),
        counts=dict(counts),pass_repair_rate=counts['pass']/expected,nonmajor_recall_count=counts['pass']+counts['minor'],
        major_including_critical=counts['major']+counts['critical'],
        structure_and_direction_valid=structure,recall_first_gate_passed=passed,screening_rule=rule,
        outputs_identical_to_v15=output_identical,adapter_sha256=metrics['adapter_sha256'],
        peak_reserved_gib=metrics['peak_reserved_gib'],full_tokens=metrics['full_tokens'],
        generation_hash=fingerprint(rows),decisions_hash=fingerprint(notes),
        next=rule['if_passed'] if passed else rule['if_failed'],
        scope='TRAIN recall gate only. Failure skips DEV generation. One seed and no algorithm efficacy claim.',
        stage_goal_complete=False,release_approved=False)
    write_json(path,report)
    print({k:report[k] for k in ('counts','recall_first_gate_passed','outputs_identical_to_v15','next')})


if __name__=='__main__':
    main()
