"""Export TRAIN reading packets or bind explicit Codex readings; never auto-grade."""
import argparse
import json
from collections import Counter
from pathlib import Path

from archive.qwen3.scripts.review_v12_factorial import binding, accept_manual
from witrans import SYSTEM_PROMPT
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',default='runs/qwen35-v3-train-discovery-recovery1.jsonl')
    parser.add_argument('--training-plan',help='Use the frozen candidate TRAIN recall set instead of starting discovery')
    parser.add_argument('--readings')
    parser.add_argument('--reuse-perf',action='store_true')
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--limit',type=int,default=32)
    args=parser.parse_args()
    output=Path(args.generation)
    rows=read_jsonl(output)
    expected_sha='fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
    if args.training_plan:
        training=json.loads(Path(args.training_plan).read_text(encoding='utf-8'))
        refs=read_jsonl(training['recall_path'])
        assert fingerprint(refs)==training['recall_hash']
        metrics=json.loads((Path(training['output'])/'metrics.json').read_text(encoding='utf-8'))
        assert metrics['plan_hash']==fingerprint(training) and metrics['selected_checkpoint']==training['updates']
        expected_sha=metrics['adapter_sha256']
        assert not args.reuse_perf, 'Starting-weight timing readings cannot grade changed candidate weights'
    else:
        refs=read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
        manifest=json.loads(Path('data/prepared/v16-reviewed-families/resolved-manifest.json').read_text(encoding='utf-8'))
        assert fingerprint(refs)==manifest['train_ready_hash']
    expected=len(refs)
    indexed={r['id']:r for r in refs};generated={r['id']:r for r in rows}
    assert len(generated)==len(rows) and set(generated)<=set(indexed)
    summary_path=output.with_suffix('.summary.json')
    if summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        assert summary['adapter_sha256']==expected_sha
        assert summary['base']['revision']=='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        assert summary['prompt_hash']==fingerprint(SYSTEM_PROMPT)
        assert summary['data_hash']==fingerprint(refs) and summary['generation_hash']==fingerprint(rows)
        assert summary['quantization']=='nf4' and summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
        assert summary['rows']==expected and summary['cpu_parameter_count']==0 and summary['peak_reserved_gib']<=6.5
    manual_path=output.with_name(output.stem+'-manual.jsonl')
    notes=read_jsonl(manual_path) if manual_path.exists() else []
    old={r['id']:r for r in notes}
    assert len(old)==len(notes)
    for row in rows:
        ref=indexed[row['id']]
        validate_record(ref,True,True,purpose='training')
        assert row['input']==ref['input'] and row['reference']==ref['output']
        assert row.get('raw') is not None, 'Failed generation must be inspected separately'
    for note in notes:
        accept_manual(generated[note['id']],note)
        ref=indexed[note['id']]
        assert note['binding_hash']==binding(generated[note['id']])
        assert note['chosen_content_hash']==ref['review']['content_hash']
        assert note['group_id']==ref['group_id'] and note['category']==ref['category']
        assert note['target_lang']==ref['input']['target_lang']
        assert note['format_valid']==('prediction' in generated[note['id']]) and note['ended']==generated[note['id']].get('ended')
    if args.reuse_perf:
        assert not args.readings
        cached=read_jsonl('runs/qwen35-v3-performance-train-overlap-manual.jsonl')
        perf=read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance.jsonl')
        source_hashes={fingerprint(r) for r in perf}
        plan=json.loads(Path('runs/qwen35-v3-diagnostic-recovery1/plan.json').read_text(encoding='utf-8'))
        incoming=[]
        for prior in cached:
            assert prior['reviewer']=='Codex' and prior['source_row_hash'] in source_hashes
            assert prior['adapter_sha256']==plan['adapter_sha256']
            rid=prior['id']
            if rid not in generated or rid in old: continue
            raw=generated[rid];ref=indexed[rid]
            if prior['binding_hash']!=binding(raw): continue
            assert prior['chosen_content_hash']==ref['review']['content_hash']
            incoming.append(dict(id=rid,group_id=ref['group_id'],category=ref['category'],target_lang=ref['input']['target_lang'],
                reviewer='Codex',at=now(),generation_hash=fingerprint(raw),binding_hash=binding(raw),
                verdict=prior['verdict'],language_correct=prior['language_correct'],note=prior['note'],
                chosen_content_hash=ref['review']['content_hash'],format_valid='prediction' in raw,ended=raw.get('ended'),
                review_source='Exact earlier TRAIN/performance output reading; both original and current output hashes verified',
                reused_reading_hash=fingerprint(prior),scope=prior['scope']))
        for row in incoming: append_jsonl(manual_path,row)
        notes+=incoming
        write_json(output.with_name(output.stem+'-review-progress.json'),dict(at=now(),generated=len(rows),expected=expected,
            reviewed=len(notes),counts=dict(Counter(r['verdict'] for r in notes)),review_hash=fingerprint(notes),
            complete=len(rows)==len(notes)==expected and summary_path.exists(),release_approved=False))
        print(dict(reused=len(incoming),reviewed=len(notes)),flush=True)
    elif args.readings:
        # TSV explicit literal: id, verdict, language_correct, explanatory note.
        incoming=[]
        for line in Path(args.readings).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            rid,verdict,language,note=line.split('\t',3)
            assert rid in generated and rid not in old
            assert verdict in ('pass','minor','major','critical') and language in ('true','false') and note.strip()
            raw=generated[rid];ref=indexed[rid]
            incoming.append(dict(id=rid,group_id=ref['group_id'],category=ref['category'],
                target_lang=ref['input']['target_lang'],reviewer='Codex',at=now(),
                generation_hash=fingerprint(raw),binding_hash=binding(raw),
                verdict=verdict,language_correct=language=='true',note=note,
                chosen_content_hash=ref['review']['content_hash'],
                format_valid='prediction' in raw,ended=raw.get('ended'),
                scope='Individual authorized Codex AI reading of Qwen3.5 real TRAIN generation; not a human or independent test.'))
        assert len({r['id'] for r in incoming})==len(incoming)
        for row in incoming: append_jsonl(manual_path,row)
        notes+=incoming
        write_json(output.with_name(output.stem+'-review-progress.json'),dict(at=now(),generated=len(rows),expected=expected,
            reviewed=len(notes),counts=dict(Counter(r['verdict'] for r in notes)),
            review_hash=fingerprint(notes),complete=len(rows)==len(notes)==expected and summary_path.exists(),release_approved=False))
        print(dict(reviewed=len(notes),counts=dict(Counter(r['verdict'] for r in notes))),flush=True)
    else:
        for row in rows[args.start:args.start+args.limit]:
            ref=indexed[row['id']]
            if args.compact:
                print(json.dumps(dict(id=row['id'],category=ref['category'],input=row['input'],reference=row['reference'],
                    raw=row['raw'],ended=row.get('ended'),already_reviewed=row['id'] in old,
                    source={k:ref['source'][k] for k in ('name','license')}),ensure_ascii=False),flush=True)
                continue
            print(json.dumps(dict(id=row['id'],category=ref['category'],group_id=ref['group_id'],input=row['input'],
                reference=row['reference'],raw=row['raw'],ended=row.get('ended'),source=ref['source'],
                already_reviewed=row['id'] in old,generation_hash=fingerprint(row)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
