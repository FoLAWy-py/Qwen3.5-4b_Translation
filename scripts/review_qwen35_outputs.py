"""Reuse exact content-bound readings only; export concise new reading packets."""
import argparse
import json
from collections import Counter
from pathlib import Path
from scripts.review_v12_factorial import binding,collect,accept_manual
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',required=True,choices=('base','finetuned'))
    parser.add_argument('--split',required=True,choices=('known','public'))
    parser.add_argument('--plan',default='data/prepared/qwen35-v2/plan.json')
    args=parser.parse_args()
    plan=load(args.plan);spec=plan['datasets'][args.split]
    refs=read_jsonl(spec['path'])
    assert fingerprint(refs)==spec['hash']
    indexed={r['id']:r for r in refs}
    stem=f'runs/qwen35-base-v2-{args.split}' if args.role=='base' else f'runs/qwen35-finetuned-{args.split}'
    rows=read_jsonl(stem+'.jsonl')
    assert len(rows)==len({r['id'] for r in rows}) and {r['id'] for r in rows}<=set(indexed)
    if Path(stem+'.summary.json').exists():
        summary=load(stem+'.summary.json')
        assert summary['base']==plan['base'] and summary['prompt_hash']==plan['prompt_hash']
        assert summary['data_hash']==spec['hash'] and summary['quantization']=='nf4'
        assert summary['decoding']==dict(do_sample=False,max_length=1024,max_new_tokens=256)
        if args.role=='base': assert summary['adapter_sha256'] is None
        else: assert summary['adapter_sha256']==load(Path(plan['cpo']['output'])/'witrans_adapter.json')['adapter_sha256']
    manual_path=Path(stem+'-manual.jsonl')
    notes=read_jsonl(manual_path) if manual_path.exists() else []
    manual={n['id']:n for n in notes}
    assert len(manual)==len(notes)
    caches=[]
    prior_stems=(['runs/v15-error-selected-dev','runs/v13-start-dev'] if args.split=='known'
        else ['runs/v13-v7-public','runs/v15-error-candidate-public'])
    prior_stems += [f'runs/qwen35-base-v2-{args.split}' if r=='base' else f'runs/qwen35-finetuned-{args.split}'
                    for r in ('base','finetuned') if r!=args.role]
    for other in prior_stems:
        semantics=Path(other+'-semantic.jsonl')
        if not semantics.exists(): continue
        raw={r['id']:r for r in read_jsonl(other+'.jsonl')}
        for note in read_jsonl(semantics):
            generated=raw[note['id']]
            if note.get('binding_hash')!=binding(generated): continue
            if note.get('reviewer')!='Codex' or not note.get('note'): continue
            caches.append((other,note))
    decisions=[];pending=[]
    for row in rows:
        decision=collect(row,indexed[row['id']],manual,caches) if row.get('raw') is not None else None
        if decision is None:
            pending.append(dict(id=row['id'],category=row['category'],input=row['input'],
                reference=row['reference'],translation=row.get('prediction',{}).get('translation'),
                raw=row.get('raw'),ended=row.get('ended'),error=row.get('error'),generation_hash=fingerprint(row)))
        else: decisions.append(decision)
    write_jsonl(stem+'-semantic.jsonl',decisions)
    write_jsonl(stem+'-reading-packets.jsonl',pending)
    write_json(stem+'-review-progress.json',dict(at=now(),generated=len(rows),expected=len(refs),reviewed=len(decisions),
        pending=len(pending),counts=dict(Counter(d['verdict'] for d in decisions)),
        complete=len(decisions)==len(refs) and Path(stem+'.summary.json').exists(),
        scope='Source-grounded AI reading; exact text cache reuse only. Identity not hidden; no professional human review claim.',release_approved=False))
    print({'generated':len(rows),'reviewed':len(decisions),'pending':len(pending),'counts':dict(Counter(d['verdict'] for d in decisions))})


if __name__=='__main__': main()
