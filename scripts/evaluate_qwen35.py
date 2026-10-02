"""Raw matched NF4 generation for the requested Qwen3.5 comparison."""
import argparse
import json
import time
from pathlib import Path
from witrans import parse_translation
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json
from witrans_tools.data import check_constraints
from witrans_tools.qwen35 import Qwen35Translator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter-dir')
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-new-tokens',type=int,default=256)
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists() and not dest.with_suffix('.summary.json').exists(), 'Preserve generation attempt'
    import torch
    inputs=read_jsonl(args.input)
    torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter()
    translator=Qwen35Translator(adapter_dir=args.adapter_dir)
    loaded=time.perf_counter()-began
    rows=[]
    for ref in inputs:
        row=dict(id=ref['id'],category=ref['category'],input=ref['input'],reference=ref['output'],semantic_review='pending')
        started=time.perf_counter()
        try:
            raw,ended=translator.generate_raw(**ref['input'],max_new_tokens=args.max_new_tokens)
            row.update(raw=raw,ended=ended,**translator.last_generation_stats)
            row['prediction']=parse_translation(raw)
            row['constraint_warnings']=check_constraints(ref['input']['text'],row['prediction']['translation'])
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc))
            if not row.get('raw') and len(rows)<2:
                row['seconds']=time.perf_counter()-started
                append_jsonl(dest,row)
                raise RuntimeError('Generation interface failed before any raw output; preserve attempt and stop') from exc
        torch.cuda.synchronize()
        row['seconds']=time.perf_counter()-started
        append_jsonl(dest,row);rows.append(row)
        print({'id':row['id'],'json':'prediction' in row,'eos':row.get('ended'),'seconds':row['seconds']},flush=True)
    cpu=sum(p.numel() for p in translator.model.parameters() if p.device.type=='cpu')
    write_json(dest.with_suffix('.summary.json'),dict(at=now(),model_id='Qwen/Qwen3.5-4B',
        base=json.loads(Path('models/Qwen3.5-4B/witrans_base.json').read_text(encoding='utf-8')),
        adapter_dir=args.adapter_dir,adapter_sha256=translator.adapter_sha256,
        prompt_hash=fingerprint(__import__('witrans').SYSTEM_PROMPT),data_hash=fingerprint(inputs),
        generation_hash=fingerprint(rows),rows=len(rows),quantization='nf4',
        decoding=dict(do_sample=False,max_length=1024,max_new_tokens=args.max_new_tokens),
        loading_info=translator.loading_info,load_seconds=loaded,cpu_parameter_count=cpu,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        valid_json=sum('prediction' in r for r in rows),ended=sum(bool(r.get('ended')) for r in rows),
        errors=sum('error' in r for r in rows),release_approved=False))


if __name__=='__main__':
    main()
