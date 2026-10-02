"""Retry only failed blind source API batches, preserving originals."""
from scripts.independent_source_pipeline_v2 import ROOT,batch_job,cards
from witrans_tools.common import append_jsonl,fingerprint,now,read_jsonl,write_json

def main():
    allcards=cards();existing={b['batch_index'] for b in read_jsonl(ROOT/'batches.jsonl')}
    for error in read_jsonl(ROOT/'errors.jsonl'):
        index=error['batch_index']
        if index in existing:continue
        result=batch_job(allcards[index:index+10])
        append_jsonl(ROOT/'batches.jsonl',dict(batch_index=index,retry_of=error,**result));existing.add(index)
        print(dict(recovered_batch=index),flush=True)
    write_json(ROOT/'retry-receipt.json',dict(at=now(),batches=len(existing),candidate_outputs_read=False,
        original_errors_preserved=True,status='local_full_review_pending'))

if __name__=='__main__':main()
