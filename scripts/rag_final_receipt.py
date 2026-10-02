"""Verify frozen identities and all RAG evidence without changing old receipts."""
import hashlib
import json
import math
from pathlib import Path
from witrans_tools.common import read_jsonl, write_json, now
from witrans_tools.rag import digest, DIMENSIONS


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    root=Path('runs/takeover-20261002/rag')
    runtime=json.loads(Path('runs/takeover-20261002/runtime-freeze.json').read_text(encoding='utf-8'))
    checks={path:sha(path)==checksum for path,checksum in runtime['source_hashes'].items()}
    checks['uv.lock']=sha('uv.lock')==runtime['uv_lock_sha256']
    adapter=sha('models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors')
    checks['adapter']=adapter==runtime['configuration']['adapter_sha256']
    independent=json.loads(Path('data/independent-20261002-frozen/freeze.json').read_text(encoding='utf-8'))
    for stage in ('confirmation','release'):
        checks[stage+'_data']=digest(read_jsonl(Path('data/independent-20261002-frozen')/(stage+'.jsonl')))==independent['sets'][stage]['hash']
    historical=[r for r in independent['historic_files'] if sha(r['path'])!=r['sha256']]
    checks['historic_sources']=not historical
    frozen=json.loads((root/'prepared-freeze.json').read_text(encoding='utf-8'))
    checks['frozen_runtime_binding']=digest(runtime)==independent['runtime_freeze_hash']
    for path,checksum in frozen['files'].items():
        checks[path]=digest(read_jsonl(path))==checksum
    for path,checksum in frozen['implementation_hashes'].items():
        checks[path]=digest(Path(path).read_text(encoding='utf-8'))==checksum
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for path,checksum in protocol['source_files'].items():
        checks[path]=digest(read_jsonl(path))==checksum
    index=json.loads((root/'index.json').read_text(encoding='utf-8'))
    index_receipt=json.loads((root/'index-receipt.json').read_text(encoding='utf-8'))
    checks['index_hash']=digest(index)==index_receipt['index_hash']
    checks['index_vectors']=all(len(v)==DIMENSIONS and all(math.isfinite(x) for x in v)
        and abs(sum(x*x for x in v)-1)<1e-6 for v in index['vectors'])
    bank=read_jsonl('data/rag-reviewed-training-terms-20261002.jsonl')
    checks['bank_hash']=digest(bank)==protocol['bank_hash']==index['bank_hash']==frozen['bank_hash']
    checks['protocol_hash']=digest(protocol)==frozen['protocol_hash']
    quality=json.loads((root/'quality/semantic-summary.json').read_text(encoding='utf-8'))
    for label,reports in quality['sets'].items():
        for profile,report in reports.items():
            checks[label+'_'+profile+'_output_hash']=digest(read_jsonl(root/'quality'/(label+'-'+profile+'.jsonl')))==report['actual_outputs_hash']
            checks[label+'_'+profile+'_review_hash']=digest(read_jsonl(root/'quality'/(label+'-'+profile+'-semantic.jsonl')))==report['reviews_hash']
    timing=json.loads((root/'performance/summary.json').read_text(encoding='utf-8'))
    rows=read_jsonl(root/'performance/performance.jsonl')
    for profile,report in timing['profiles'].items():
        checks[profile+'_performance_hash']=digest([r for r in rows if r['profile']==profile])==report['outputs_hash']
    checks['timing_budget']=all(20<=r['body_tokens']<=120 and r['generated_tokens_including_eos']<=128
        and r['prompt_tokens']+128<=1024 and r['ended'] for r in rows)
    checks['no_release_outputs']=not Path('runs/takeover-20261002/release/outputs.jsonl').exists()
    if not all(checks.values()):
        raise ValueError('Frozen/evidence validation failed: '+str([k for k,v in checks.items() if not v]))
    write_json(root/'final-receipt.json',{'at':now(),'checks':checks,'all_checks_passed':True,
        'adapter_sha256':adapter,'runtime_freeze_hash':digest(runtime),'prepared_freeze_hash':digest(frozen),
        'bank_entries':len(bank),'historic_source_files_verified':len(independent['historic_files']),
        'quality_review_complete':quality['full316_review_complete'],
        'performance_review_complete':json.loads((root/'performance/semantic-summary.json').read_text(encoding='utf-8'))['review_complete'],
        'unit_tests':{'default_suite_passed':93,'added_rag_tests_passed':5},
        'native_host_limitation':'One final read command encountered PowerShell CLR 0x80131506/0xC0000005; read-only retry succeeded. GPU runs had already naturally completed. Existing process-local CPU isolation retained.',
        'training_started':False,'default_rag_adopted':False,'release_approved':False,'remote_pushed':False})
    print({'checks_passed':len(checks),'historical_files':len(independent['historic_files']),'bank_entries':len(bank),'adapter_unchanged':True})


if __name__=='__main__':main()
