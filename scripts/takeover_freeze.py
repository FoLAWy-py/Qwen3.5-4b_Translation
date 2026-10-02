"""Hash local artifacts and initial protocol before any independent model outputs."""
import hashlib
import json
from pathlib import Path
import subprocess

from witrans_tools.common import now, write_json, fingerprint
from witrans_tools.qwen35 import validate_identity, DEFAULT_ADAPTER
from witrans_tools.protocol import SYSTEM_PROMPT


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    base=Path('models/Qwen3.5-4B')
    receipt,digest=validate_identity(base,DEFAULT_ADAPTER,1024)
    checks=[]
    for row in receipt['files']:
        path=base/row['file'];actual=sha(path)
        if actual!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError(f'Base shard mismatch: {path}')
        checks.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size))
    old=[]
    for path in sorted(Path('data').rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.md','.csv','.tsv') and 'independent-test-protocol' not in path.name:
            old.append(dict(path=str(path),sha256=sha(path)))
    protocol=Path('data/independent-test-protocol-20261002.json')
    write_json('runs/takeover-20261002/model-freeze.json',dict(at=now(),adapter_sha256=digest,
        base_revision=receipt['revision'],base_weights=checks,prompt_hash=fingerprint(SYSTEM_PROMPT),
        base_small_files={str(p):sha(p) for p in sorted(base.iterdir()) if p.is_file() and p.suffix in ('.json','.jinja','.txt')},
        independent_protocol_sha256=sha(protocol),historic_source_inventory=old,
        historic_source_inventory_hash=fingerprint(old),weights_frozen=True,training_authorized=False,
        runtime_frozen=False,reason='Runtime experiments still under old-development-only selection',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),release_approved=False))
    print('Verified both official base shards, v2 weights, prompt and historical-source inventory.',flush=True)


if __name__=='__main__':main()
