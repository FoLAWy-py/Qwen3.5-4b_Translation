"""Correct source-lineage hash metadata; preserve frozen training content."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json

root = Path('data/prepared/v10-source-family-checked')
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
actual = fingerprint(read_jsonl('data/prepared/v10-source/reviewed-partial.jsonl'))
audit = json.loads(Path('runs/v10-source-overlap-audit.json').read_text(encoding='utf-8'))
if actual!=audit['reviewed_hash'] or fingerprint(read_jsonl(root/'train.jsonl'))!=manifest['train_hash']:
    raise ValueError('Frozen content changed')
old = manifest['reviewed_source_hash']
if old==actual:
    raise ValueError('Already corrected')
write_json(root/'manifest.before-source-hash-fix.json',manifest)
manifest['reviewed_source_hash'] = actual
manifest['metadata_correction'] = {'at':now(),'previous_source_hash':old,
    'reason':'Original source hash was computed after mutating in-memory family copies. Rebind original reviewed source before family annotation; frozen train.jsonl remains byte-for-byte unchanged.'}
write_json(root/'manifest.json',manifest)
write_json('runs/v10-source-family-review.json',manifest)
print({'corrected_source_hash':actual,'unchanged_train_hash':manifest['train_hash']})
