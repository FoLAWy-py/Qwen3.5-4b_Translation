"""Verify completed shards from both preserved download attempts without overwrite."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from witrans_tools.common import now, write_json


def main():
    assert not Path('runs/qwen35-acquisition.json').exists(), 'Preserve acquisition receipt'
    revision=json.loads(Path('runs/qwen35-download-progress.json').read_text(encoding='utf-8'))['revision']
    info=HfApi(token=False).model_info('Qwen/Qwen3.5-4B',revision=revision,files_metadata=True)
    dest=Path('models/Qwen3.5-4B')
    files=[]
    for sibling in info.siblings:
        if not sibling.rfilename.endswith('.safetensors'): continue
        path=dest/sibling.rfilename
        assert path.stat().st_size==sibling.size
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert digest==sibling.lfs.sha256, 'Official LFS SHA256 mismatch'
        files.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest,verified_official_lfs=True))
        print({'file':path.name,'sha256':digest,'verified':True},flush=True)
    assert len(files)==2
    receipt=dict(model_id=info.id,revision=revision,completed_at=now(),status='complete',license='apache-2.0',
        source='https://huggingface.co/Qwen/Qwen3.5-4B',files=files,
        transport='Shard2 completed original HTTP; shard1 assembled from official cache-busted Range chunks. Both official LFS hashes verified. Original stalled download and range duplicate-file guard preserved.',
        terminology='Official post-trained checkpoint before project fine-tuning, not a pretraining-only Base checkpoint.')
    write_json(dest/'witrans_base.json',receipt)
    write_json('runs/qwen35-acquisition.json',receipt)


if __name__=='__main__': main()
