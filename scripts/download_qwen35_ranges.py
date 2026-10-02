"""Cache-busted ranged transport of official immutable LFS objects, hash verified."""
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
import httpx
from huggingface_hub import HfApi
from witrans_tools.common import now, write_json


def main():
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    revision=json.loads(Path('runs/qwen35-download-progress.json').read_text(encoding='utf-8'))['revision']
    info=HfApi(token=False).model_info('Qwen/Qwen3.5-4B',revision=revision,files_metadata=True)
    dest=Path('models/Qwen3.5-4B')
    chunk_size=64*1024**2
    files=[]
    for sibling in info.siblings:
        if not sibling.rfilename.endswith('.safetensors'):
            continue
        size=sibling.size
        sha=sibling.lfs.sha256
        chunks=Path('runs/qwen35-range-chunks')/sibling.rfilename
        chunks.mkdir(parents=True,exist_ok=True)
        count=math.ceil(size/chunk_size)
        def fetch(index):
            start=index*chunk_size;end=min(size,(index+1)*chunk_size)-1
            path=chunks/f'{index:04}.chunk'
            if path.exists():
                assert path.stat().st_size==end-start+1
                return index
            for attempt in range(4):
                partial=chunks/f'{index:04}.attempt{attempt}.partial'
                try:
                    with httpx.Client(follow_redirects=True,timeout=90) as client:
                        with client.stream('GET',f'https://huggingface.co/{info.id}/resolve/{revision}/{sibling.rfilename}',
                            params={'download':'true','witrans_range':str(time.time_ns())},
                            headers={'Range':f'bytes={start}-{end}'}) as response:
                            response.raise_for_status()
                            assert response.status_code==206 and response.headers['content-range'].startswith(f'bytes {start}-{end}/')
                            with partial.open('xb') as stream:
                                for block in response.iter_bytes(chunk_size=1024**2):
                                    stream.write(block)
                    assert partial.stat().st_size==end-start+1
                    partial.rename(path)
                    print({'file':sibling.rfilename,'chunk':index+1,'chunks':count,'bytes':path.stat().st_size},flush=True)
                    return index
                except Exception as exc:
                    print({'file':sibling.rfilename,'chunk':index+1,'attempt':attempt+1,'error_type':type(exc).__name__},flush=True)
                    if attempt==3:
                        raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch,range(count)))
        output=dest/sibling.rfilename
        assert not output.exists(), 'Preserve downloaded weights'
        digest=hashlib.sha256()
        with output.open('xb') as stream:
            for index in range(count):
                with (chunks/f'{index:04}.chunk').open('rb') as part:
                    while block:=part.read(8*1024**2):
                        digest.update(block);stream.write(block)
        assert output.stat().st_size==size and digest.hexdigest()==sha, 'Official LFS fingerprint mismatch'
        files.append(dict(file=sibling.rfilename,bytes=size,sha256=sha,verified_official_lfs=True))
    receipt=dict(model_id=info.id,revision=revision,completed_at=now(),status='complete',license='apache-2.0',
        source='https://huggingface.co/Qwen/Qwen3.5-4B',files=files,
        transport='Official cache-busted HTTP Range; all assembled hashes match official LFS metadata.',
        terminology='Official post-trained checkpoint before project fine-tuning, not a pretraining-only Base checkpoint.')
    assert files
    write_json(dest/'witrans_base.json',receipt)
    write_json('runs/qwen35-acquisition.json',receipt)
    write_json('runs/qwen35-range-download-progress.json',dict(at=now(),status='complete',revision=revision))
    print({'status':'complete','revision':revision,'files':len(files)},flush=True)


if __name__=='__main__':
    main()
