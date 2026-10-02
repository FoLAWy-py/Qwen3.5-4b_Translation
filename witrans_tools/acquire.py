"""Public immutable Qwen3.5 download with official LFS checksum verification."""
import hashlib
import json
from pathlib import Path

from .common import now, write_json
from .qwen35 import MODEL_ID, REVISION


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    base=Path(args.base_dir)
    receipt=base/'witrans_base.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        if prior.get('model_id')!=MODEL_ID or prior.get('revision')!=REVISION:
            raise ValueError('Destination belongs to another model/revision')
    info=HfApi(token=False).model_info(MODEL_ID,revision=REVISION,files_metadata=True)
    if info.sha!=REVISION:raise ValueError('Official immutable revision mismatch')
    snapshot_download(MODEL_ID,revision=REVISION,token=False,local_dir=str(base),max_workers=2,
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.md','LICENSE*'])
    files=[]
    for sibling in info.siblings:
        if not sibling.rfilename.endswith('.safetensors'):continue
        path=base/sibling.rfilename
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if not sibling.lfs or digest!=sibling.lfs.sha256 or path.stat().st_size!=sibling.size:
            raise ValueError(f'Official weight checksum/size mismatch: {sibling.rfilename}')
        files.append(dict(file=sibling.rfilename,bytes=path.stat().st_size,sha256=digest,verified_official_lfs=True))
    if len(files)!=2:raise ValueError('Expected both official weight shards')
    write_json(receipt,dict(model_id=MODEL_ID,revision=REVISION,status='complete',completed_at=now(),
        source=f'https://huggingface.co/{MODEL_ID}/tree/{REVISION}',license='apache-2.0',files=files))
    print(f'Verified {MODEL_ID}@{REVISION} in {base}')
