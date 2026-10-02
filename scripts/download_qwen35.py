"""Download the explicitly requested public checkpoint at an immutable revision."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from witrans_tools.common import now, write_json


def main():
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    dest = Path('models/Qwen3.5-4B')
    receipt = Path('runs/qwen35-acquisition.json')
    assert not receipt.exists(), 'Preserve completed acquisition'
    info = HfApi(token=False).model_info('Qwen/Qwen3.5-4B', files_metadata=True)
    revision = info.sha
    write_json('runs/qwen35-download-progress.json',dict(at=now(),status='downloading',model_id=info.id,
        revision=revision,source='https://huggingface.co/Qwen/Qwen3.5-4B',token=False))
    snapshot_download(repo_id=info.id,revision=revision,local_dir=str(dest),token=False,max_workers=4,
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.md'])
    files = []
    for path in sorted(dest.glob('*.safetensors')):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        files.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest))
    assert files and (dest/'config.json').is_file()
    config = json.loads((dest/'config.json').read_text(encoding='utf-8'))
    assert config['model_type']=='qwen3_5' and config['text_config']['model_type']=='qwen3_5_text'
    record = dict(model_id=info.id,revision=revision,completed_at=now(),status='complete',
        license='apache-2.0',source='https://huggingface.co/Qwen/Qwen3.5-4B',files=files,
        terminology='User-requested base means the official post-trained checkpoint before project fine-tuning, not a separately released pretraining-only Base checkpoint.')
    write_json(dest/'witrans_base.json',record)
    write_json(receipt,record)
    write_json('runs/qwen35-download-progress.json',dict(at=now(),status='complete',revision=revision))
    print({'revision':revision,'weight_files':len(files),'bytes':sum(f['bytes'] for f in files)},flush=True)


if __name__=='__main__':
    main()
