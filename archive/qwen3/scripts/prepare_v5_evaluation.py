"""Bind next development generation to finished immutable training evidence."""
import hashlib
import json
from pathlib import Path
from archive.qwen3.runtime import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    destination = Path('runs/v5-evaluation-plan.json')
    if destination.exists():
        raise ValueError('Evaluation plan already frozen')
    root = Path('models/witrans-4b-v5-sft').resolve()
    selection = load('runs/v5-selection.json')
    manifest = load('data/prepared/v5-mixed/manifest.json')
    config = load(root / 'run_config.json')
    metrics = load(root / 'metrics.json')
    candidate = selection['selected']
    directory = Path(candidate['directory']).resolve()
    if directory.parent != root:
        raise ValueError('Selected checkpoint outside current run')
    if candidate != min(selection['candidates'], key=lambda row:row['dev_nll']) or metrics['selected'] != candidate:
        raise ValueError('Frozen lowest-NLL selection mismatch')
    if len(metrics['history']) != manifest['configuration']['steps'] or metrics['history'][-1]['step'] != manifest['configuration']['steps']:
        raise ValueError('Training not complete')
    for name in ('train', 'dev'):
        rows = read_jsonl(root / f'{name}_snapshot.jsonl')
        if fingerprint(rows) != config[f'{name}_hash'] or fingerprint(rows) != manifest[name]['hash']:
            raise ValueError('Training snapshot not frozen corpus')
        if fingerprint(read_jsonl(f'data/prepared/v5-mixed/{name}.jsonl')) != fingerprint(rows):
            raise ValueError('Prepared corpus changed')
        for row in rows:
            validate_record(row, True, True, purpose='training' if name == 'train' else 'evaluation')
    metadata = load(directory / 'witrans_adapter.json')
    actual_sha = hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest()
    if actual_sha != candidate['adapter_sha256'] or metadata['adapter_sha256'] != actual_sha:
        raise ValueError('Chosen weights changed')
    if metadata['prompt_hash'] != fingerprint(SYSTEM_PROMPT) or config['quantization'] != 'nf4':
        raise ValueError('Protocol or precision mismatch')
    start = Path('models/witrans-4b-v2-selected/adapter')
    start_sha = hashlib.sha256((start / 'adapter_model.safetensors').read_bytes()).hexdigest()
    if config['parent_adapter_sha256'] != start_sha:
        raise ValueError('Starting model changed')
    plan = {'at':now(), 'selected':candidate, 'starting_adapter_sha256':start_sha,
        'training_hash':config['train_hash'], 'development_hash':config['dev_hash'], 'prompt_hash':fingerprint(SYSTEM_PROMPT),
        'quantization':'nf4', 'decoding':{'do_sample':False, 'max_length':1024, 'max_new_tokens':256},
        'input':'data/prepared/v5-mixed/dev.jsonl',
        'outputs':{'selected':'runs/v5-selected-dev.jsonl', 'start':'runs/v5-start-dev.jsonl'},
        'policy':'New generation of both models on known development only; original weights selected by NLL before either output. Individually review candidate. Prior starting review reusable only for exact semantic and structural evidence. No release approval or default promotion.'}
    write_json(destination, plan)
    print(plan)

if __name__ == '__main__':
    main()
