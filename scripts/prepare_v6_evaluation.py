"""Freeze both matched candidates before semantic generation, without NLL selection."""
import json
from pathlib import Path
from scripts.v6_progress import main as verify_training
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record

def main():
    destination = Path('runs/v6-evaluation-plan.json')
    if destination.exists():
        raise ValueError('Evaluation plan already frozen')
    verify_training()
    report = json.loads(Path('runs/v6-progress.json').read_text(encoding='utf-8'))
    if any(r['state'] != 'trained_semantic_acceptance_pending' for r in report['artifacts'].values()):
        raise ValueError('Both training runs must finish')
    refs = read_jsonl('data/prepared/v4/dev.jsonl')
    if len(refs) != 200 or len({r['group_id'] for r in refs}) != 100:
        raise ValueError('Development coverage changed')
    for row in refs:
        validate_record(row, True, True, purpose='evaluation')
    train = read_jsonl('data/prepared/v6-preference/train.jsonl')
    if {r['group_id'] for r in refs} & {r['group_id'] for r in train}:
        raise ValueError('Semantic development leaked')
    start_summary = json.loads(Path('runs/v5-selected-dev.summary.json').read_text(encoding='utf-8'))
    decoding = {'do_sample':False, 'max_length':1024, 'max_new_tokens':256}
    if (start_summary['adapter_sha256'] != report['protocol']['starting_adapter_sha256']
            or start_summary['data_hash'] != fingerprint(refs) or start_summary['quantization'] != 'nf4'
            or start_summary['prompt_hash'] != fingerprint(SYSTEM_PROMPT) or start_summary['decoding'] != decoding):
        raise ValueError('Starting semantic evidence uses different protocol')
    plan = {'at':now(), 'candidates':report['artifacts'], 'development_hash':fingerprint(refs),
        'prompt_hash':fingerprint(SYSTEM_PROMPT), 'quantization':'nf4', 'decoding':decoding,
        'input':'data/prepared/v4/dev.jsonl',
        'outputs':{'cpo':'runs/v6-cpo-dev.jsonl', 'sft_control':'runs/v6-sft-control-dev.jsonl'},
        'starting_evidence':'runs/v5-selected-dev.jsonl',
        'starting_adapter_sha256':report['protocol']['starting_adapter_sha256'],
        'selection':'Both candidates screened semantically; known16-pair dev objective auxiliary only. Require200-row root review; no default promotion by loss.',
        'limitations':'Known repeatedly used development, not a new release test. Starting generation already recorded with identical weights/input/prompt/NF4/greedy settings; no fresh matched wall-time comparison claim.',
        'release_approved':False}
    write_json(destination, plan)
    print(plan)

if __name__ == '__main__':
    main()
