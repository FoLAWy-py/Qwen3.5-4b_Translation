"""Freeze critical-span ablation against already trained standard CPO/SFT."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import validated_annotations

def main():
    destination = Path('runs/v7-trial-plan.json')
    if destination.exists():
        raise ValueError('Trial plan already frozen')
    rows = read_jsonl('data/prepared/v6-preference/train.jsonl')
    annotations_path = Path('data/prepared/v7-critical-spans/annotations.json')
    annotations = json.loads(annotations_path.read_text(encoding='utf-8'))
    validated_annotations(rows, annotations)
    standard = json.loads(Path('models/witrans-4b-v6-cpo/run_config.json').read_text(encoding='utf-8'))
    if standard['train_hash'] != fingerprint(rows) or standard['steps'] != 22 or standard['accumulation'] != 8:
        raise ValueError('Matched standard CPO changed')
    output = 'models/witrans-4b-v7-critical-cpo'
    if Path(output).exists():
        raise ValueError('Candidate must not predate frozen trial')
    plan = {'at':now(), 'method':'critical_span_cpo', 'output':output,
        'data_dir':'data/prepared/v6-preference', 'starting_adapter':'models/witrans-4b-v5-sft/checkpoint-85',
        'starting_adapter_sha256':standard['starting_adapter_sha256'], 'train_hash':standard['train_hash'],
        'dev_hash':standard['dev_hash'], 'annotations':str(annotations_path),
        'annotation_hash':annotations['annotation_hash'], 'annotated_rows':len(annotations['annotations']),
        'steps':22, 'accumulation':8, 'learning_rate':1e-5, 'beta':.1, 'seed':42, 'quantization':'nf4',
        'change':'Chosen normalized NLL weighted3 on explicit reviewed spans; summed positive/negative logp unchanged. Remaining69 pairs and all16 dev pairs use unweighted NLL.',
        'controls':['models/witrans-4b-v6-cpo','models/witrans-4b-v6-sft-control'],
        'selection':'Common known16-pair objective auxiliary; require200-row semantic comparison to v5, standard CPO and SFT. No loss-only promotion.',
        'limitations':'Single seed, small partly authored preference pool; no innovation efficacy claim without group-level interval excluding zero and three-seed replication.',
        'release_approved':False}
    write_json(destination, plan)
    print(plan)

if __name__ == '__main__':
    main()
