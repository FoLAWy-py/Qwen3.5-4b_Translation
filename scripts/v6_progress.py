"""Report verified experimental artifacts without promoting by training loss."""
import hashlib
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json

def main():
    root = Path('data/prepared/v6-preference')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    protocol = manifest['protocol']
    artifacts = {}
    for mode in protocol['modes']:
        directory = Path('models') / ('witrans-4b-v6-' + ('cpo' if mode == 'cpo' else 'sft-control'))
        result = {'directory':str(directory), 'state':'not_started'}
        config_path = directory / 'run_config.json'
        metrics_path = directory / 'metrics.json'
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            if (cfg['mode'] != mode or cfg['steps'] != protocol['steps']
                    or cfg['accumulation'] != protocol['accumulation']
                    or cfg['starting_adapter_sha256'] != protocol['starting_adapter_sha256']
                    or cfg['train_hash'] != manifest['train']['hash'] or cfg['dev_hash'] != manifest['dev']['hash']):
                raise ValueError('Trial configuration drift')
            for split in ('train','dev'):
                if fingerprint(read_jsonl(directory / f'{split}_snapshot.jsonl')) != manifest[split]['hash']:
                    raise ValueError('Training snapshot changed')
            result['state'] = 'started_completion_not_yet_recorded'
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            metadata = json.loads((directory / 'witrans_adapter.json').read_text(encoding='utf-8'))
            digest = hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest()
            if (digest != metadata['adapter_sha256'] or len(metrics['history']) != protocol['steps']
                    or metadata['parent_adapter_sha256'] != protocol['starting_adapter_sha256']):
                raise ValueError('Incomplete or modified adapter')
            result.update(state='trained_semantic_acceptance_pending', adapter_sha256=digest,
                dev_cpo_loss=metrics['eval_cpo_loss'], dev_preference_accuracy=metrics['dev_preference_accuracy'],
                train_seconds=metrics['train_seconds'], peak_reserved_gib=metrics['peak_reserved_gib'])
        artifacts[mode] = result
    train = read_jsonl(root / 'train.jsonl')
    negative_kinds = {'naturally_generated':sum('negative_provenance' in r for r in train),
        'authored_semantic_contrasts':sum('negative_provenance' not in r for r in train)}
    both_trained = all(r['state'] == 'trained_semantic_acceptance_pending' for r in artifacts.values())
    cpo_review_path = Path('runs/v6-cpo-candidate-acceptance.json')
    cpo_review = json.loads(cpo_review_path.read_text(encoding='utf-8')) if cpo_review_path.exists() else None
    comparison_path = Path('runs/v6-model-acceptance.json')
    comparison = json.loads(comparison_path.read_text(encoding='utf-8')) if comparison_path.exists() else None
    report = {'at':now(), 'training':manifest['train'], 'development':manifest['dev'],
        'cpo_development_result':{'counts':cpo_review['counts'], 'gate':cpo_review['development_promotion_gates']['cpo'],
            'comparison_complete':False} if cpo_review else None,
        'stage':'complete_development_comparison' if comparison else ('both_trained_semantic_acceptance_pending' if both_trained else 'paired_training_incomplete'),
        'development_comparison':{'counts':comparison['counts'], 'gates':comparison['development_promotion_gates']} if comparison else None,
        'negative_kinds':negative_kinds,
        'scope':'Mixed pool:60 previously approved authored contrasts plus25 natural model-error pairs; not85 natural negatives. Compares these negatives versus matched positive-only training, not natural-vs-authored negative selection.',
        'protocol':protocol, 'artifacts':artifacts, 'default_promoted':False, 'release_approved':False,
        'next':('Improve semantic quality using training-only material or matched loss ablation; both v6 candidates failed development promotion.' if comparison
            else ('Generate and individually review both candidates on known200-row semantic development; objective alone does not select a release candidate.'
            if both_trained else 'Complete both matched training runs before freezing semantic generation.'))}
    write_json('runs/v6-progress.json', report)
    print(report)

if __name__ == '__main__':
    main()
