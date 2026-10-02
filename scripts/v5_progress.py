"""Report persisted next-run evidence without interpreting training loss as quality."""
import json
import hashlib
from pathlib import Path
from witrans_tools.common import now, write_json

def main():
    manifest = json.loads(Path('data/prepared/v5-mixed/manifest.json').read_text(encoding='utf-8'))
    output = Path('models/witrans-4b-v5-sft')
    selection_path = Path('runs/v5-selection.json')
    selected = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else None
    checkpoints = []
    for directory in output.glob('checkpoint-*'):
        metadata_path = directory / 'witrans_adapter.json'
        weight_path = directory / 'adapter_model.safetensors'
        if not metadata_path.exists() or not weight_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        if digest != metadata['adapter_sha256']:
            raise ValueError('Saved checkpoint hash mismatch')
        checkpoints.append({'directory': str(directory), 'dev_nll': metadata['selected_dev_loss'], 'adapter_sha256': digest})
    candidate_path = Path('runs/v5-candidate-acceptance.json')
    candidate = json.loads(candidate_path.read_text(encoding='utf-8')) if candidate_path.exists() else None
    stage = candidate['stage'] if candidate else ('training_finished_semantic_review_pending' if selected else 'training_started')
    comparison_path = Path('runs/v5-model-acceptance.json')
    comparison = json.loads(comparison_path.read_text(encoding='utf-8')) if comparison_path.exists() else None
    if comparison:
        stage = comparison['status']
    remaining = ['Compare source-group paired quality with starting model using fresh matching protocol evidence',
        'Create new600-row/300-group release test, audit sources and all three models, verify context and multi-sentence coverage',
        'Complete fixed short-input3-run performance and1024-token budget checks against full goal policy']
    if not candidate:
        remaining.insert(0, 'Generate selected200-row known development outputs and individually review')
    if comparison:
        remaining.pop(0)
        if not comparison['development_promotion_gate']:
            remaining.insert(0, 'Improve semantic quality using new independent training material; current known-development screening failed')
    report = {'at': now(), 'stage': stage,
        'training': manifest['train'], 'development': manifest['dev'], 'configuration': manifest['configuration'],
        'run_config_exists': (output / 'run_config.json').exists(),
        'saved_checkpoints': checkpoints, 'selection': selected,
        'default_promoted': False, 'release_approved': False,
        'candidate_verdicts':candidate['verdicts'] if candidate else None,
        'paired_stats':comparison['paired_stats'] if comparison else None,
        'remaining': remaining}
    write_json('runs/v5-progress.json', report)
    print(report)

if __name__ == '__main__':
    main()
