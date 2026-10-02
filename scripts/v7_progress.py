"""Verify the ongoing span ablation's actual artifacts and live project processes."""
import hashlib
import json
from pathlib import Path
import psutil
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import validated_annotations

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    plan = load('runs/v7-trial-plan.json')
    directory = Path(plan['output'])
    report = {'at':now(), 'state':'configuration_not_yet_saved', 'plan':plan,
        'default_promoted':False, 'release_approved':False}
    processes = []
    for process in psutil.process_iter(['pid','name','cmdline','cwd']):
        try:
            info = process.info
            args = info['cmdline'] or []
            if ('python' in (info['name'] or '').lower() and Path(info['cwd'] or '.').resolve()==Path.cwd().resolve()
                    and ('scripts.train_v7' in args or plan['output'] in args)):
                processes.append({'pid':info['pid'],'status':process.status()})
        except (psutil.Error,OSError):
            continue
    report['live_training_processes'] = processes
    if (directory / 'run_config.json').exists():
        config = load(directory / 'run_config.json')
        expected = {'mode':'cpo','method':'critical_span_cpo','steps':plan['steps'],
            'accumulation':plan['accumulation'],'starting_adapter_sha256':plan['starting_adapter_sha256'],
            'train_hash':plan['train_hash'],'dev_hash':plan['dev_hash'],'annotation_hash':plan['annotation_hash'],
            'learning_rate':plan['learning_rate'],'beta':plan['beta'],'seed':plan['seed'],'quantization':'nf4'}
        if any(config.get(key)!=value for key,value in expected.items()):
            raise ValueError('Ablation configuration drift')
        train = read_jsonl(directory / 'train_snapshot.jsonl')
        dev = read_jsonl(directory / 'dev_snapshot.jsonl')
        if fingerprint(train)!=plan['train_hash'] or fingerprint(dev)!=plan['dev_hash']:
            raise ValueError('Ablation data changed')
        annotations = load(directory / 'critical_spans_snapshot.json')
        validated_annotations(train, annotations)
        if annotations['annotation_hash']!=plan['annotation_hash']:
            raise ValueError('Ablation annotation changed')
        report['state'] = 'started_completion_not_recorded'
    if (directory / 'metrics.json').exists():
        metrics = load(directory / 'metrics.json')
        metadata = load(directory / 'witrans_adapter.json')
        sha = hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest()
        if (len(metrics['history'])!=plan['steps'] or metadata['adapter_sha256']!=sha
                or metadata['training_method']!='critical_span_cpo'
                or metadata['parent_adapter_sha256']!=plan['starting_adapter_sha256']):
            raise ValueError('Incomplete ablation or altered adapter')
        report.update(state='training_finished_semantic_review_pending', adapter_sha256=sha,
            metrics=metrics, next='Freeze200-row semantic generation before examining outputs; compare with v5 and both v6 controls. Full goal remains unproven.')
    acceptance_path = Path('runs/v7-model-acceptance.json')
    if acceptance_path.exists():
        acceptance = load(acceptance_path)
        if not acceptance['comparison_complete']:
            raise ValueError('Partial acceptance cannot mark comparison complete')
        report.update(state='complete_development_comparison',
            semantic_counts=acceptance['counts'], development_gates=acceptance['development_promotion_gates'],
            next='Improve semantic quality and complete performance checks; current candidate failed development promotion, release goal remains unproven.')
    write_json('runs/v7-progress.json',report)
    print(report)

if __name__ == '__main__':
    main()
