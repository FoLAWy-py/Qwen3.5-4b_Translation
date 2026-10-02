"""Sequential owned training and frozen candidate decoding; manual acceptance follows."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    progress = Path('runs/v15-error-optimization-progress.json')
    if progress.exists():
        raise ValueError('Preserve previous pipeline attempt')
    plan = load('data/prepared/v15-error-repair/plan.json')
    for name in ('runs/v15-training-hardness-job.json','runs/v15-training-triage-decoding-job.json'):
        job = load(name)
        if job['status']!='finished' or job.get('exit_code')!=0:
            raise ValueError('Previous GPU owner not finished successfully')
    state = {'at':now(),'phase':'training','plan_hash':fingerprint(plan),'release_approved':False}
    write_json(progress,state)
    try:
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run',
                        '--module','scripts.train_v15_error_repair'],check=True)
        output = Path(plan['output'])
        metrics, config = load(output/'metrics.json'),load(output/'run_config.json')
        sha = hashlib.sha256((output/'adapter_model.safetensors').read_bytes()).hexdigest()
        if (metrics['plan_hash']!=fingerprint(plan) or config['plan_hash']!=fingerprint(plan)
            or metrics['adapter_sha256']!=sha or metrics['selected_checkpoint']!=16
            or metrics['forward_calls']!=256 or metrics['backward_calls']!=192
            or metrics['full_tokens']!=config['expected_full_tokens'] or metrics['peak_reserved_gib']>6.5
            or [r['step'] for r in metrics['history']]!=list(range(1,17))):
            raise ValueError('Training completion differs from frozen protocol')
        write_json('runs/v15-error-fixed-selection.json',{'at':now(),'training_plan_hash':fingerprint(plan),
            'selected':{'directory':str(output),'step':16,'adapter_sha256':sha},
            'selection':plan['selection'],'release_approved':False})
        previous = load('runs/v14-evaluation-plan.json')
        datasets = previous['datasets']
        for spec in datasets.values():
            if fingerprint(read_jsonl(spec['path']))!=spec['hash']:
                raise ValueError('Frozen heldout source changed')
        parent = next(r for r in previous['targets'] if r['role']=='v7' and r['split']=='public')
        targets = [parent]+[{'role':'v15-error-cpo','split':split,'baseline':False,
            'adapter':str(output),'adapter_sha256':sha,'output':f'runs/v15-error-{split}.jsonl'} for split in datasets]
        write_json('runs/v15-error-evaluation-plan.json',{'at':now(),'training_plan_hash':fingerprint(plan),
            'datasets':datasets,'targets':targets,'semantic_cache_roles':['v7'],'base':plan['base'],
            'prompt_hash':plan['prompt_hash'],'quantization':'nf4','max_length':1024,'max_new_tokens':256,
            'policy':'Actual TRAIN errors only; fixed16 checkpoint; versus best v7; no new base evaluation or release claim.'})
        for target in targets[1:]:
            state.update(phase='decoding',split=target['split'],updated_at=now())
            write_json(progress,state)
            if Path(target['output']).exists():
                raise ValueError('Preserve candidate outputs')
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                '--adapter-dir',target['adapter'],'--input',datasets[target['split']]['path'],
                '--output',target['output'],'--quantization','nf4','--max-length','1024','--max-new-tokens','256'],check=True)
            summary = load(Path(target['output']).with_suffix('.summary.json'))
            if (summary['adapter_sha256']!=sha or summary['data_hash']!=datasets[target['split']]['hash']
                or summary['base']!=plan['base'] or summary['prompt_hash']!=plan['prompt_hash']):
                raise ValueError('Candidate decoding does not bind protocol')
        state.update(phase='individual_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',error=str(exc),updated_at=now())
        raise
    finally:
        write_json(progress,state)


if __name__=='__main__':
    main()
