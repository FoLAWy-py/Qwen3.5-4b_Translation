"""Train one direct optimization candidate then generate only necessary v7 comparisons."""
import json
import subprocess
import sys
from pathlib import Path
from archive.qwen3.scripts.run_v12_factorial_evaluation import validate_selection
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def main():
    plan_path = Path('data/prepared/v13-public-optimization/plan.json')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    state_path = Path('runs/v13-optimization-progress.json')
    if state_path.exists():
        raise ValueError('Preserve existing optimization attempt')
    if len(plan['arms']) != 1 or plan['arms'][0]['start'] != 'v7':
        raise ValueError('Optimization must start from current best adapter')
    arm = plan['arms'][0]
    state = {'at':now(),'phase':'training_current_best','plan_hash':fingerprint(plan),
             'completed_generations':[],'release_approved':False}
    write_json(state_path,state)
    try:
        subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','--module',
                        'scripts.train_v12_factorial','--plan',str(plan_path),'--arm',arm['name']],check=True)
        selected = validate_selection(plan,arm)
        datasets = {'known':plan['dev'],'public':plan['public_dev']}
        for spec in datasets.values():
            if fingerprint(read_jsonl(spec['path'])) != spec['hash']:
                raise ValueError('Frozen optimization evaluation set changed')
        targets = [{'role':'v7','split':'public','baseline':False,'adapter':plan['starting_adapter'],
                    'adapter_sha256':plan['starting_adapter_sha256'],'output':'runs/v13-v7-public.jsonl'}]
        targets += [{'role':arm['name'],'split':split,'baseline':False,'adapter':selected['directory'],
                     'adapter_sha256':selected['adapter_sha256'],'output':f'runs/v13-candidate-{split}.jsonl'}
                    for split in datasets]
        if any(Path(target['output']).exists() for target in targets):
            raise ValueError('Preserve existing generation evidence')
        evaluation = {'at':now(),'training_plan_hash':fingerprint(plan),'datasets':datasets,
                      'semantic_cache_roles':['v7'],
                      'base':plan['base'],'prompt_hash':plan['prompt_hash'],'targets':targets,
                      'quantization':'nf4','max_length':1024,'max_new_tokens':256,
                      'policy':'Current best v7 versus direct candidate; no new original-base comparison; known/auxiliary DEV only, no release claim.'}
        write_json('runs/v13-evaluation-plan.json',evaluation)
        for target in targets:
            state.update(phase='generating_candidate_comparison',current=target['role']+'/'+target['split'],updated_at=now())
            write_json(state_path,state)
            subprocess.run([sys.executable,'-X','utf8','-u','-m','scripts.low_cpu_run','evaluate',
                            '--adapter-dir',target['adapter'],'--input',datasets[target['split']]['path'],
                            '--output',target['output'],'--quantization','nf4','--max-length','1024',
                            '--max-new-tokens','256'],check=True)
            summary = json.loads(Path(target['output']).with_suffix('.summary.json').read_text(encoding='utf-8'))
            if (summary['adapter_sha256'] != target['adapter_sha256'] or summary['prompt_hash'] != plan['prompt_hash']
                    or summary['data_hash'] != datasets[target['split']]['hash']):
                raise ValueError('Generation protocol or weights changed')
            state['completed_generations'].append(state['current'])
        state.update(phase='individual_semantic_review_pending',updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence',updated_at=now(),error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(state_path,state)


if __name__ == '__main__':
    main()
