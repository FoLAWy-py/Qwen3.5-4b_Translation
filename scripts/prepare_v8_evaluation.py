"""Verify completed immutable SFT training before freezing development generation."""
import hashlib
import json
import argparse
from pathlib import Path
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',choices=('v8','v11'),default='v8')
    args = parser.parse_args()
    version = args.version
    destination = Path(f'runs/{version}-evaluation-plan.json')
    if destination.exists():
        raise ValueError('Preserve frozen generation protocol')
    root = Path(f'models/witrans-4b-{version}-sft')
    manifest = load(f'data/prepared/{version}-mixed/manifest.json')
    config = load(root / 'run_config.json')
    metrics = load(root / 'metrics.json')
    selection = load(f'runs/{version}-selection.json')
    cfg = manifest['configuration']
    for key,value in cfg.items():
        if config.get(key)!=value:
            raise ValueError('Training configuration differs from frozen plan')
    for split in ('train','dev'):
        source = read_jsonl(f'data/prepared/{version}-mixed/{split}.jsonl')
        snapshot = read_jsonl(root / f'{split}_snapshot.jsonl')
        if (fingerprint(source)!=manifest[split]['hash'] or fingerprint(snapshot)!=manifest[split]['hash']
                or config[split+'_hash']!=manifest[split]['hash']):
            raise ValueError('Actual training/development snapshot differs')
    parent = Path(cfg['starting_adapter'])
    if (digest(parent / 'adapter_model.safetensors')!=cfg['starting_adapter_sha256']
            or config['parent_adapter_sha256']!=cfg['starting_adapter_sha256']):
        raise ValueError('Frozen parent weight differs')
    if ([r['step'] for r in metrics['history']]!=list(range(1,cfg['steps']+1))
            or config['quantization']!='nf4' or config['max_length']!=1024):
        raise ValueError('Actual update history or runtime differs')
    candidates = selection['candidates']
    if [r['step'] for r in candidates]!=cfg['checkpoint_steps']:
        raise ValueError('Frozen checkpoints not completed')
    for candidate in candidates:
        directory = root / f"checkpoint-{candidate['step']}"
        metadata = load(directory / 'witrans_adapter.json')
        if (Path(candidate['directory']).resolve()!=directory.resolve()
                or digest(directory / 'adapter_model.safetensors')!=candidate['adapter_sha256']
                or metadata['adapter_sha256']!=candidate['adapter_sha256']
                or metadata['parent_adapter_sha256']!=cfg['starting_adapter_sha256']
                or metadata['prompt_hash']!=fingerprint(SYSTEM_PROMPT)
                or metadata['selected_dev_loss']!=candidate['dev_nll']):
            raise ValueError('Actual checkpoint identity or NLL differs')
    selected = min(candidates,key=lambda r:r['dev_nll'])
    if selection['selected']!=selected or metrics['selected']!=selected or selection['dev_hash']!=manifest['dev']['hash']:
        raise ValueError('Actual selection differs from frozen NLL rule')
    previous = load('runs/v7-evaluation-plan.json')
    prior_summary = load('runs/v7-critical-dev.summary.json')
    if (prior_summary['adapter_sha256']!=cfg['starting_adapter_sha256']
            or prior_summary['data_hash']!=manifest['dev']['hash']
            or prior_summary['prompt_hash']!=fingerprint(SYSTEM_PROMPT)
            or prior_summary['quantization']!='nf4' or prior_summary['decoding']!=previous['decoding']):
        raise ValueError('Cached starting generation protocol differs')
    description = ('758条/346来源组，新增完整80条已审核多句正例并回放678条；从v7继续48次NF4 SFT更新，24/48检查点按已知开发答案/EOS NLL选择。'
        if version=='v8' else '992条/459来源组，758条回放加58条v9约束及176条v10逐条验收来源；从v7继续64次NF4 SFT更新，学习率5e-6，32/64检查点按已知开发答案/EOS NLL选择。数据和学习率同时改变，不作单变量算法结论。')
    plan = {'at':now(),'candidates':{'sft':{**selected,'method':'continued_multisentence_sft' if version=='v8' else 'continued_broader_low_lr_sft'}},
        'candidate_roles':['sft'],'evidence':{'start':'v7-critical','sft':f'{version}-selected'},
        'input':'data/prepared/v4/dev.jsonl','development_hash':manifest['dev']['hash'],
        'prompt_hash':fingerprint(SYSTEM_PROMPT),'quantization':'nf4','decoding':previous['decoding'],
        'outputs':{'sft':f'runs/{version}-selected-dev.jsonl'},'starting_adapter_sha256':cfg['starting_adapter_sha256'],
        'starting_evidence':'runs/v7-critical-dev.jsonl','starting_label':'v7起点',
        'metadata_update_roles':['sft'],'report_stem':f'runs/{version}-model-acceptance','report_prefix':f'runs/{version}',
        'report_title':'# v8完整多句正例SFT开发验收' if version=='v8' else '# v11扩大来源及降低学习率SFT开发验收',
        'report_training_description':description,
        'acceptance_scope':'Repeatedly used200-row/100-group development only. Cached v7 starting outputs match source/weights/prompt/NF4/eager/greedy settings; no fresh matched timing or independent release claim.',
        'release_approved':False}
    if version=='v11':
        plan['additional_review_cache_stems'] = ['v8-selected']
        plan['comparison_controls'] = {'baseline':{'stem':'v11-base','adapter_sha256':None}}
    write_json(destination,plan)
    print({'selected':selected,'train_hash':manifest['train']['hash'],'output':plan['outputs']['sft']},flush=True)


if __name__=='__main__':
    main()
