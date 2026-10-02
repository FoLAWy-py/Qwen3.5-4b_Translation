"""Bind completed span-weight ablation to200-row development before generation."""
import json
from pathlib import Path
from scripts.v7_progress import main as verify_training
from witrans_tools.common import now, write_json

def main():
    destination = Path('runs/v7-evaluation-plan.json')
    if destination.exists():
        raise ValueError('Evaluation plan already frozen')
    verify_training()
    progress = json.loads(Path('runs/v7-progress.json').read_text(encoding='utf-8'))
    if progress['state'] != 'training_finished_semantic_review_pending':
        raise ValueError('Training must complete before generation plan')
    old = json.loads(Path('runs/v6-evaluation-plan.json').read_text(encoding='utf-8'))
    comparison = json.loads(Path('runs/v6-model-acceptance.json').read_text(encoding='utf-8'))
    if not comparison['comparison_complete']:
        raise ValueError('Controls not fully reviewed')
    candidate = {'directory':progress['plan']['output'],'adapter_sha256':progress['adapter_sha256'],
        'dev_cpo_loss':progress['metrics']['eval_cpo_loss'], 'method':'critical_span_cpo'}
    plan = {**old, 'at':now(), 'candidates':{**old['candidates'],'critical_cpo':candidate},
        'candidate_roles':['cpo','sft_control','critical_cpo'],
        'evidence':{'start':'v5-selected','cpo':'v6-cpo','sft_control':'v6-sft-control','critical_cpo':'v7-critical'},
        'outputs':{**old['outputs'],'critical_cpo':'runs/v7-critical-dev.jsonl'},
        'metadata_update_roles':['critical_cpo'], 'report_stem':'runs/v7-model-acceptance',
        'report_prefix':'runs/v7', 'report_title':'# v7关键片段加权与标准CPO/SFT开发对照',
        'selection':'Compare complete200-row semantics of critical-span CPO with v5 and both v6 controls; common16-pair objective auxiliary only.',
        'annotation_hash':progress['plan']['annotation_hash'],
        'limitations':'Known repeatedly used development. Existing control outputs have same source/prompt/NF4/greedy protocol. One seed is not algorithmic innovation evidence. No release approval.',
        'release_approved':False}
    write_json(destination,plan)
    print({'candidate':candidate,'development_hash':plan['development_hash'],'output':plan['outputs']['critical_cpo']})

if __name__ == '__main__':
    main()
