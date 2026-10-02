"""Report verified direct optimization progress without inventing acceptance."""
import json
from pathlib import Path
from witrans_tools.common import now, write_json


def main():
    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    training = load('models/witrans-4b-v13-public-sft/metrics.json')
    selection = load('runs/v13-public-sft-selection.json')
    pipeline = load('runs/v13-optimization-progress.json')
    reviews = {}
    for name in ('v7-public','candidate-known','candidate-public'):
        path = Path(f'runs/v13-{name}-semantic.summary.json')
        if path.exists():
            reviews[name] = load(path)
    acceptance = Path('runs/v13-model-acceptance.json')
    report = {
        'at':now(),'goal_status':'active','base_comparison_enabled':False,
        'scope':'Optimize current best adapter; no new original-base comparison.',
        'training_complete':True,'actual_training_input_tokens':training['actual_training_input_tokens'],
        'selected_checkpoint':selection['selected'],
        'balanced_nll_relative_decrease':1-selection['selected']['nll']['balanced_mean']/selection['initial_nll']['balanced_mean'],
        'peak_training_reserved_gib':training['peak_reserved_gib'],
        'train_seconds_including_evaluation_and_save':training['train_seconds_including_evaluation_and_save'],
        'generation':pipeline,'semantic_reviews':reviews,
        'semantic_acceptance_complete':acceptance.exists(),'default_promoted':False,'release_approved':False,
        'next_fact_pair_manifest':'data/prepared/v14-public-fact-pairs/manifest.json',
        'next_fact_pair_training_started':False,
        'interpretation':'NLL decrease is not semantic improvement. Partial generated subsets are not full-development acceptance.',
    }
    if acceptance.exists():
        report['semantic_acceptance'] = load(acceptance)
    next_job = Path('runs/v14-matching-job.json')
    if next_job.exists():
        report['next_fact_trial'] = {'plan':'data/prepared/v14-fact-trial/plan.json',
                                    'job':load(next_job),'mode':'matching',
                                    'other_arms_started':False,'release_approved':False}
        report['next_fact_pair_training_started'] = Path('models/witrans-4b-v14-fact-matching/run_config.json').exists()
        next_progress = Path('models/witrans-4b-v14-fact-matching/progress.json')
        if next_progress.exists():
            report['next_fact_trial']['last_verified_update'] = load(next_progress)['history'][-1]
        next_evaluation = Path('runs/v14-matching-evaluation-progress.json')
        if next_evaluation.exists():
            report['next_fact_trial']['evaluation'] = load(next_evaluation)
        next_metrics = Path('models/witrans-4b-v14-fact-matching/metrics.json')
        if next_metrics.exists():
            metrics = load(next_metrics)
            report['next_fact_trial'].update(training_complete=True,selected=metrics['selected'],
                                             counters={k:metrics[k] for k in ('full_tokens','forward_calls','backward_calls')},
                                             peak_reserved_gib=metrics['peak_reserved_gib'])
        for split in ('known','public'):
            review_path = Path(f'runs/v14-matching-{split}-semantic.summary.json')
            if review_path.exists():
                report['next_fact_trial'][f'{split}_semantic_review'] = load(review_path)
        next_acceptance = Path('runs/v14-matching-model-acceptance.json')
        if next_acceptance.exists():
            report['next_fact_trial']['semantic_acceptance'] = load(next_acceptance)
        diagnostics = Path('runs/v14-parent-fact-diagnostics.summary.json')
        if diagnostics.exists():
            report['next_fact_trial']['parent_training_gradient_diagnostics'] = load(diagnostics)
    corrected_pairs = Path('data/prepared/v15-public-fact-pairs/manifest.json')
    if corrected_pairs.exists():
        report['corrected_fact_data'] = load(corrected_pairs)
    mining_job = Path('runs/v15-training-hardness-job.json')
    if mining_job.exists():
        report['next_training_source_audit'] = {'job':load(mining_job),'decode_job':'runs/v15-training-triage-decoding-job.json',
                                               'new_contrasts_training_started':False,'release_approved':False}
        mining_progress = Path('runs/v15-training-hardness/progress.json')
        if mining_progress.exists():
            report['next_training_source_audit']['progress'] = load(mining_progress)
        decoded_progress = Path('runs/v15-training-triage-decoding-progress.json')
        if decoded_progress.exists():
            report['next_training_source_audit']['decoding_progress'] = load(decoded_progress)
        manual_readings = Path('runs/v15-training-triage-manual.jsonl')
        if manual_readings.exists():
            from witrans_tools.common import read_jsonl
            report['next_training_source_audit']['individually_reviewed_outputs'] = len(read_jsonl(manual_readings))
    repair_job = Path('runs/v15-error-optimization-job.json')
    if repair_job.exists():
        report['actual_error_optimization'] = {'job':load(repair_job),
            'plan':'data/prepared/v15-error-repair/plan.json','release_approved':False}
        for key,path in (('pipeline','runs/v15-error-optimization-progress.json'),
                         ('training','models/witrans-4b-v15-error-cpo/progress.json'),
                         ('metrics','models/witrans-4b-v15-error-cpo/metrics.json')):
            if Path(path).exists():
                report['actual_error_optimization'][key] = load(path)
        report['next_training_source_audit']['new_contrasts_training_started'] = Path('models/witrans-4b-v15-error-cpo/run_config.json').exists()
    write_json('runs/v13-quality-progress.json',report)
    prior = load('runs/v12-goal-progress.json')
    prior.update(at=report['at'], phase='v13_semantic_acceptance_running',
                 v13_last_verified_update=64, v13_last_verified_tokens=training['actual_training_input_tokens'],
                 v13_selected_balanced_nll=selection['selected']['nll']['balanced_mean'],
                 active_quality_progress='runs/v13-quality-progress.json')
    if acceptance.exists():
        prior['phase'] = 'v13_semantic_acceptance_complete_rejected'
    if next_job.exists():
        prior.update(phase='v14_fact_optimization_'+report['next_fact_trial']['job']['status'],
                     active_fact_trial_plan=report['next_fact_trial']['plan'],
                     active_fact_trial_job=str(next_job),
                     next_fact_pair_training_started=report['next_fact_pair_training_started'])
    if mining_job.exists():
        prior.update(phase='v15_training_source_audit_'+report['next_training_source_audit']['job']['status'],
                     active_training_source_audit_job=str(mining_job),
                     active_training_source_audit_decoding_job=report['next_training_source_audit']['decode_job'],
                     corrected_fact_manifest=str(corrected_pairs))
    if repair_job.exists():
        prior.update(phase='v15_actual_error_optimization_'+report['actual_error_optimization']['job']['status'],
                     active_optimization_plan='data/prepared/v15-error-repair/plan.json',
                     active_optimization_job=str(repair_job),base_comparison_enabled=False)
    write_json('runs/v12-goal-progress.json',prior)
    print({k:report[k] for k in ('training_complete','balanced_nll_relative_decrease','semantic_acceptance_complete')})


if __name__ == '__main__':
    main()
