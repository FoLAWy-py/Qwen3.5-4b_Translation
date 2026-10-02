"""TRAIN-only teacher-forced CPO score derivatives; no parameter-gradient claim."""
import json
import math
import time
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    progress_path = Path('runs/v15-repair-signal-progress.json')
    if progress_path.exists():
        raise ValueError('Preserve diagnostic attempt')
    owner = load('runs/v15-repair-recall-job.json')
    state = dict(at=now(), phase='waiting_for_owned_recall', release_approved=False)
    write_json(progress_path, state)
    try:
        while owner['status'] in ('launching', 'running'):
            time.sleep(5)
            owner = load('runs/v15-repair-recall-job.json')
        assert owner['status'] == 'finished' and owner['exit_code'] == 0
        assert load('runs/v15-repair-recall-progress.json')['phase'] == 'individual_train_recall_review_pending'
        plan = load('data/prepared/v15-error-repair/plan.json')
        pairs = read_jsonl(plan['pairs_path'])
        assert len(pairs) == 11 and fingerprint(pairs) == plan['pairs_hash']
        import torch
        import gc
        from witrans import WiTrans
        from witrans_tools.data import encode_example
        from witrans_tools.preference import answer_scores
        results = []
        for role, adapter in [('v7', plan['starting_adapter']), ('v15', plan['output'])]:
            state.update(phase='teacher_forced_train_diagnostics', role=role, updated_at=now())
            write_json(progress_path, state)
            translator = WiTrans('models/Qwen3-4B', adapter, max_length=1024, quantization='nf4')
            expected = plan['starting_adapter_sha256'] if role == 'v7' else load('runs/v15-error-fixed-selection.json')['selected']['adapter_sha256']
            assert translator.adapter_sha256 == expected
            translator.model.eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for pair in pairs:
                    scores, counts, nlls = [], [], []
                    for answer in (pair['output'], pair['rejected']):
                        encoded = encode_example(translator.tokenizer, {**pair, 'output': answer}, 1024)
                        tensors = {k: torch.tensor([v], device='cuda:0') for k, v in encoded.items()}
                        labels = tensors.pop('labels')
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            logp, nll = answer_scores(translator.model(**tensors).logits, labels)
                        scores.append(float(logp)); nlls.append(float(nll))
                        counts.append(sum(v != -100 for v in encoded['labels']))
                    margin = scores[0] - scores[1]
                    probability = float(torch.sigmoid(torch.tensor(-plan['beta'] * margin, dtype=torch.float64)))
                    coefficient = plan['beta'] * probability
                    # Norms over the vector of individual token log probabilities.
                    ratio = coefficient * math.sqrt(counts[0] * sum(counts))
                    record = dict(id=pair['id'], group_id=pair['group_id'], role=role,
                                  pair_hash=fingerprint(pair), adapter_sha256=expected,
                                  chosen_logp=scores[0], rejected_logp=scores[1], summed_margin=margin,
                                  chosen_answer_tokens=counts[0], rejected_answer_tokens=counts[1],
                                  chosen_nll=nlls[0], rejected_nll=nlls[1],
                                  mean_logp_margin=nlls[1]-nlls[0],
                                  preference_token_logp_coefficient=coefficient,
                                  chosen_sft_token_logp_coefficient=1/counts[0],
                                  preference_to_chosen_sft_token_score_gradient_norm_ratio=ratio,
                                  note='Analytic token-score derivative with dropout disabled; not measured parameter gradients or semantic quality.')
                    results.append(record)
                    print(record, flush=True)
            assert torch.cuda.max_memory_reserved()/1024**3 <= 6.5
            del tensors, labels, translator
            gc.collect(); torch.cuda.empty_cache()
        write_jsonl('runs/v15-repair-signal.jsonl', results)
        write_json('runs/v15-repair-signal.summary.json', dict(at=now(), plan_hash=fingerprint(plan),
            pairs=11, roles=['v7', 'v15'], rows=len(results), data_hash=fingerprint(pairs),
            result_hash=fingerprint(results), scope='TRAIN teacher-forced likelihood and analytic token-score gradient strengths only; raw and length-normalized margins both reported. Decoded recall and DEV review decide next experiment.', release_approved=False))
        state.update(phase='diagnostic_complete', updated_at=now())
    except Exception as exc:
        state.update(phase='failed_preserving_evidence', error=str(exc), updated_at=now())
        raise
    finally:
        write_json(progress_path, state)


if __name__ == '__main__':
    main()
