"""One frozen factorial arm; exact whole-record token accumulation, NF4 microbatch1."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from archive.qwen3.runtime import SYSTEM_PROMPT, _LocalTranslator
from witrans_tools.common import base_manifest, fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record
from witrans_tools.preference import answer_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True)
    parser.add_argument('--plan', default='data/prepared/v12-factorial-v2/plan.json')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    arm = next(a for a in plan['arms'] if a['name'] == args.arm)
    scope_path = Path('runs/optimization-scope-20261001.json')
    if scope_path.exists() and arm['start'] == 'base':
        scope = json.loads(scope_path.read_text(encoding='utf-8'))
        if scope.get('base_comparison_enabled') is False:
            raise ValueError('User revoked further base branches; optimize current adapter')
    output, selection_path = Path(arm['output']), Path(arm['selection_output'])
    if output.exists() or selection_path.exists():
        raise ValueError('Preserve existing arm evidence; explicit new run required')
    if base_manifest('models/Qwen3-4B') != plan['base'] or fingerprint(SYSTEM_PROMPT) != plan['prompt_hash']:
        raise ValueError('Frozen base or prompt changed')
    dataset = plan['datasets'][arm['dataset']]
    train = read_jsonl(dataset['path'])
    schedule = json.loads(Path(dataset['schedule_path']).read_text(encoding='utf-8'))
    devsets = {name: read_jsonl(plan[name]['path']) for name in ('dev', 'public_dev')}
    if fingerprint(train) != dataset['hash'] or fingerprint(schedule) != dataset['schedule_hash']:
        raise ValueError('Frozen training schedule or data changed')
    for name, records in devsets.items():
        if fingerprint(records) != plan[name]['hash']:
            raise ValueError('Frozen development set changed')
    for purpose, records in [('training', train)] + [('evaluation', rows) for rows in devsets.values()]:
        for row in records:
            validate_record(row, True, True, purpose=purpose)
    start = Path(plan['starting_adapter'])
    parent_hash = hashlib.sha256((start/'adapter_model.safetensors').read_bytes()).hexdigest()
    if parent_hash != plan['starting_adapter_sha256']:
        raise ValueError('Frozen starting adapter changed')
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    set_seed(plan['seed'])
    translator = _LocalTranslator('models/Qwen3-4B', None, max_length=1024, quantization='nf4')
    encoded = {row['id']: encode_example(translator.tokenizer, row, 1024) for row in train}
    dev_encoded = {name: [encode_example(translator.tokenizer, row, 1024) for row in records] for name, records in devsets.items()}
    label_counts = {key: sum(v != -100 for v in row['labels']) for key, row in encoded.items()}
    if (len(schedule) != plan['updates']
            or sum(len(encoded[key]['input_ids']) for batch in schedule for key in batch) != plan['training_input_tokens_per_arm']
            or [sum(len(encoded[key]['input_ids']) for key in batch) for batch in schedule] != dataset['update_full_tokens']
            or sum(label_counts[key] for batch in schedule for key in batch) != dataset['supervised_tokens']):
        raise ValueError('Encoded schedule differs from fixed token budget')
    model = prepare_model_for_kbit_training(translator.model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={'use_reentrant': False})
    if arm['start'] == 'base':
        model = get_peft_model(model, LoraConfig(r=plan['lora_rank'], lora_alpha=plan['lora_alpha'],
                    lora_dropout=plan['lora_dropout'], bias='none', task_type='CAUSAL_LM',
                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        parent_hash = None
    else:
        model = PeftModel.from_pretrained(model, start, is_trainable=True, local_files_only=True)
        cfg = model.peft_config['default']
        if (cfg.r, cfg.lora_alpha, cfg.lora_dropout) != (plan['lora_rank'], plan['lora_alpha'], plan['lora_dropout']):
            raise ValueError('Continued LoRA structure differs')
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=plan['learning_rate'], weight_decay=.01)
    output.mkdir(parents=True)
    config = {'at': now(), 'plan_hash': fingerprint(plan), 'arm': arm,
              'parent_adapter_sha256': parent_hash, 'seed': plan['seed'],
              'training_input_tokens': plan['training_input_tokens_per_arm'],
              'supervised_tokens': dataset['supervised_tokens'], 'loss': plan['loss'],
              'optimizer': 'AdamW8bit', 'weight_decay': .01, 'clip_grad_norm': 1.0,
              'learning_rate': plan['learning_rate'], 'train_hash': fingerprint(train),
              'dev_hashes': {name: fingerprint(rows) for name, rows in devsets.items()}}
    write_json(output/'run_config.json', config)
    write_jsonl(output/'train_snapshot.jsonl', train)
    def nll(row):
        tensors = {key: torch.tensor([value], device='cuda:0') for key, value in row.items()}
        labels = tensors.pop('labels')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits, labels)[1]
    def evaluate():
        model.eval()
        with torch.inference_mode():
            scores = {name: sum(float(nll(row)) for row in rows)/len(rows) for name, rows in dev_encoded.items()}
        model.train()
        return {**scores, 'balanced_mean': sum(scores.values())/len(scores)}
    torch.cuda.reset_peak_memory_stats()
    beginning = time.perf_counter()
    initial = evaluate()
    print({'arm': args.arm, 'initial_nll': initial}, flush=True)
    history, candidates, processed = [], [], 0
    for step, batch in enumerate(schedule, 1):
        optimizer.zero_grad(set_to_none=True)
        supervised = sum(label_counts[key] for key in batch)
        total_loss = 0.
        for key in batch:
            loss = nll(encoded[key]) * (label_counts[key]/supervised)
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        processed += sum(len(encoded[key]['input_ids']) for key in batch)
        item = {'step': step, 'loss': total_loss, 'full_tokens': processed,
                'record_visits': len(batch), 'supervised_tokens': supervised}
        history.append(item)
        print(item, flush=True)
        if step in plan['checkpoints']:
            scores = evaluate()
            checkpoint = output/f'checkpoint-{step}'
            model.save_pretrained(checkpoint, safe_serialization=True)
            translator.tokenizer.save_pretrained(checkpoint)
            digest = hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()
            metadata = {'base_model_id': 'Qwen/Qwen3-4B', 'base_revision': plan['base']['revision'],
                        'prompt_hash': plan['prompt_hash'], 'smoke_only': False, 'created_at': now(),
                        'adapter_sha256': digest, 'parent_adapter_sha256': parent_hash,
                        'recommended_quantization': 'nf4', 'training_method': 'token_budget_factorial_sft',
                        'quality_status': 'Experimental; complete semantic acceptance and release pending'}
            write_json(checkpoint/'witrans_adapter.json', metadata)
            candidate = {'step': step, 'directory': str(checkpoint), 'adapter_sha256': digest, 'nll': scores}
            candidates.append(candidate)
            print({'checkpoint': candidate}, flush=True)
    torch.cuda.synchronize()
    if processed != plan['training_input_tokens_per_arm']:
        raise ValueError('Actual token count mismatch')
    selected = min(candidates, key=lambda row: row['nll']['balanced_mean'])
    write_json(selection_path, {'at': now(), 'arm': args.arm, 'selected': selected,
                               'candidates': candidates, 'initial_nll': initial,
                               'plan_hash': fingerprint(plan), 'release_approved': False})
    write_json(output/'metrics.json', {'at': now(), 'history': history, 'selected': selected,
             'initial_nll': initial, 'actual_training_input_tokens': processed,
             'train_seconds_including_evaluation_and_save': time.perf_counter()-beginning,
             'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
             'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
    print({'arm': args.arm, 'selected': selected}, flush=True)


if __name__ == '__main__':
    main()
