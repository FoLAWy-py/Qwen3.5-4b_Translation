"""Continue NF4 SFT with frozen mixed data; choose checkpoints only by new dev."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from witrans import WiTrans, SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import encode_example, validate_record
from witrans_tools.preference import answer_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/prepared/v4')
    parser.add_argument('--output', default='models/witrans-4b-v4-sft')
    parser.add_argument('--selection-output', default='runs/v4-selection.json')
    args = parser.parse_args()
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; preserve immutable run")
    if Path(args.selection_output).exists():
        raise ValueError('Selection record exists; preserve immutable run')
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding="utf-8"))
    train, dev = [read_jsonl(data_dir / f'{name}.jsonl') for name in ('train', 'dev')]
    for name, rows in (('train', train), ('dev', dev)):
        if fingerprint(rows) != manifest[name]['hash']:
            raise ValueError("Frozen corpus changed")
        for r in rows:
            validate_record(r, True, True, purpose='training' if name == 'train' else 'evaluation')
    if {r['group_id'] for r in train} & {r['group_id'] for r in dev} or {r['input']['text'].strip().casefold() for r in train} & {r['input']['text'].strip().casefold() for r in dev}:
        raise ValueError("Training/development leakage")
    cfg = manifest['configuration']
    checkpoints = cfg.get('checkpoint_steps', [16, cfg['steps']])
    if cfg['steps'] not in checkpoints or any(not 1 <= s <= cfg['steps'] for s in checkpoints):
        raise ValueError('Invalid checkpoint plan')
    set_seed(cfg['seed'])
    start_dir = Path(cfg.get('starting_adapter', 'models/witrans-4b-v2-selected/adapter'))
    translator = WiTrans("models/Qwen3-4B", start_dir, max_length=1024, quantization='nf4')
    if cfg.get('starting_adapter_sha256') is not None and translator.adapter_sha256 != cfg['starting_adapter_sha256']:
        raise ValueError('Frozen starting adapter changed')
    base = translator.model.unload()
    if any('lora_' in name for name, _ in base.named_parameters()):
        raise ValueError("Old LoRA still present")
    if hasattr(base, 'peft_config'):
        delattr(base, 'peft_config')
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = PeftModel.from_pretrained(base, start_dir, is_trainable=True, local_files_only=True)
    model.config.use_cache = False
    encoded = [encode_example(translator.tokenizer, r, 1024) for r in train]
    dev_encoded = [encode_example(translator.tokenizer, r, 1024) for r in dev]
    config = {**cfg, 'at': now(), 'parent_adapter_sha256': translator.adapter_sha256,
        'train_hash': fingerprint(train), 'dev_hash': fingerprint(dev), 'quantization': 'nf4', 'max_length': 1024,
        'loss': 'Mean answer/EOS NLL per record, ignoring prompt; selection uses mean over dev records'}
    write_json(output / 'run_config.json', config)
    write_jsonl(output / 'train_snapshot.jsonl', train)
    write_jsonl(output / 'dev_snapshot.jsonl', dev)
    def nll(example):
        tensors = {k: torch.tensor([v], device='cuda:0') for k, v in example.items()}
        labels = tensors.pop('labels')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return answer_scores(model(**tensors).logits, labels)[1]
    def evaluate():
        model.eval()
        with torch.inference_mode():
            losses = [float(nll(row)) for row in dev_encoded]
        model.train()
        return sum(losses) / len(losses)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters, lr=cfg['lr'])
    order = list(range(len(encoded)))
    random.Random(cfg['seed']).shuffle(order)
    history, candidates = [], []
    torch.cuda.reset_peak_memory_stats()
    initial_nll = evaluate()
    print({'starting_dev_nll': initial_nll}, flush=True)
    start = time.perf_counter()
    model.train()
    for step in range(1, cfg['steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0
        for micro in range(cfg['accumulation']):
            row = encoded[order[((step - 1) * cfg['accumulation'] + micro) % len(order)]]
            loss = nll(row)
            (loss / cfg['accumulation']).backward()
            loss_total += float(loss.detach()) / cfg['accumulation']
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        history.append({'step': step, 'loss': loss_total})
        print(history[-1], flush=True)
        if step in checkpoints:
            score = evaluate()
            destination = output / f'checkpoint-{step}'
            model.save_pretrained(destination, safe_serialization=True)
            translator.tokenizer.save_pretrained(destination)
            digest = hashlib.sha256((destination / 'adapter_model.safetensors').read_bytes()).hexdigest()
            metadata = json.loads((start_dir / 'witrans_adapter.json').read_text(encoding='utf-8'))
            for key in ('selected_by', 'selected_checkpoint', 'selected_dev_loss', 'acceptance_report'):
                metadata.pop(key, None)
            metadata.update(smoke_only=False, adapter_sha256=digest, prompt_hash=fingerprint(SYSTEM_PROMPT),
                created_at=now(), recommended_quantization='nf4', quality_status='Experimental; development semantic acceptance pending',
                parent_adapter_sha256=translator.adapter_sha256, training_method='continued_sft', selected_dev_loss=score)
            write_json(destination / 'witrans_adapter.json', metadata)
            candidates.append({'step': step, 'directory': str(destination), 'dev_nll': score, 'adapter_sha256': digest})
            print(candidates[-1], flush=True)
    torch.cuda.synchronize()
    selected = min(candidates, key=lambda r: r['dev_nll'])
    write_json(args.selection_output, {'at': now(), 'selected': selected, 'candidates': candidates,
        'starting_dev_nll': initial_nll, 'dev_hash': fingerprint(dev), 'policy': 'Frozen checkpoint choice before newdev generation; no release test used'})
    write_json(output / 'metrics.json', {'at': now(), 'history': history, 'train_seconds_including_evaluation_and_save': time.perf_counter() - start,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'selected': selected, 'starting_dev_nll': initial_nll})
    print({'selected': selected}, flush=True)

if __name__ == '__main__':
    main()
