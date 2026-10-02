"""Serial, no-update probe of current-error margins and actual LoRA gradients."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import psutil
from witrans import SYSTEM_PROMPT
from scripts.review_v12_factorial import accept_manual, binding
from witrans_tools.common import append_jsonl, fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import encode_critical_spans
from witrans_tools.data import encode_example, validate_record


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', default='runs/qwen35-v3-error-span-draft-001.json')
    parser.add_argument('--owner-job', default='runs/qwen35-v3-train-adopted-job.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    dest = Path(args.output)
    assert not dest.exists(), 'Preserve all attempts'
    state = dict(at=now(), status='running', phase='CPU validation', optimizer_updates=0,
                 stage_goal_complete=False, release_approved=False, default_promoted=False)
    write_json(dest, state)
    try:
        draft = load(args.draft)
        entries = draft['entries']
        assert fingerprint(entries) == draft['entries_hash']
        refs = read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
        assert fingerprint(refs) == draft['source_train_hash']
        train = {r['id']: r for r in refs}
        generation = read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')
        outputs = {r['id']: r for r in generation}
        notes = read_jsonl('runs/qwen35-v3-train-discovery-recovery1-manual.jsonl')
        reviews = {r['id']: r for r in notes}
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-4B', local_files_only=True, trust_remote_code=False)
        encoded = {}
        for entry in entries:
            rid = entry['id']
            row, raw, note = train[rid], outputs[rid], reviews[rid]
            validate_record(row, True, True, purpose='training')
            accept_manual(raw, note)
            assert fingerprint(raw) == entry['generation_hash'] and binding(raw) == entry['binding_hash']
            assert fingerprint(note) == entry['review_hash'] and note['verdict'] in ('minor', 'major', 'critical')
            assert row['output'] == entry['chosen'] and raw['prediction'] == entry['rejected']
            assert row['review']['content_hash'] == entry['chosen_content_hash']
            chosen = encode_critical_spans(tokenizer, row, entry['critical_spans'], 1024, entry['multiplier'])
            rejected = encode_example(tokenizer, {**row, 'output': entry['rejected']}, 1024)
            encoded[rid] = (chosen, rejected)
        adapter = Path('models/witrans-qwen35-v2-critical-cpo')
        with (adapter/'adapter_model.safetensors').open('rb') as stream:
            adapter_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert adapter_sha == 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
        protocol = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            draft_hash=fingerprint(draft), ids=list(encoded), adapter_sha256=adapter_sha,
            base_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', prompt_hash=fingerprint(SYSTEM_PROMPT),
            quantization='nf4', max_length=1024, beta=0.1, seed=20261002,
            components=['weighted_span_NLL_contribution', 'remaining_NLL_contribution', 'weighted_chosen_NLL', 'preference_only'],
            gradient_method='Four independent backwards without updates. Span/rest use the SAME total chosen weight denominator, EOS in rest. Preference uses unweighted summed chosen/rejected log likelihood.',
            vector_reduction_dtype='CPU float64 for norms and dot products; gradient vectors stored CPU float32, all model parameters remain CUDA',
            row_selection='Explicit pre-existing content-bound draft entries; diagnostic subset, not full-pool efficacy estimate or trial data selection.',
            expected_forward_calls=5*len(entries), expected_backward_calls=4*len(entries),
            encoded_forward_tokens=sum(4*len(c['input_ids'])+len(r['input_ids']) for c,r in encoded.values()),
            memory_cap_gib=6.5, optimizer_updates=0,
            runtime='No compilation; train mode activates gradient checkpointing, Dropout modules individually eval and attention dropout required0; deterministic component gradients with training-equivalent FP32 non-vocabulary preparation.')
        state.update(protocol=protocol, protocol_hash=fingerprint(protocol), gpu_executed=False)
        write_json(dest, state)
        if args.preflight:
            state.update(status='CPU preflight passed', phase='finished')
            write_json(dest, state)
            print(dict(status=state['status'], pairs=len(entries), protocol_hash=state['protocol_hash']), flush=True)
            return
        # Fail closed while the existing GPU owner still runs.
        owner = load(args.owner_job)
        assert owner['status'] == 'finished'
        if owner['exit_code']!=0:
            # Recovery is allowed only for this probe's preserved memory-guard
            # failure, with zero optimizer steps and its process fully exited.
            command=owner['command']
            assert 'scripts.probe_qwen35_error_gradients' in command
            previous=load(command[command.index('--output')+1])
            assert previous['status']=='failed' and previous['optimizer_updates']==0
            assert previous['error']=='AssertionError: ' and previous['phase']=='gradient measurements'
            assert previous['protocol']['draft_hash']==fingerprint(draft)
            diagnosis=load('runs/qwen35-v3-gradient-memory-guard-diagnosis.json')
            assert diagnosis['failed_probe_hash']==fingerprint(previous) and diagnosis['memory_guard_triggered']
            assert "assert torch.cuda.max_memory_reserved()/1024**3 <= protocol['memory_cap_gib']" in Path(owner['log']).read_text(encoding='utf-8')
            state['recovery_predecessor_hash']=fingerprint(previous)
        assert len(generation) == len(notes) == 416 and set(outputs) == set(reviews) == set(train)
        summary = load('runs/qwen35-v3-train-discovery-recovery1.summary.json')
        assert summary['generation_hash'] == fingerprint(generation) and summary['adapter_sha256'] == adapter_sha
        own_family = {psutil.Process().pid, *(p.pid for p in psutil.Process().parents())}
        for process in psutil.process_iter(['pid', 'cmdline']):
            if process.pid in own_family:
                continue
            command = ' '.join(process.info['cmdline'] or [])
            if any(s in command for s in ('-m scripts.low_cpu_run', '-m scripts.evaluate_qwen35', '-m scripts.train_qwen35_recipe',
                                         '-m scripts.probe_qwen35_compiled_runtime', '-m scripts.probe_qwen35_error_gradients')):
                raise RuntimeError(f'Other model process remains: {process.pid}')
        import torch
        import torch.nn.functional as F
        from peft import PeftModel
        from transformers import set_seed
        from witrans_tools.qwen35 import Qwen35Translator
        set_seed(protocol['seed'])
        torch.cuda.reset_peak_memory_stats()
        state.update(gpu_executed=True, phase='model loading')
        write_json(dest, state)
        started = time.perf_counter()
        translator = Qwen35Translator()
        base = translator.model
        vocabulary = base.get_input_embeddings().weight
        for p in base.parameters():
            p.requires_grad = False
            if p is not vocabulary and p.__class__.__name__ != 'Params4bit' and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.to(torch.float32)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True, local_files_only=True)
        model.train()
        assert base.config.get_text_config().attention_dropout==0
        dropout_count=0
        for module in model.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd):
                module.eval()
                dropout_count+=1
        assert model.is_gradient_checkpointing
        model.config.use_cache = False
        assert all(p.device.type == 'cuda' for p in model.parameters())
        params = [(name, p) for name,p in model.named_parameters() if p.requires_grad]
        assert params and all('lora_' in name for name,p in params)
        state.update(gpu_executed=True, phase='gradient measurements', load_seconds=time.perf_counter()-started,
                     trainable_parameters=sum(p.numel() for name,p in params), trainable_tensors=len(params),
                     gradient_checkpointing_enabled=True,disabled_dropout_modules=dropout_count)
        write_json(dest, state)
        forwards = backwards = tokens = 0
        def losses(example):
            nonlocal forwards, tokens
            forwards += 1
            tokens += len(example['input_ids'])
            tensors = {k: torch.tensor([v], device='cuda:0') for k,v in example.items() if k != 'token_weights'}
            labels = tensors.pop('labels')[:,1:]
            mask = labels != -100
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**tensors).logits
            loss = F.cross_entropy(logits[:,:-1][mask].float(), labels[mask], reduction='none')
            del logits
            weights = torch.tensor(example.get('token_weights', [1.0]*len(example['labels'])), device='cuda:0')[1:][mask[0]].float()
            return loss, weights
        def snapshot(loss):
            nonlocal backwards
            assert torch.isfinite(loss)
            loss.backward()
            backwards += 1
            vector = torch.cat([(p.grad.detach().float().cpu().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for name,p in params])
            assert torch.isfinite(vector).all()
            by_module = {}
            active = 0
            for name,p in params:
                key = '.'.join(name.split('.')[-4:-2])
                sq = float(p.grad.detach().float().square().sum()) if p.grad is not None else 0.0
                by_module[key] = by_module.get(key,0.0)+sq
                active += int(sq>0)
            norm = float(torch.linalg.vector_norm(vector.double()))
            result = dict(loss=float(loss.detach()), gradient_norm=norm, active_tensors=active,
                          module_gradient_norm={k:math.sqrt(v) for k,v in by_module.items()})
            model.zero_grad(set_to_none=True)
            return result, vector
        for entry in entries:
            rid = entry['id']; chosen,rejected = encoded[rid]
            metrics = {}; vectors = {}
            began = time.perf_counter()
            for component in protocol['components']:
                values, weights = losses(chosen)
                if component == 'weighted_span_NLL_contribution':
                    selected = weights>1
                    loss = (values[selected]*weights[selected]).sum()/weights.sum()
                elif component == 'remaining_NLL_contribution':
                    selected = weights==1
                    loss = values[selected].sum()/weights.sum()
                elif component == 'weighted_chosen_NLL':
                    loss = (values*weights).sum()/weights.sum()
                    metrics['chosen_logp'] = float(-values.detach().sum())
                    metrics['chosen_mean_nll'] = float(values.detach().mean())
                else:
                    rejected_values, _ = losses(rejected)
                    metrics['rejected_logp'] = float(-rejected_values.detach().sum())
                    metrics['rejected_mean_nll'] = float(rejected_values.detach().mean())
                    metrics['summed_margin'] = metrics['chosen_logp']-metrics['rejected_logp']
                    metrics['mean_margin'] = metrics['rejected_mean_nll']-metrics['chosen_mean_nll']
                    loss = -F.logsigmoid(protocol['beta']*(-values.sum()+rejected_values.sum()))
                metrics[component], vectors[component] = snapshot(loss)
                del loss, values, weights
                if component == 'preference_only':
                    del rejected_values
                state.update(last_id=rid,last_component=component,forward_calls=forwards,
                    backward_calls=backwards,encoded_forward_tokens=tokens,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                write_json(dest,state)
                assert state['peak_reserved_gib'] <= protocol['memory_cap_gib']
            span = vectors[protocol['components'][0]]; rest = vectors[protocol['components'][1]]
            chosen_vector = vectors['weighted_chosen_NLL']; preference = vectors['preference_only']
            def cosine(a,b):
                a64,b64=a.double(),b.double()
                denom = float(torch.linalg.vector_norm(a64)*torch.linalg.vector_norm(b64))
                value=float(torch.dot(a64,b64))/denom if denom else None
                assert value is None or -1.000000000001<=value<=1.000000000001
                return value
            component_norm_sum = float(torch.linalg.vector_norm(span.double()))+float(torch.linalg.vector_norm(rest.double()))
            metrics.update(span_rest_cosine=cosine(span,rest), chosen_preference_cosine=cosine(chosen_vector,preference),
                span_fraction_of_component_norms=float(torch.linalg.vector_norm(span.double()))/component_norm_sum if component_norm_sum else None,
                span_rest_additivity_relative_error=float(torch.linalg.vector_norm(span.double()+rest.double()-chosen_vector.double()))/max(float(torch.linalg.vector_norm(chosen_vector.double())),1e-12))
            append_jsonl(dest.with_suffix('.jsonl'), dict(id=rid, at=now(), draft_entry_hash=fingerprint(entry),
                protocol_hash=state['protocol_hash'], metrics=metrics, seconds=time.perf_counter()-began,
                forward_calls=forwards, backward_calls=backwards, encoded_forward_tokens=tokens,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
            del vectors, span, rest, chosen_vector, preference
            state.update(completed_pairs=len(read_jsonl(dest.with_suffix('.jsonl'))))
            write_json(dest,state)
        assert forwards == protocol['expected_forward_calls'] and backwards == protocol['expected_backward_calls']
        assert tokens == protocol['encoded_forward_tokens']
        state.update(status='finished', phase='finished', forward_calls=forwards, backward_calls=backwards,
            encoded_forward_tokens=tokens, peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            cpu_parameter_count=sum(p.numel() for p in model.parameters() if p.device.type=='cpu'))
        write_json(dest,state)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error), at=now())
        write_json(dest,state)
        raise


if __name__ == '__main__':
    main()
