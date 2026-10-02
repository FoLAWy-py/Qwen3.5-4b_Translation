"""CPU snapshot audit of an identity-verified live repair process and its budget prefix."""
import argparse
import hashlib
import math
from pathlib import Path

import psutil
from scripts.decode_qwen35_repair_recall import load
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import encode_critical_spans, validated_annotations
from witrans_tools.data import encode_example


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--created',type=float,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();dest=Path(args.output)
    assert not dest.exists(), 'Preserve prior observations'
    process=psutil.Process(args.pid)
    assert abs(process.create_time()-args.created)<.001
    command=process.cmdline()
    assert 'scripts.train_v15_error_repair' in command and args.plan in command
    plan=load(args.plan);root=Path(plan['output']);config=load(root/'run_config.json')
    assert config['plan']==plan and config['plan_hash']==fingerprint(plan)
    assert config['expected_full_tokens']==plan['full_token_budget']
    assert config['script_sha256']==hashlib.sha256(Path('scripts/train_v15_error_repair.py').read_bytes()).hexdigest()
    # A transient partial JSON read must be retried by the observer, never
    # interpreted as a training failure or a reason to restart its process.
    progress=load(root/'progress.json');history=progress['history']
    assert history and [r['step'] for r in history]==list(range(1,len(history)+1))
    assert len(history)<=plan['updates']
    pairs=read_jsonl(plan['pairs_path']);replay=read_jsonl(plan['replay_path'])
    assert fingerprint(pairs)==plan['pairs_hash'] and fingerprint(replay)==plan['replay_hash']
    annotations=None
    if plan.get('critical_annotations'):
        report=load(plan['critical_annotations'])
        assert fingerprint(report)==plan['critical_annotations_hash']
        annotations=validated_annotations(pairs,report)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(plan['base_dir'],local_files_only=True,trust_remote_code=False)
    pair_lengths={}
    for row in pairs:
        if annotations is None:
            chosen=encode_example(tokenizer,row,plan['max_length'])
        else:
            annotation=annotations[row['id']]
            chosen=encode_critical_spans(tokenizer,row,annotation['critical_spans'],plan['max_length'],annotation['multiplier'])
        rejected=encode_example(tokenizer,{**row,'output':row['rejected']},plan['max_length'])
        pair_lengths[row['id']]=len(chosen['input_ids'])+len(rejected['input_ids'])
    replay_lengths={row['id']:len(encode_example(tokenizer,row,plan['max_length'])['input_ids']) for row in replay}
    forward=backward=tokens=0
    for record in history:
        start=(record['step']-1)*plan['accumulation']
        for visit in plan['schedule'][start:start+plan['accumulation']]:
            forward+=2+len(visit['replay']);backward+=1+len(visit['replay'])
            tokens+=pair_lengths[visit['pair']]+sum(replay_lengths[rid] for rid in visit['replay'])
        assert (record['forward_calls'],record['backward_calls'],record['full_tokens'])==(forward,backward,tokens)
        assert math.isfinite(record['loss']) and math.isfinite(record['gradient_norm_before_clip'])
        assert 0<record['peak_reserved_gib']<=plan['memory_cap_gib']
    with (Path(plan['starting_adapter'])/'adapter_model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['starting_adapter_sha256']
    write_json(dest,dict(at=now(),plan_hash=fingerprint(plan),run_config_hash=fingerprint(config),
        snapshot_hash=fingerprint(progress),completed_updates=len(history),full_tokens=tokens,
        forward_calls=forward,backward_calls=backward,reported_peak_reserved_gib=max(r['peak_reserved_gib'] for r in history),
        actual_process=dict(pid=process.pid,created_epoch=args.created,command=command),
        frozen_budget_prefix_verified=True,gpu_loaded_by_auditor=False,
        scope='Executed-prefix accounting snapshot only; not final training completion, direct live device-placement proof, semantic improvement or full memory/performance acceptance.',
        stage_goal_complete=False,release_approved=False,default_promoted=False))
    print(dict(verified_updates=len(history),full_tokens=tokens,output=str(dest)),flush=True)


if __name__=='__main__':main()
