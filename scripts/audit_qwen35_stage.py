"""Read-only evidence verification; write a fresh, durable stage audit."""
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import psutil
from scripts.finalize_v13_optimization import complete_review
from scripts.compare_v12_blind import validate_decisions
from scripts.review_v12_factorial import binding
from witrans import SYSTEM_PROMPT
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.data import validate_record
from witrans_tools.paired_stats import paired_pass_interval


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def legacy_known(stem, spec):
    frozen = validate_decisions(read_jsonl('runs/v12-blind-packets.jsonl'),
                                read_jsonl('runs/v12-blind-decisions.jsonl'))
    rows = read_jsonl(stem+'.jsonl')
    old = {d['id']: d for d in read_jsonl(stem+'-semantic.jsonl')}
    refs = {r['id']: r for r in read_jsonl(spec['path'])}
    assert len(rows)==len(old)==len(refs)==200
    result = []
    for row in rows:
        ref = refs[row['id']]; decision = old[row['id']]
        item = frozen[decision['packet_id'],decision['label']]
        assert row['input']==ref['input'] and row['reference']==ref['output']
        assert decision['output_hash']==fingerprint({k:row[k] for k in ('input','raw','reference')})
        assert item['input_raw_hash']==fingerprint({'input':row['input'],'raw':row['raw']})
        assert item['grade']==decision['verdict'] and item['note']==decision['note']
        assert decision['reviewer']=='Codex' and type(decision['language_correct']) is bool
        assert decision['format_valid']==('prediction' in row) and decision['ended']==row.get('ended')
        assert decision['group_id']==ref['group_id'] and decision['category']==ref['category']
        result.append({**decision,'binding_hash':binding(row),'generation_hash':fingerprint(row)})
    return rows,result


def main():
    dest = Path('runs/qwen35-v3-verified-start-audit.json')
    assert not dest.exists(), 'Preserve prior audits'
    paths = ['runs/qwen35-v2-user-scope.json', 'runs/qwen35-v2-model-comparison.json',
             'runs/qwen35-v2-model-comparison.md', 'runs/qwen35-v2-experiment-audit.json',
             'runs/qwen35-v2-short-benchmark.json', 'runs/qwen35-acquisition.json',
             'data/prepared/qwen35-v2/plan.json',
             'data/prepared/v16-reviewed-families/resolved-manifest.json',
             'runs/stage-optimization-v19-checkpoint.json',
             'runs/stage-independent-confirmation-protocol.json', 'data/release_acceptance_policy.md']
    plan = load(paths[6]); acquisition = load(paths[5]); manifest = load(paths[7])
    assert acquisition == plan['base'] == load('models/Qwen3.5-4B/witrans_base.json')
    assert acquisition['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    assert fingerprint(SYSTEM_PROMPT) == plan['prompt_hash']
    assert load(paths[0])['plan_hash'] == fingerprint(plan)
    verified_weights = {}
    for item in acquisition['files']:
        path = Path(plan['base_dir']) / item['file']
        actual = sha(path)
        assert actual == item['sha256'] and path.stat().st_size == item['bytes']
        verified_weights[str(path)] = actual
    candidate = Path(plan['cpo']['output'])
    candidate_sha = sha(candidate / 'adapter_model.safetensors')
    assert candidate_sha == 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
    verified_weights[str(candidate / 'adapter_model.safetensors')] = candidate_sha
    metadata = load(candidate / 'witrans_adapter.json')
    assert metadata['adapter_sha256'] == candidate_sha and metadata['plan_hash'] == fingerprint(plan)
    training = {}
    audit = load(paths[3])
    for phase in ('sft', 'cpo'):
        spec = plan[phase]; metrics = audit['phases'][phase]['metrics']
        assert metrics['plan_hash'] == fingerprint(plan)
        assert len(metrics['history']) == spec['updates']
        assert metrics['full_tokens'] == spec['full_tokens']
        assert fingerprint(read_jsonl(spec['path'])) == spec['hash']
        training[phase] = {k: metrics[k] for k in ('full_tokens', 'forward_calls', 'backward_calls', 'seconds', 'peak_reserved_gib', 'cpu_parameter_count')}
        training[phase]['updates'] = len(metrics['history'])
    rows = read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    assert len(rows) == 416 and len({r['group_id'] for r in rows}) == 200
    assert fingerprint(rows) == manifest['train_ready_hash']
    decisions = read_jsonl('runs/v16-family-source-audit.jsonl')
    assert fingerprint(decisions) == manifest['decision_hash']
    index = {d['group_id']: d for d in decisions}
    for row in rows:
        validate_record(row, True, True, purpose='training')
        assert row['family_source_audit']['decision_hash'] == fingerprint(index[row['group_id']])
    resolutions = load('data/prepared/v16-reviewed-families/source-overlap-manual-resolution.json')
    assert fingerprint(resolutions['decisions']) == manifest['overlap_resolution_hash']
    sections = {}
    legacy = {'known': 'runs/v15-error-start-dev', 'public': 'runs/v13-v7-public'}
    v7_sha = sha('models/witrans-4b-v7-critical-cpo/adapter_model.safetensors')
    assert v7_sha == '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424'
    for split, spec in plan['datasets'].items():
        refs = read_jsonl(spec['path'])
        assert fingerprint(refs) == spec['hash'] == manifest['heldout_hashes']['dev' if split == 'known' else 'public_dev']
        assert not {r['group_id'] for r in refs} & {r['group_id'] for r in rows}
        norm = lambda text: ' '.join(text.casefold().split())
        assert not {norm(r['input']['text']) for r in refs} & {norm(r['input']['text']) for r in rows}
        generated, reviewed = complete_review(f'runs/qwen35-finetuned-{split}', spec)
        summary = load(f'runs/qwen35-finetuned-{split}.summary.json')
        assert summary['generation_hash'] == fingerprint(generated)
        assert summary['adapter_sha256'] == candidate_sha and summary['data_hash'] == spec['hash']
        _, v7 = (legacy_known if split=='known' else complete_review)(legacy[split], spec)
        assert load(legacy[split]+'.summary.json')['adapter_sha256'] == v7_sha
        assert sum(d['verdict']=='pass' for d in v7) == (140 if split=='known' else 95)
        by_id = {r['id']: r for r in generated}
        strata = {}
        for key in sorted({d['category'] + '/' + d['target_lang'] for d in reviewed}):
            current = [d for d in reviewed if d['category'] + '/' + d['target_lang'] == key]
            old = [d for d in v7 if d['category'] + '/' + d['target_lang'] == key]
            strata[key] = dict(rows=len(current), current=dict(Counter(d['verdict'] for d in current)),
                               v7=dict(Counter(d['verdict'] for d in old)),
                               pass_change_vs_v7=(sum(d['verdict']=='pass' for d in current)-sum(d['verdict']=='pass' for d in old))/len(current))
        sections[split] = dict(legacy_bound_stem=legacy[split], counts=dict(Counter(d['verdict'] for d in reviewed)),
            protocol={k: sum(d[k] for d in reviewed) for k in ('format_valid','language_correct','ended')},
            strata=strata, paired_vs_v7=paired_pass_interval(reviewed,v7),
            outstanding=[dict(review=d, generation=by_id[d['id']]) for d in reviewed
                         if d['verdict'] != 'pass' or not d['language_correct']],
            scope='Existing bound development readings, not a new independent evaluation.')
    processes = []
    for proc in psutil.process_iter(['pid','name','cmdline','create_time']):
        if proc.info['name'] and any(s in proc.info['name'].lower() for s in ('python','uv')):
            processes.append(proc.info)
    result = dict(at=now(), input_file_sha256={p:sha(p) for p in paths},
        verified_weights=verified_weights, dependencies={k:importlib.metadata.version(k) for k in
        ('torch','transformers','peft','bitsandbytes','accelerate','safetensors','psutil')},
        training=training, reviewed_train=dict(rows=416,groups=200,hash=fingerprint(rows),
        strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in rows)),
        reuse='Existing positive reviews verified; new Qwen3.5 negatives require fresh real generation and individual reading.'),
        sections=sections, processes=processes, historical_v19_job=load('runs/v19-balanced-repair-job.json'),
        discrepancies=['runs/qwen35-v3-start-audit.json is INVALID for v7 comparison: mistakenly used v15 selected candidate. Preserved; superseded by this weight-verified audit.',
        'Resolved TRAIN manifest hard counts say43 per direction, actual content says47; total416 and content hash are correct. Original manifest preserved.',
        'Historical v19 checkpoint running PID is stale; live job is finished exit0.',
        'Adapter metadata says semantic comparison pending; complete bound comparison exists.',
        'Old confirmation protocol omits frozen current Qwen3.5 third comparator required by new Goal.',
        'Old six-sentence timing uses output256 and has no Qwen3.5 body20–120 eligibility audit; not formal acceptance.',
        'Project dependency ranges describe original environment; isolated Qwen3.5 has transformers5.18/peft0.21.2. No upgrades performed.'],
        stage_goal_complete=False, default_promoted=False, release_approved=False)
    write_json(dest,result)
    print({k:result[k] for k in ('at','training','reviewed_train','discrepancies')},flush=True)
    print({s:{k:v[k] for k in ('counts','protocol','paired_vs_v7','strata')} for s,v in sections.items()},flush=True)


if __name__ == '__main__':
    main()
