"""Validate complete real TRAIN readings before freezing any current-error pool.

Report-only mode can audit an accumulating file. --pool-dir fails closed until
all416 generations, their model summary and individual reviews are complete.
This prepares source-bound major/critical pairs; it does not select a trial.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from witrans import SYSTEM_PROMPT, parse_translation
from scripts.review_v12_factorial import accept_manual, binding
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--pool-dir')
    args = parser.parse_args()
    report_path = Path(args.report)
    assert not report_path.exists(), 'Preserve evidence'
    train = read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    manifest = load('data/prepared/v16-reviewed-families/resolved-manifest.json')
    assert fingerprint(train) == manifest['train_ready_hash'] == 'fbeb94a8b28245a9cf5c3c91e7fdf9febc39d3aefab77214de6f627e57dca6e7'
    assert len(train) == 416 and len({r['group_id'] for r in train}) == 200
    source_notes = read_jsonl('runs/v16-family-source-audit.jsonl')
    assert fingerprint(source_notes) == manifest['decision_hash']
    sources = {n['group_id']: n for n in source_notes}
    resolutions = load('data/prepared/v16-reviewed-families/source-overlap-manual-resolution.json')
    assert fingerprint(resolutions['decisions']) == manifest['overlap_resolution_hash']
    refs = {r['id']:r for r in train}
    for row in train:
        validate_record(row, True, True, purpose='training')
        assert row['family_source_audit']['decision_hash'] == fingerprint(sources[row['group_id']])
    plan = load('data/prepared/qwen35-v2/plan.json')
    normalize = lambda text: ' '.join(text.casefold().split())
    heldout_hashes = {}
    for split,spec in plan['datasets'].items():
        heldout = read_jsonl(spec['path'])
        assert fingerprint(heldout) == spec['hash'] == manifest['heldout_hashes']['dev' if split=='known' else 'public_dev']
        assert not {r['group_id'] for r in heldout} & {r['group_id'] for r in train}
        assert not {normalize(r['input']['text']) for r in heldout} & {normalize(r['input']['text']) for r in train}
        heldout_hashes[split] = fingerprint(heldout)
    generation_path = Path('runs/qwen35-v3-train-discovery-recovery1.jsonl')
    generation = read_jsonl(generation_path)
    reviews = read_jsonl(generation_path.with_name(generation_path.stem+'-manual.jsonl'))
    generated = {r['id']:r for r in generation}
    notes = {r['id']:r for r in reviews}
    assert len(generated) == len(generation) and len(notes) == len(reviews)
    assert set(notes) <= set(generated) <= set(refs)
    pairs = []; unusable = []
    for note in reviews:
        rid = note['id']; raw = generated[rid]; ref = refs[rid]
        accept_manual(raw, note)
        assert note['binding_hash'] == binding(raw) and note['generation_hash'] == fingerprint(raw)
        assert raw['input'] == ref['input'] and raw['reference'] == ref['output']
        assert note['chosen_content_hash'] == ref['review']['content_hash']
        assert note['group_id'] == ref['group_id'] and note['category'] == ref['category']
        assert note['target_lang'] == ref['input']['target_lang']
        if note['verdict'] not in ('major', 'critical'):
            continue
        if 'prediction' not in raw or not raw.get('ended'):
            unusable.append(dict(id=rid,reason='Malformed or unfinished real error; keep raw evidence, do not invent a rejected answer.'))
            continue
        assert parse_translation(raw['raw']) == raw['prediction']
        assert raw['prediction'] != ref['output']
        pair = {**ref, 'rejected':raw['prediction'], 'preference_issue':note['note']}
        content = {k:pair[k] for k in ('input','output','rejected','preference_issue')}
        pair['preference_review'] = dict(reviewer='Codex', at=note['at'], status='approved',
            hash=fingerprint(content), verdict=note['verdict'], language_correct=note['language_correct'],
            generation_hash=fingerprint(raw), review_hash=fingerprint(note), binding_hash=binding(raw),
            scope='User-authorized individually read current Qwen3.5 TRAIN negative; AI review, not human review.')
        validate_record(pair,True,True,purpose='training')
        validate_record({**pair,'output':pair['rejected']},True,False,purpose='training')
        pairs.append(pair)
    summary_path = generation_path.with_suffix('.summary.json')
    complete = len(generation) == len(reviews) == 416 and set(notes) == set(refs) and summary_path.exists()
    summary = None
    if complete:
        summary = load(summary_path)
        with Path('models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors').open('rb') as stream:
            adapter_sha = hashlib.file_digest(stream,'sha256').hexdigest()
        assert summary['adapter_sha256'] == adapter_sha == 'fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27'
        assert summary['base']['revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        assert summary['generation_hash'] == fingerprint(generation) and summary['data_hash'] == fingerprint(train)
        assert summary['prompt_hash'] == fingerprint(SYSTEM_PROMPT) and summary['quantization'] == 'nf4'
        assert summary['decoding'] == dict(do_sample=False,max_length=1024,max_new_tokens=256)
        assert summary['rows'] == 416 and summary['cpu_parameter_count'] == 0 and summary['peak_reserved_gib'] <= 6.5
    report = dict(at=now(), generated=len(generation), reviewed=len(reviews), expected=416,
        complete_review_validated=complete, pool_written=False, major_critical_pairs=len(pairs),
        excluded_malformed_or_unfinished_errors=unusable, train_hash=fingerprint(train),
        review_hash=fingerprint(reviews), generation_snapshot_hash=fingerprint(generation),
        source_family_audit_hash=fingerprint(source_notes), overlap_resolution_hash=fingerprint(resolutions['decisions']),
        heldout_hashes=heldout_hashes, counts=dict(Counter(n['verdict'] for n in reviews)),
        diagnostic_error_strata=dict(Counter(r['category']+'/'+r['input']['target_lang'] for r in pairs)),
        selection='All individually reviewed major/critical real current errors with valid JSON and EOS. Minor/style-only outputs excluded. This is a diagnostic pool, not frozen experiment selection.',
        authorization='Current user stage Goal authorizes new negatives from these true current errors; legacy v16 positive-only manifest remains unchanged.',
        stage_goal_complete=False, release_approved=False, default_promoted=False)
    if args.pool_dir:
        assert complete, 'All416 real outputs, summary and individual reviews required before freezing pool'
        directory = Path(args.pool_dir)
        assert not directory.exists(), 'Preserve prior data'
        write_jsonl(directory/'pairs.jsonl',pairs)
        report.update(pool_written=True,pairs_hash=fingerprint(pairs),pool_path=str(directory/'pairs.jsonl'),summary_hash=fingerprint(summary))
        write_json(directory/'manifest.json',report)
    write_json(report_path,report)
    print({k:report[k] for k in ('generated','reviewed','complete_review_validated','pool_written','major_critical_pairs','diagnostic_error_strata')},flush=True)


if __name__ == '__main__':
    main()
