"""Build a fresh conservative view from all individual source readings, preserving history."""
import copy
from collections import Counter
from pathlib import Path

from scripts.decode_qwen35_repair_recall import load
from scripts.report_qwen35_error_gradients import summarize
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    report_path = Path('runs/qwen35-v3-focused32-negative-final-audit.json')
    assert not report_path.exists()
    draft_path = 'data/prepared/qwen35-v3-reviewed32-current-errors/pairs.jsonl'
    draft = read_jsonl(draft_path)
    actual = read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')
    raw = {r['id']: r for r in actual}
    notes_path = 'runs/qwen35-v3-focused32-negative-quality-rereview.jsonl'
    notes = read_jsonl(notes_path)
    by_id = {r['id']: r for r in notes}
    assert len(draft) == len(notes) == len(by_id) == 32
    assert set(by_id) == {r['id'] for r in draft}
    for pair in draft:
        note = by_id[pair['id']]
        assert note['pair_hash'] == fingerprint(pair)
        assert note['actual_generation_hash'] == fingerprint(raw[pair['id']])
        assert pair['input'] == raw[pair['id']]['input']
        assert pair['rejected'] == raw[pair['id']]['prediction']
        assert note['valid_major_negative'] == (note['verdict'] == 'major')
    correction_path = 'runs/qwen35-v3-focused32-negative-corrections-001.tsv'
    corrections = []
    effective = copy.deepcopy(by_id)
    for line in Path(correction_path).read_text(encoding='utf-8-sig').splitlines():
        rid, verdict, note = line.split('\t', 2)
        assert rid in effective and effective[rid]['verdict'] == 'major'
        assert verdict == 'minor' and note.strip()
        prior = copy.deepcopy(effective[rid])
        effective[rid].update(verdict=verdict, valid_major_negative=False, note=note,
            correction=dict(prior_reading_hash=fingerprint(prior), prior_note=prior['note'],
                correction_path=correction_path, original_verdict=prior['verdict']))
        corrections.append(dict(id=rid, prior_reading_hash=fingerprint(prior),
            new_reading_hash=fingerprint(effective[rid]), verdict=verdict, note=note))
    assert len(corrections) == 3
    effective_rows = [effective[r['id']] for r in draft]
    old_view_path = 'runs/qwen35-v3-train-discovery-sweetener-corrected-manual.jsonl'
    old_view = read_jsonl(old_view_path)
    view = copy.deepcopy(old_view)
    for row in view:
        if row['id'] not in effective:
            continue
        reading = effective[row['id']]
        assert row['generation_hash'] == fingerprint(raw[row['id']])
        prior = copy.deepcopy(row)
        row.update(verdict=reading['verdict'], note=reading['note'], at=now(),
            rereview=dict(previous_path=old_view_path, previous_review_hash=fingerprint(prior),
                focused_reading_hash=fingerprint(reading), final_audit=str(report_path),
                reviewer='Codex AI; not independent human review'))
    # Preserve original approved positive metadata. Attach latest evidence separately.
    pairs = []
    for pair in draft:
        reading = effective[pair['id']]
        if not reading['valid_major_negative']:
            continue
        copied = copy.deepcopy(pair)
        copied['focused_negative_rereview'] = dict(reading_hash=fingerprint(reading),
            report=str(report_path), verdict='major', note=reading['note'],
            actual_generation_hash=reading['actual_generation_hash'])
        pairs.append(copied)
    assert {r['id'] for r in pairs} == {r['id'] for r in view if r['verdict'] == 'major'}
    assert len(pairs) == 23
    counts = dict(Counter(r['verdict'] for r in view))
    assert counts == {'pass': 328, 'minor': 65, 'major': 23}
    measured = read_jsonl('runs/qwen35-v3-gradient-fp64.jsonl')
    selected = [r for r in measured if r['id'] in {p['id'] for p in pairs}]
    assert len(selected) == len(pairs)
    old_gradient = load('runs/qwen35-v3-reviewed32-gradient-diagnosis.json')
    gradient = dict(at=now(), original_probe_report_hash=old_gradient['original_probe_report_hash'],
        original_executed_accounting=old_gradient['original_executed_accounting'],
        selected_real_error_rows=len(selected), selected_rows_hash=fingerprint(selected),
        clean_pairs_hash=fingerprint(pairs), overall=summarize(selected),
        scope='CPU subset of original actual FP64 gradient probe. No new GPU calls; original205 forwards164 backwards remain counted.')
    outputs = dict(view='runs/qwen35-v3-train-discovery-focused-corrected-manual.jsonl',
        effective_readings='runs/qwen35-v3-focused32-negative-effective-readings.jsonl',
        pairs='data/prepared/qwen35-v3-reviewed23-current-errors/pairs.jsonl',
        gradient='runs/qwen35-v3-reviewed23-gradient-diagnosis.json')
    for path in outputs.values():
        assert not Path(path).exists()
    report = dict(at=now(), reviewer='Codex AI; not independent human review',
        draft_path=draft_path, draft_hash=fingerprint(draft), actual_hash=fingerprint(actual),
        individual_readings_path=notes_path, individual_readings_hash=fingerprint(notes),
        corrections=corrections, previous_view_path=old_view_path, previous_view_hash=fingerprint(old_view),
        outputs=outputs, hashes=dict(view=fingerprint(view), effective_readings=fingerprint(effective_rows),
            pairs=fingerprint(pairs), gradient=fingerprint(gradient)),
        reviewed=32, retained_major_negatives=len(pairs), groups=len({r['group_id'] for r in pairs}),
        excluded_ids=[r['id'] for r in draft if not effective[r['id']]['valid_major_negative']],
        counts=counts, effective_negative_counts=dict(Counter(r['verdict'] for r in effective_rows)),
        original416_positive_reference_unchanged=True,
        dictionary_support=[
            'https://dictionary.cambridge.org/us/dictionary/english-chinese-simplified/speed',
            'https://dictionary.cambridge.org/dictionary/english-chinese-simplified/tart',
            'https://dictionary.cambridge.org/dictionary/english/frog-in-throat'],
        historical_trials_remain_quarantined=True, new_trial_frozen=False,
        scope='Reconciled TRAIN diagnostic view only. No historical outputs/reviews changed; no new independently confirmed quality or valid optimization round claimed.',
        stage_goal_complete=False, release_approved=False, default_promoted=False)
    write_jsonl(outputs['view'], view)
    write_jsonl(outputs['effective_readings'], effective_rows)
    write_jsonl(outputs['pairs'], pairs)
    write_json(outputs['gradient'], gradient)
    write_json(report_path, report)
    assert fingerprint(read_jsonl(draft_path)) == report['draft_hash']
    assert fingerprint(read_jsonl(notes_path)) == report['individual_readings_hash']
    assert fingerprint(read_jsonl(old_view_path)) == report['previous_view_hash']
    print(dict(report=str(report_path), retained=len(pairs), groups=report['groups'], counts=counts))


if __name__ == '__main__':
    main()
