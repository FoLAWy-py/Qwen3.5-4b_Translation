"""Bind individually selected diagnostic spans to real reviewed TRAIN errors.

This partial draft is neither a training pool nor a frozen experiment. The
complete 416-row review must precede selecting and freezing a repair round.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from archive.qwen3.scripts.review_v12_factorial import binding, accept_manual
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import encode_critical_spans
from witrans_tools.data import validate_record


# Root's individual readings of each approved chosen answer and actual error.
SPANS = {
    'v10-daily-19-en': ['until a small damp area had been checked by a professional'],
    'v9-constraint-006-zh-CN': ['洗好的衣服还没有取回来'],
    'public-short-0356-en': ["There's probably been a traffic accident."],
    'public-short-0224-en': ['was on television'],
    'v8-multi-001-zh-CN': ['只有我们两个人都不能在你到达之前回到家时'],
    'public-short-0355-en': ['Between you and me'],
    'public-short-0520-en': ['You talk as if you knew everything'],
    'public-short-0056-en': ['Most people'],
    'v9-constraint-004-zh-CN': ['如果旧套子合适'],
    'v10-travel-09-zh-CN': ['自己游览城市期间', '帮忙保管', '这位旅行者'],
    'public-short-0411-zh-CN': ['下周的今天'],
    'v10-travel-01-zh-CN': ['另行预订自行车位'],
    'v10-travel-16-en': ['either morning of their two-night stay'],
    'v4-mining-005-zh-CN': ['北行有轨电车'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/qwen35-v3-error-span-draft-001.json')
    parser.add_argument('--extra-spans', action='append', default=[])
    args = parser.parse_args()
    destination = Path(args.output)
    assert not destination.exists(), 'Preserve prior draft evidence'
    train = read_jsonl('data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl')
    assert fingerprint(train) == 'fbeb94a8b28245a9cf5c3c91e7fdf9febc39d3aefab77214de6f627e57dca6e7'
    refs = {r['id']: r for r in train}
    generated = read_jsonl('runs/qwen35-v3-train-discovery-recovery1.jsonl')
    outputs = {r['id']: r for r in generated}
    notes = read_jsonl('runs/qwen35-v3-train-discovery-recovery1-manual.jsonl')
    reviewed = {r['id']: r for r in notes}
    assert len(outputs) == len(generated) and len(reviewed) == len(notes)
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-4B', local_files_only=True, trust_remote_code=False)
    selected = dict(SPANS)
    for extra_path in args.extra_spans:
        for line in Path(extra_path).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            rid, literal_spans = line.split('\t',1)
            assert rid not in selected, 'Do not duplicate or replace explicit earlier spans'
            selected[rid] = json.loads(literal_spans)
    entries = []
    for rid, spans in selected.items():
        row, raw, note = refs[rid], outputs[rid], reviewed[rid]
        validate_record(row, True, True, purpose='training')
        accept_manual(raw, note)
        assert note['verdict'] in ('major','critical')
        assert note['binding_hash'] == binding(raw) and note['generation_hash'] == fingerprint(raw)
        assert note['chosen_content_hash'] == row['review']['content_hash']
        assert note['group_id'] == row['group_id'] and raw['reference'] == row['output']
        assert raw['input'] == row['input'] and raw['ended'] and 'prediction' in raw
        assert raw['prediction'] != row['output']
        encoded = encode_critical_spans(tokenizer, row, spans, 1024, 3.0)
        weights = [w for w, label in zip(encoded['token_weights'], encoded['labels']) if label != -100]
        entries.append(dict(id=rid, group_id=row['group_id'], category=row['category'],
            target_lang=row['input']['target_lang'], chosen=row['output'], rejected=raw['prediction'],
            generation_hash=fingerprint(raw), review_hash=fingerprint(note), binding_hash=binding(raw),
            chosen_content_hash=row['review']['content_hash'], critical_spans=spans, multiplier=3.0,
            reviewer='Codex', at=now(), note=note['note'],
            supervised_tokens=len(weights), span_tokens=sum(w > 1 for w in weights),
            weighted_span_share=sum(w for w in weights if w > 1) / sum(weights),
            eos_weight=weights[-1], actual_parameter_gradient='not yet measured'))
    write_json(destination, dict(at=now(), entries=entries, entries_hash=fingerprint(entries),
        source_train_hash=fingerprint(train), reviewed_snapshot_count=len(notes),
        source_review_snapshot_hash=fingerprint(notes),
        scope='Partial individual diagnostic annotation draft, not finalized training data; no optimizer or GPU work executed.',
        required_next='Complete all416 real outputs and readings, then freeze error pool and serial margin/gradient probe before one training experiment.',
        stage_goal_complete=False, release_approved=False, default_promoted=False))
    print(dict(path=str(destination), annotated=len(entries),
        span_tokens=sum(e['span_tokens'] for e in entries)), flush=True)


if __name__ == '__main__':
    main()
