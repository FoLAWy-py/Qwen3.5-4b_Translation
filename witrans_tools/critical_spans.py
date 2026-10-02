"""Explicit reviewed positive-answer spans; no automatic error-to-span inference."""
import json
import math
from .data import encode_example
from .common import fingerprint

def validated_annotations(rows, report):
    if report['train_hash'] != fingerprint(rows) or report['annotation_hash'] != fingerprint(report['annotations']):
        raise ValueError('Frozen training or annotations changed')
    train = {row['id']:row for row in rows}
    annotations = {entry['id']:entry for entry in report['annotations']}
    if not annotations or len(train) != len(rows) or len(annotations) != len(report['annotations']) or set(annotations) - set(train):
        raise ValueError('Duplicate or non-training annotation')
    for key,entry in annotations.items():
        row = train[key]
        content = {k:row[k] for k in ('input','output','rejected','preference_issue')}
        if (entry.get('reviewer') != 'Codex' or not entry.get('at') or not entry.get('note')
                or entry['content_hash'] != fingerprint({**content, 'critical_spans':entry['critical_spans'], 'multiplier':entry['multiplier']})):
            raise ValueError('Critical annotation lacks individual content-bound review')
    return annotations

def encode_critical_spans(tokenizer, record, spans, max_length=1024, multiplier=3.0):
    if not math.isfinite(multiplier) or multiplier <= 1:
        raise ValueError('Critical-span multiplier must exceed one')
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Fast tokenizer offsets required')
    if not spans or any(not isinstance(span, str) or not span for span in spans) or len(set(spans)) != len(spans):
        raise ValueError('Explicit nonduplicate spans required')
    encoded = encode_example(tokenizer, record, max_length)
    answer = json.dumps(record['output'], ensure_ascii=False, separators=(',', ':'))
    translated = record['output']['translation']
    value_start = len('{"translation":"')
    if not answer.startswith('{"translation":"'):
        raise ValueError('Expected single translation value')
    intervals = []
    for span in spans:
        if not isinstance(span, str) or not span or translated.count(span) != 1:
            raise ValueError('Each reviewed span must occur exactly once')
        start = translated.index(span)
        escaped_start = len(json.dumps(translated[:start], ensure_ascii=False)[1:-1])
        escaped_end = len(json.dumps(translated[:start+len(span)], ensure_ascii=False)[1:-1])
        intervals.append((value_start+escaped_start, value_start+escaped_end))
    answer_encoding = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    supervised = [v for v in encoded['labels'] if v != -100]
    if list(answer_encoding['input_ids']) != supervised[:-1] or supervised[-1] != tokenizer.eos_token_id:
        raise ValueError('Offset tokenization changed supervised answer/EOS')
    weights = [0.0 if label == -100 else 1.0 for label in encoded['labels']]
    prefix = len(encoded['labels'])-len(supervised)
    covered = set()
    for index,(start,end) in enumerate(answer_encoding['offset_mapping']):
        touched = [i for i,(left,right) in enumerate(intervals) if start < right and end > left]
        if touched:
            weights[prefix+index] = multiplier
            covered.update(touched)
    if len(covered) != len(intervals):
        raise ValueError('Reviewed span has no supervised tokens')
    return {**encoded, 'token_weights':weights}
