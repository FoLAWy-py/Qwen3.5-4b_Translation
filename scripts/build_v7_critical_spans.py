"""Freeze root-selected spans in training references; no held-out annotation."""
from pathlib import Path
from transformers import AutoTokenizer
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import encode_critical_spans
from witrans_tools.data import validate_record

# Each literal was individually selected from the approved positive translation.
SPANS = {
    '001-zh-CN':['只有我们两个人都不能在你到达之前回到家时'],
    '004-en':['When I tighten the screws'],
    '010-zh-CN':['闭馆前一小时停止接待游客'],
    '011-zh-CN':['在小路分岔之前','通向坡上'],
    '012-zh-CN':['多一晚'],
    '012-en':['even if breakfast is more expensive'],
    '013-zh-CN':['两班出发时间','接续班次'],
    '013-en':['connecting service'],
    '014-zh-CN':['它不包括导览步行，导览步行必须另行预订'],
    '015-zh-CN':['沿着雪松环线行进'],
    '024-zh-CN':['馅饼','把它从托盘上移走可能会弄碎底部'],
    '031-zh-CN':['这个指标','如果可能与另一个指标混淆，就不要把它缩写成保留度'],
    '031-en':['if it could be confused with another metric, do not abbreviate it to retention'],
    '034-zh-CN':['我们说好了等所有人到齐'],
    '035-zh-CN':['现在该由你来决定了'],
    '040-zh-CN':['她对自己的想法守口如瓶'],
}

def main():
    destination = Path('data/prepared/v7-critical-spans/annotations.json')
    if destination.exists():
        raise ValueError('Annotations already frozen')
    train = read_jsonl('data/prepared/v6-preference/train.jsonl')
    if fingerprint(train) != 'daf077b8c46d389f70192bdb868863425f45e7c494deb2e3609711fe456af826':
        raise ValueError('Matched training pool changed')
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B', local_files_only=True, trust_remote_code=False)
    selected = {r['id']:r for r in train if r['id'].startswith('v8-multi-')}
    if set(selected) != {'v8-multi-'+key for key in SPANS}:
        raise ValueError('Explicit span coverage changed')
    annotations, counts = [], []
    for key, spans in SPANS.items():
        row = selected['v8-multi-'+key]
        validate_record(row, True, True)
        encoded = encode_critical_spans(tokenizer, row, spans)
        content = {k:row[k] for k in ('input','output','rejected','preference_issue')}
        binding = fingerprint({**content, 'critical_spans':spans, 'multiplier':3.0})
        annotations.append({'id':row['id'], 'critical_spans':spans, 'multiplier':3.0,
            'reviewer':'Codex', 'at':now(), 'content_hash':binding,
            'note':'Individually selected positive spans preserving role, condition, concept, quantity, or contextual sense; absent unwanted words cannot be annotated.'})
        counts.append({'id':row['id'], 'supervised_tokens':sum(x!=-100 for x in encoded['labels']),
            'upweighted_tokens':sum(w>1 for w in encoded['token_weights']),
            'eos_weight':encoded['token_weights'][-1]})
    report = {'at':now(), 'train_hash':fingerprint(train), 'annotations':annotations,
        'annotation_hash':fingerprint(annotations), 'encoding_probe':counts,
        'method':'Only chosen-answer normalized NLL uses weight3 on explicit spans; unweighted summed preference log-likelihood, prompt masking and EOS supervision unchanged.',
        'scope':'16 approved training-only natural-error pairs. Other69 pairs use uniform NLL. No dev/test spans, no automatic error inference, no efficacy claim.',
        'release_approved':False}
    write_json(destination, report)
    print({'annotated_rows':len(annotations), 'upweighted_tokens':sum(c['upweighted_tokens'] for c in counts),
        'train_hash':report['train_hash'], 'annotation_hash':report['annotation_hash']})

if __name__ == '__main__':
    main()
