"""Root's completed individual audit of48 context-contrast teacher labels."""
import hashlib
from collections import Counter
from pathlib import Path
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = 'c8899823add682ce349102b8d9f065f591f3b7e8be941721996d5915d08e6cab'
FIXES = {
    '005-en': ('Is this joint damaged?', '接头不添加背景中的threaded性质。'),
    '010-zh-CN': ('鳎鱼看起来不错。', 'sole保留鳎鱼，不扩大为比目鱼，也不添加鱼片。'),
    '011-zh-CN': ('航站楼很繁忙。', 'busy保留繁忙，不强制变为拥挤。'),
    '015-zh-CN': ('球棒不见了。', 'baseball语境中的bat是球棒，教师错译为蝙蝠。'),
    '018-zh-CN': ('卧底很难发现。', 'mole保留潜伏在组织中的卧底含义，不泛化为间谍。'),
    '020-zh-CN': ('吠叫声被去掉了。', '被动句不添加教师输出中的我们。'),
    '021-zh-CN': ('这封信的意思不明确。', 'letter信件的unclear指表达的意思不明确，改善中文搭配。'),
    '022-zh-CN': ('这个字母看不清。', '模糊字符语境中的unclear指视觉上看不清，改善中文搭配。'),
    '023-zh-CN': ('泳池几乎没水了。', 'nearly empty只说明水量，不把背景排水行动写入译文。'),
    '024-zh-CN': ('连接池几乎空了。', '不将池接近空推断为连接被用尽的原因。'),
}

def main():
    destination = Path('data/prepared/v7-context/train.jsonl')
    if destination.exists():
        raise ValueError('Accepted context corpus immutable')
    path = Path('data/generated/v7_context_candidates.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Teacher outputs changed; read again')
    rows = read_jsonl(path)
    refs = {r['id']: r for r in read_jsonl('data/prepared/v7-context/references.jsonl')}
    if len(rows) != 48 or {r['id'] for r in rows} != set(refs):
        raise ValueError('Incomplete individually reviewed batch')
    decisions = []
    for row in rows:
        if row['input'] != refs[row['id']]['input'] or row['group_id'] != refs[row['id']]['group_id']:
            raise ValueError('Frozen source or context changed')
        raw_hash = fingerprint(row)
        key = row['id'].removeprefix('v7-context-')
        note = '逐条按实际原文和语境核对，接受等义表达；不强制套用双向参考措辞。'
        if key in FIXES:
            answer, note = FIXES[key]
            row['output'] = {'translation': answer}
        reviewed(row, 'Codex read all48 actual teacher outputs against immutable sources and contexts; explicit repairs; same-source context alternatives retained in one group')
        validate_record(row, True, True)
        decisions.append({'id': row['id'], 'action': 'correct' if key in FIXES else 'accept_teacher', 'note': note,
            'raw_record_hash': raw_hash, 'reviewed_hash': fingerprint({'input': row['input'], 'output': row['output']})})
    if not set(FIXES).issubset({r['id'].removeprefix('v7-context-') for r in rows}):
        raise ValueError('Repair ID absent')
    write_jsonl(destination, rows)
    write_jsonl('data/generated/v7_context_decisions.jsonl', decisions)
    report = {'at': now(), 'reviewed_rows': 48, 'groups': len({r['group_id'] for r in rows}),
        'actions': dict(Counter(r['action'] for r in decisions)), 'raw_sha256': RAW_SHA, 'train_hash': fingerprint(rows),
        'context_dependent_english_source_rows': 24,
        'scope': 'Training-only same-source contrast labels. Reverse Chinese sources already specify senses. No release evidence.'}
    write_json('runs/v7-context-data-acceptance.json', report)
    print(report)

if __name__ == '__main__':
    main()
