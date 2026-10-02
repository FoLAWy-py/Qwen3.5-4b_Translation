"""Apply root's source-grounded reading of all60 next-batch teacher outputs."""
import hashlib
from pathlib import Path
from collections import Counter
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = '1b30ce7db053b10683836a985a8b3821345556872b200d76e64243f94672d9d0'
FIXES = {
    '005-zh-CN': ('这条缆索铁路沿山坡向上运行。它是地面缆车，所以车厢沿轨道行驶，不是悬挂在空中的缆绳上。', 'funicular是地面缆车，不是齿轨铁路。'),
    '010-zh-CN': ('包好饺子封口之前，馅料需要先冷却。否则饺子皮可能变软并破裂。', 'dumplings留英文，恢复完整中文词及自然封口表述。'),
    '020-zh-CN': ('演员阵容已经被撤下了。', 'removed保留撤下，不使用含替换的撤换；原双语参考措辞不完全等义，分方向核对。'),
    '026-zh-CN': ('兴趣正在增加。', '语境仅消歧，不将students及astronomy扩展为原文新事实。'),
}

def main():
    destination = Path('data/prepared/v5-seed/train.jsonl')
    if destination.exists():
        raise ValueError('Accepted batch already frozen')
    path = Path('data/generated/v5_seed_candidates.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Candidates changed; review again')
    rows = read_jsonl(path)
    refs = {r['id']: r for r in read_jsonl('data/prepared/v5-seed/references.jsonl')}
    assert len(rows) == len(refs) == 60
    decisions = []
    for row in rows:
        assert row['input'] == refs[row['id']]['input']
        key = row['id'].removeprefix('v5-seed-')
        if key in FIXES:
            text, note = FIXES[key]
            row['output'] = {'translation': text}
        else:
            note = '逐条核对原文与语境，接受等义措辞。'
        if key == '020-en':
            note = '中文原文撤换含替换，教师replaced忠实；不强制套用原参考removed。双向各按实际原文审核。'
        reviewed(row, 'Codex individually read all60 teacher candidates; repairs bound to original immutable raw')
        validate_record(row, True, True)
        decisions.append({'id': row['id'], 'at': now(), 'action': 'correct' if key in FIXES else 'accept_teacher', 'note': note,
            'reviewed_hash': fingerprint({'input': row['input'], 'output': row['output']})})
    assert set(FIXES).issubset({r['id'].removeprefix('v5-seed-') for r in rows})
    write_jsonl(destination, rows)
    write_jsonl('data/generated/v5_seed_decisions.jsonl', decisions)
    write_json('runs/v5-seed-data-acceptance.json', {'at': now(), 'reviewed_rows': 60, 'groups': len({r['group_id'] for r in rows}),
        'actions': dict(Counter(r['action'] for r in decisions)), 'raw_sha256': RAW_SHA, 'train_hash': fingerprint(rows),
        'scope': 'Next training seed batch only; does not alter v4 frozen training; original reference wording discrepancy recorded for020'})
    print({'accepted_training_rows': len(rows), 'actions': dict(Counter(r['action'] for r in decisions))})

if __name__ == '__main__':
    main()
