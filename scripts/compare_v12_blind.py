"""Unmask only complete frozen paired decisions and report known-DEV limits."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from data.v12_blind_reviews import PACKETS_HASH
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.paired_stats import paired_pass_interval

DECISION_HASH = '6b77eab99ae56d60feb604ee2fcfe719cbada0b3cfee823ef5025c0d4f66df39'


def validate_decisions(packets, decisions):
    if fingerprint(packets) != PACKETS_HASH or fingerprint(decisions) != DECISION_HASH:
        raise ValueError('Frozen masked evidence changed')
    indexed = {(r['packet_id'], r['label']): r for r in decisions}
    if len(indexed) != 400 or len(decisions) != 400 or len(packets) != 200:
        raise ValueError('Complete200 pairs required before unmasking')
    for packet in packets:
        for label in ('A', 'B'):
            decision = indexed[(packet['id'], label)]
            if (decision['packet_hash'] != fingerprint(packet)
                    or decision['input_raw_hash'] != fingerprint({'input': packet['input'], 'raw': packet['outputs'][label]})
                    or decision['grade'] not in ('pass', 'minor', 'major', 'critical')
                    or not decision['note']):
                raise ValueError('Unbound masked decision')
    return indexed


def main():
    packets = read_jsonl('runs/v12-blind-packets.jsonl')
    decisions = read_jsonl('runs/v12-blind-decisions.jsonl')
    indexed = validate_decisions(packets, decisions)
    receipt_path = Path('runs/v12-blind-unmask-receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['decision_hash'] != DECISION_HASH or receipt['packets_hash'] != PACKETS_HASH:
            raise ValueError('Previously unmasked evidence changed')
    else:
        # Receipt precedes key loading. Subsequent annotation writes are prohibited.
        write_json(receipt_path, {'at': now(), 'decision_hash': DECISION_HASH,
                                 'packets_hash': PACKETS_HASH, 'decisions': 400,
                                 'status': 'Frozen complete evidence; identity map may now be read'})
    identities = json.loads(Path('runs/v12-blind-identity-key.json').read_text(encoding='utf-8'))
    if identities['packets_hash'] != PACKETS_HASH or len(identities['mapping']) != 200:
        raise ValueError('Identity map differs from frozen packet order')
    refs = {r['id']: r for r in read_jsonl('data/prepared/v4/dev.jsonl')}
    if fingerprint(list(refs.values())) != 'a4ec0478d5aca8b263e605bfe3e7ec2b57dd9411510135cfaf227575ede03237':
        raise ValueError('Frozen DEV changed')
    raw, old, summaries = {}, {}, {}
    for role, stem in (('base', 'v12-base'), ('v7', 'v7-critical')):
        raw[role] = {r['id']: r for r in read_jsonl(f'runs/{stem}-dev.jsonl')}
        old[role] = {r['id']: r for r in read_jsonl(f'runs/{stem}-dev-semantic.jsonl')}
        summaries[role] = json.loads(Path(f'runs/{stem}-dev.summary.json').read_text(encoding='utf-8'))
        if set(raw[role]) != set(refs) or set(old[role]) != set(refs):
            raise ValueError('Incomplete original run')
    for key in ('base', 'prompt_hash', 'data_hash', 'quantization', 'decoding'):
        if summaries['base'][key] != summaries['v7'][key]:
            raise ValueError('Matched protocol changed')
    if summaries['base']['adapter_sha256'] is not None or summaries['v7']['adapter_sha256'] != '2f62f2610b99fe455b9ae4293f2e5d40e026884dfd0547fefc374974f36e3424':
        raise ValueError('Weight identity differs')
    evidence = {'base': [], 'v7': []}
    differences = []
    for packet in packets:
        mapping = identities['mapping'][packet['id']]
        ref = refs[mapping['source_id']]
        if mapping['packet_hash'] != fingerprint(packet) or packet['input'] != ref['input'] or mapping['group_id'] != ref['group_id']:
            raise ValueError('Packet identity binding differs')
        if {mapping['A'], mapping['B']} != {'base', 'v7'}:
            raise ValueError('Invalid paired roles')
        for label in ('A', 'B'):
            role = mapping[label]
            generation = raw[role][ref['id']]
            decision = indexed[(packet['id'], label)]
            if (generation['raw'] != packet['outputs'][label] or generation['input'] != ref['input']
                    or generation['reference'] != ref['output']):
                raise ValueError('Mapped generation changed')
            row = {'id': ref['id'], 'group_id': ref['group_id'], 'category': ref['category'],
                   'target_lang': ref['input']['target_lang'], 'verdict': decision['grade'],
                   'note': decision['note'], 'packet_id': packet['id'], 'label': label,
                   'output_hash': fingerprint({k: generation[k] for k in ('input', 'raw', 'reference')}),
                   'format_valid': 'prediction' in generation, 'ended': generation['ended'],
                   'reviewer': decision['reviewer']}
            evidence[role].append(row)
            if row['verdict'] != old[role][ref['id']]['verdict']:
                differences.append({'role': role, 'id': row['id'], 'old': old[role][ref['id']]['verdict'],
                                    'new': row['verdict'], 'note': row['note']})
    counts = {role: dict(Counter(r['verdict'] for r in rows)) for role, rows in evidence.items()}
    strata = {}
    for role, rows in evidence.items():
        buckets = defaultdict(Counter)
        for row in rows:
            buckets[row['category'] + '/' + row['target_lang']][row['verdict']] += 1
        strata[role] = {key: dict(value) for key, value in buckets.items()}
        write_jsonl(f'runs/v12-blind-{role}-semantic.jsonl', rows)
    stats = paired_pass_interval(evidence['v7'], evidence['base'])
    report = {'at': now(), 'rows': 200, 'groups': 100, 'counts': counts, 'strata': strata,
              'v7_vs_base': stats, 'packets_hash': PACKETS_HASH, 'frozen_decision_hash': DECISION_HASH,
              'format_valid': {role: sum(r['format_valid'] for r in rows) for role, rows in evidence.items()},
              'eos': {role: sum(r['ended'] for r in rows) for role, rows in evidence.items()},
              'differences_from_prior_review': differences, 'release_approved': False,
              'limitations': 'Same Codex evaluator had prior knowledge of outputs. All200 pairs newly read with identity key withheld until400 content-bound decisions froze; semantic display removes JSON wrappers. This is a masked re-review, not an independent fully blinded evaluator. Known DEV repeatedly used; no causal algorithm or release claim.'}
    write_json('runs/v12-blind-comparison.json', report)
    lines = ['# 原始base与v7身份隐藏配对复审', '', '| 模型 | pass | minor | major | JSON |', '|---|---:|---:|---:|---:|']
    for role in ('base', 'v7'):
        c = counts[role]
        lines.append(f"| {role} | {c.get('pass',0)} | {c.get('minor',0)} | {c.get('major',0)+c.get('critical',0)} | {report['format_valid'][role]}/200 |")
    interval = stats['group_bootstrap_95_interval']
    lines += ['', f"v7通过率相对base提高{stats['pass_difference']*100:.1f}个百分点，来源组95%区间[{interval[0]*100:.1f},{interval[1]*100:.1f}]。",
              '', '全部200对新读，400项判断在读取身份映射前绑定输入与原始输出冻结。格式独立统计；语义视图仅解析已有JSON包装，没有改写生成结果。',
              '', '同一评审者曾见过输出，身份隐藏不能消除已有记忆，不能称完全独立盲审。已知DEV被反复研究使用；结果不能证明发布质量或算法因果收益。新旧判定分歧完整保留于JSON；旧报告不覆盖。',
              '', '下一步固定训练token预算的起点×数据2×2筛选；新发布测试仍待候选冻结后建立。Goal未完成。']
    Path('runs/v12-blind-comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({'counts': counts, 'v7_vs_base': stats, 'changed_old_decisions': len(differences), 'release_approved': False}, flush=True)


if __name__ == '__main__':
    main()
