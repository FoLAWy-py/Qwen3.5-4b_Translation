"""Reuse only frozen identical readings; all changed factorial outputs remain unread."""
import argparse
import json
from collections import Counter
from pathlib import Path

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def binding(row):
    return fingerprint({key: row.get(key) for key in ('input','raw','reference','ended','prediction')})


def accept_manual(row, decision):
    if (decision.get('reviewer') != 'Codex' or decision.get('generation_hash') != fingerprint(row)
            or decision.get('verdict') not in ('pass','minor','major','critical')
            or not decision.get('note') or type(decision.get('language_correct')) is not bool):
        raise ValueError('Manual reading is missing or stale')
    return decision


def qwen35_review_caches(plan,split):
    from scripts.audit_qwen35_stage import legacy_known
    from archive.qwen3.scripts.finalize_v13_optimization import complete_review
    result=[]
    roles=plan['semantic_cache_roles']
    assert set(roles)=={'frozen_qwen35','v7'}
    for role in roles:
        target=next(t for t in plan['targets'] if t['role']==role and t['split']==split)
        assert target['cached'] is True
        stem=str(Path(target['output']).with_suffix(''))
        rows,notes=(legacy_known if role=='v7' and split=='known' else complete_review)(stem,plan['datasets'][split])
        assert fingerprint(rows)==target['generation_hash'] and fingerprint(notes)==target['decisions_hash']
        summary=json.loads(Path(stem+'.summary.json').read_text(encoding='utf-8'))
        assert fingerprint(summary)==target['summary_hash'] and summary['adapter_sha256']==target['adapter_sha256']
        assert summary['data_hash']==plan['datasets'][split]['hash'] and summary['prompt_hash']==plan['prompt_hash']
        result.extend((role+'/'+split,note) for note in notes)
    return result


def collect(row, ref, manual, caches):
    if row['input'] != ref['input'] or row['reference'] != ref['output']:
        raise ValueError('Frozen reference changed')
    if row['id'] in manual:
        decision = accept_manual(row, manual[row['id']])
        method = 'new_individual_reading'
    else:
        matches = [(name,d) for name,d in caches if d['id'] == row['id'] and d['binding_hash'] == binding(row)]
        if not matches:
            return None
        if len({(d['verdict'],d['language_correct']) for _,d in matches}) != 1:
            raise ValueError('Conflicting exact readings require explicit re-review')
        method, decision = matches[0]
    return {'id':row['id'],'group_id':ref['group_id'],'category':ref['category'],
            'target_lang':ref['input']['target_lang'],'verdict':decision['verdict'],'note':decision['note'],
            'language_correct':decision['language_correct'],'format_valid':'prediction' in row,
            'ended':row.get('ended'), 'binding_hash':binding(row), 'generation_hash':fingerprint(row),
            'output_hash':fingerprint({key:row[key] for key in ('input','raw','reference')}),
            'reviewer':'Codex','review_source':method}


def main():
    parser = argparse.ArgumentParser()
    # The frozen plan, rather than a historical version allow-list, owns roles.
    parser.add_argument('--role', required=True)
    parser.add_argument('--plan', default='runs/v12-factorial-evaluation-plan.json')
    parser.add_argument('--split', required=True, choices=('known','public'))
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    target = next(target for target in plan['targets'] if target['role'] == args.role and target['split'] == args.split)
    rows = read_jsonl(target['output'])
    refs_list = read_jsonl(plan['datasets'][args.split]['path'])
    if fingerprint(refs_list) != plan['datasets'][args.split]['hash']:
        raise ValueError('Frozen development set changed')
    refs = {row['id']:row for row in refs_list}
    if len({row['id'] for row in rows}) != len(rows) or {row['id'] for row in rows} - set(refs):
        raise ValueError('Duplicate or unknown generated IDs')
    summary_path = Path(target['output']).with_suffix('.summary.json')
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary['adapter_sha256'] != target['adapter_sha256'] or summary['prompt_hash'] != plan['prompt_hash']
                or summary['base'] != plan['base'] or summary['data_hash'] != plan['datasets'][args.split]['hash']
                or summary['quantization'] != 'nf4' or summary['decoding'] != {'do_sample':False,'max_length':1024,'max_new_tokens':256}):
            raise ValueError('Generation protocol differs from frozen comparison')
    stem = str(Path(target['output']).with_suffix(''))
    manual_path = Path(stem+'-manual.jsonl')
    manual_rows = read_jsonl(manual_path) if manual_path.exists() else []
    manual = {row['id']:row for row in manual_rows}
    if len(manual) != len(manual_rows) or set(manual) - {row['id'] for row in rows}:
        raise ValueError('Duplicate or ungenerated manual decision')
    caches = []
    is_qwen35=plan['base']['model_id']=='Qwen/Qwen3.5-4B'
    if is_qwen35:
        caches=qwen35_review_caches(plan,args.split)
    elif args.split == 'known':
        from archive.qwen3.scripts.compare_v12_blind import DECISION_HASH
        masked_rows = read_jsonl('runs/v12-blind-decisions.jsonl')
        if fingerprint(masked_rows) != DECISION_HASH:
            raise ValueError('Frozen pre-unmask decisions changed')
        masked = {(row['packet_id'],row['label']):row for row in masked_rows}
        stems = {'base':'v12-base','v7':'v7-critical'}
        allowed_cache_roles = plan.get('semantic_cache_roles',['v7'] if 'v13' in args.plan else ['base','v7'])
        cache_roles = [(role,stems[role]) for role in allowed_cache_roles]
        for role, original_stem in cache_roles:
            original = {r['id']:r for r in read_jsonl(f'runs/{original_stem}-dev.jsonl')}
            directions = {r['id']:r for r in read_jsonl(f'runs/{original_stem}-dev-semantic.jsonl')}
            for decision in read_jsonl(f'runs/v12-blind-{role}-semantic.jsonl'):
                raw = original[decision['id']]
                digest = fingerprint({key:raw[key] for key in ('input','raw','reference')})
                direction = directions[decision['id']]
                frozen = masked[(decision['packet_id'],decision['label'])]
                if (decision['output_hash'] != digest or direction['output_hash'] != digest
                        or decision['verdict'] != frozen['grade'] or decision['note'] != frozen['note']
                        or decision['format_valid'] != ('prediction' in raw) or decision['ended'] != raw.get('ended')):
                    raise ValueError('Frozen masked baseline cache changed')
                caches.append(('masked-'+role,{**decision,'binding_hash':binding(raw),
                                              'language_correct':direction['language_correct']}))
    for other in plan['targets']:
        if other['split'] != args.split or other['output'] == target['output']:
            continue
        if is_qwen35 and other.get('cached'):
            continue  # Already validated, including the legacy v7 binding bridge.
        other_path = Path(other['output'])
        semantic_path = other_path.with_name(other_path.stem+'-semantic.jsonl')
        if not semantic_path.exists():
            continue
        other_rows = {row['id']:row for row in read_jsonl(other_path)}
        for decision in read_jsonl(semantic_path):
            raw = other_rows[decision['id']]
            if (decision['binding_hash'] != binding(raw) or decision['reviewer'] != 'Codex'
                    or decision['verdict'] not in ('pass','minor','major','critical')
                    or not decision.get('note') or type(decision.get('language_correct')) is not bool):
                raise ValueError('Prior factorial review cache changed')
            caches.append((other['role']+'/'+other['split'], decision))
    decisions, pending = [], []
    for row in rows:
        if 'raw' not in row:
            raise ValueError('Failed generation requires explicit inspection before semantic review')
        decision = collect(row, refs[row['id']], manual, caches)
        (decisions if decision is not None else pending).append(decision if decision is not None else row)
    write_jsonl(stem+'-semantic.jsonl', decisions)
    write_jsonl(stem+'-unmatched.jsonl', pending)
    report = {'at':now(),'role':args.role,'split':args.split,'generated':len(rows),'expected':len(refs),
              'reviewed':len(decisions),'pending_individual_reading':len(pending),
              'complete':len(decisions)==len(refs) and summary_path.exists(),
              'verdicts':dict(Counter(row['verdict'] for row in decisions)),
              'review_sources':dict(Counter(row['review_source'] for row in decisions)),
              'release_approved':False,'scope':'Exact input/raw/reference/ended/prediction equality only. Changed outputs require explicit content-bound Codex reading. Public DEV is auxiliary, not release.'}
    write_json(stem+'-semantic.summary.json', report)
    print(report,flush=True)


if __name__ == '__main__':
    main()
