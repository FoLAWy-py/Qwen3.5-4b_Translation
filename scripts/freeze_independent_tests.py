"""Source-first confirmation/release freeze, with unresolved overlaps blocking it."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from rapidfuzz import fuzz,process

from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.protocol import make_messages, parse_translation


def normalize(text):
    return re.sub(r'\W+','',unicodedata.normalize('NFKC',text).casefold())


def retrieve_neighbors(key,old,all_text):
    """Full-corpus normalized edit similarity; semantic adjudication stays manual."""
    result=[]
    for text,ratio,_ in process.extract(key,list(old),scorer=fuzz.ratio,score_cutoff=65,limit=None):
        result.append(dict(text=text,locations=old[text],ratio=ratio/100))
    for text,ratio,_ in process.extract(key,list(all_text),scorer=fuzz.ratio,score_cutoff=65,limit=None):
        result.append(dict(text=text,new_group=all_text[text],ratio=ratio/100))
    return sorted(result,key=lambda r:(r['text'],r.get('new_group','')))


def old_inventory(exclude_paths=()):
    texts={};files=[]
    excluded={Path(p).resolve() for p in exclude_paths}
    def collect(value,location):
        if isinstance(value,list):
            for item in value:collect(item,location)
        elif isinstance(value,dict):
            for key in ('input','output','reference'):
                nested=value.get(key)
                if isinstance(nested,dict):
                    text=nested.get('text',nested.get('translation'))
                    if isinstance(text,str):texts.setdefault(normalize(text),[]).append(location+':'+str(value.get('id')))
            if isinstance(value.get('text'),str) and 'target_lang' in value:
                texts.setdefault(normalize(value['text']),[]).append(location)
            for nested in value.values():
                if isinstance(nested,(dict,list)):collect(nested,location)
    for folder in ('data','runs'):
        for path in sorted(Path(folder).rglob('*')):
            if not path.is_file() or path.suffix not in ('.jsonl','.json'):continue
            if path.resolve() in excluded or 'takeover-20261002' in str(path):continue
            files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            values=read_jsonl(path) if path.suffix=='.jsonl' else json.loads(path.read_text(encoding='utf-8-sig'))
            collect(values,str(path))
    return texts,files


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--confirmation',required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--runtime-freeze',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root=Path(args.output)
    if root.exists():raise FileExistsError(root)
    runtime=json.loads(Path(args.runtime_freeze).read_text(encoding='utf-8'))
    if runtime.get('runtime_frozen') is not True:raise ValueError('Final runtime must be frozen before new sources')
    protocol=json.loads(Path('data/independent-test-protocol-20261002.json').read_text(encoding='utf-8'))
    catalog=json.loads(Path('runs/takeover-20261002/historic-source-catalog.json').read_text(encoding='utf-8'))
    old,files=catalog['texts'],catalog['files']
    for entry in files:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Historic source changed after exclusion catalog freeze: '+entry['path'])
    groups={};origins={};all_text={};audits=[];sets={}
    for stage,path in (('confirmation',args.confirmation),('release',args.release)):
        rows=read_jsonl(path);sets[stage]=rows;requirements=protocol[stage]
        if len(rows)<requirements['rows_min']:raise ValueError('Insufficient '+stage+' rows')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate item IDs')
        stagegroups=defaultdict(set)
        for row in rows:
            source=row['source'];audit=row['source_review'];group=row['group_id']
            if not isinstance(group,str) or not group or row['category'] not in protocol['categories']:
                raise ValueError('Explicit source family and allowed category required')
            make_messages(**row['input'])
            parse_translation(json.dumps(row['output'],ensure_ascii=False))
            if source.get('kind') not in ('synthetic_Codex','synthetic_AI','licensed_external','independent_human'):
                raise ValueError('Explicit source composition required')
            if not all(source.get(k) for k in ('name','license','attribution','origin_id')):
                raise ValueError('Traceable license/attribution/origin required')
            if audit.get('reviewer')=='Codex AI' and audit.get('human_acceptance') is True:
                raise ValueError('AI review must not claim independent human acceptance')
            if origins.get(source['origin_id'],group)!=group:
                raise ValueError('Same original source split into different families')
            origins[source['origin_id']]=group
            content={k:row[k] for k in ('input','output','source','group_id','category')}
            if audit.get('content_hash')!=fingerprint(content) or audit.get('status')!='approved':
                raise ValueError('Pre-output source/reference review not content-bound')
            if not all(audit.get(k) is True for k in ('license_checked','attribution_checked','reference_checked','context_checked','ambiguity_resolved','historic_family_checked')):
                raise ValueError('Incomplete individual source preflight')
            if groups.get(group,stage)!=stage:raise ValueError('Cross-set source family')
            groups[group]=stage;stagegroups[group].add(row['category'])
            for text in (row['input']['text'],row['output']['translation']):
                key=normalize(text)
                if key in old:raise ValueError(f'Historic exact source/reference overlap: {row["id"]}')
                if key in all_text and all_text[key]!=group:raise ValueError('Exact new overlap assigned different groups')
                # Approximate matches are retrieval only. Never auto-judge semantics.
                candidates=retrieve_neighbors(key,old,{k:v for k,v in all_text.items() if v!=group})
                retrieval_hash=fingerprint(candidates)
                if candidates and audit.get('overlap_adjudications',{}).get(retrieval_hash)!='distinct_source_family':
                    raise ValueError(f'Unresolved approximate overlap for {row["id"]}: {retrieval_hash}')
                audits.append(dict(id=row['id'],text_hash=fingerprint(text),retrieval_hash=retrieval_hash,candidates=candidates,
                    resolution='distinct_source_family' if candidates else 'no_retrieved_match; individual source audit still required'))
                all_text[key]=group
        if len(stagegroups)<requirements['groups_min'] or any(len(v)!=1 for v in stagegroups.values()):raise ValueError('Source-group coverage/category mismatch')
        strata=Counter((r['category'],r['input']['target_lang']) for r in rows)
        if any(strata[(c,d)]<requirements['category_direction_min'] for c in protocol['categories'] for d in protocol['directions']):raise ValueError('Stratum coverage')
        if sum(bool(r['input'].get('context') or r['input'].get('glossary')) for r in rows)<requirements['context_or_glossary_min']:raise ValueError('Context coverage')
        if sum(r.get('multisentence') is True for r in rows)<requirements['multisentence_min']:raise ValueError('Multisentence coverage')
    root.mkdir(parents=True)
    for stage,rows in sets.items():write_jsonl(root/(stage+'.jsonl'),rows)
    write_json(root/'source-audit.json',audits)
    write_json(root/'freeze.json',dict(at=now(),protocol_hash=fingerprint(protocol),runtime_freeze_hash=fingerprint(runtime),
        sets={k:dict(rows=len(v),groups=len({r['group_id'] for r in v}),hash=fingerprint(v)) for k,v in sets.items()},
        historic_files=files,historic_files_hash=fingerprint(files),source_audit_hash=fingerprint(audits),
        retrieval='RapidFuzz normalized full-corpus ratio>=65; changed before outputs to make full all-source audit tractable, never automatic semantic grading',
        outputs_seen=False,reviewer='Codex AI if declared per row; no human acceptance inferred',release_approved=False))


if __name__=='__main__':main()
