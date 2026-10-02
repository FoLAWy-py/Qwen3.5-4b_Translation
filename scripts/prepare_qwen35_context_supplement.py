"""Audit authored TRAIN source drafts; publish only after explicit overlap readings."""
import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.data import encode_example,validate_record


def normalized(text):
    return re.sub(r'[^\w]','',text.casefold())


def grams(text):
    tokens=re.findall(r'[a-z0-9]+|[\u3400-\u9fff]',text.casefold())
    return {tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--approve-readings')
    parser.add_argument('--audit-reference',default='runs/qwen35-v3-context-supplement-overlap-audit.json')
    args=parser.parse_args()
    dest=Path(args.output)
    assert not dest.exists(), 'Preserve prior evidence'
    draft=json.loads(Path('data/qwen35-v3-context-supplement-draft.json').read_text(encoding='utf-8'))
    assert len(draft['sources'])==len({s['key'] for s in draft['sources']})==20
    old={}; groups=set(); file_hashes={}
    for path in sorted(set(Path('data').rglob('*.jsonl'))|set(Path('runs').glob('*.jsonl'))):
        if 'qwen35-v3-context-supplement' in str(path):
            continue
        rows=read_jsonl(path)
        relevant=[]
        for row in rows:
            if not isinstance(row,dict):
                continue
            item=row.get('input')
            if not isinstance(item,dict) or not isinstance(item.get('text'),str):
                continue
            relevant.append(row)
            groups.add(row.get('group_id'))
            key=normalized(item['text'])
            old.setdefault(key,dict(id=row.get('id'),group_id=row.get('group_id'),
                path=str(path),text=item['text'],grams=grams(item['text'])))
        if relevant:
            file_hashes[str(path)]=fingerprint(relevant)
    assert len(old)>=416, 'All old source scan unexpectedly incomplete'
    candidates=[]; audit=[]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-4B',local_files_only=True,trust_remote_code=False)
    for source in draft['sources']:
        group='q35-v3-supplement-'+source['key']
        assert group not in groups
        for target,src,answer in (('zh-CN',source['en'],source['zh']),('en',source['zh'],source['en'])):
            assert normalized(src) not in old
            glossary=source['terms'] if target=='zh-CN' else {v:k for k,v in source['terms'].items()}
            # A term may be grammatically split; retain only verbatim source
            # occurrences so each supplied mapping can actually apply.
            glossary={k:v for k,v in glossary.items() if k in src}
            assert glossary
            row=dict(id=group+'-'+target,group_id=group,category=source['category'],
                input=dict(text=src,target_lang=target,context=source['context'],glossary=glossary),
                output=dict(translation=answer),source=dict(name='Original Codex-authored context/terminology TRAIN supplement',
                    license='Original synthetic project data authored for this authorized task',training_allowed=True,
                    external_labeling_allowed=True,construction='New fictional scenario; no third-party text copied. Two directions share one family.',
                    author='Codex AI',draft_source_hash=fingerprint(source)),
                review=dict(status='approved',reviewer='Codex AI (also author; not independent human review)',at=now(),
                    method='Individual bilingual author review of context, roles, negation, quantities, conditions, terms and multi-sentence relationships',
                    content_hash=fingerprint(dict(input=dict(text=src,target_lang=target,context=source['context'],glossary=glossary),output=dict(translation=answer))),
                    notes=source['note']))
            validate_record(row,True,True,purpose='training')
            encoded=encode_example(tokenizer,row,1024)
            newgrams=grams(src)
            near=sorted(old.values(),key=lambda item:len(newgrams&item['grams'])/max(1,len(newgrams|item['grams'])),reverse=True)[:3]
            nearest=[dict(id=item['id'],path=item['path'],group_id=item['group_id'],text=item['text'],
                trigram_jaccard=len(newgrams&item['grams'])/max(1,len(newgrams|item['grams'])),
                character_similarity=SequenceMatcher(None,normalized(src),normalized(item['text']),autojunk=False).ratio()) for item in near]
            candidates.append(row)
            audit.append(dict(id=row['id'],input_hash=fingerprint(row['input']),reference_hash=fingerprint(row['output']),
                full_tokens=len(encoded['input_ids']),nearest_old_sources=nearest))
    assert len(candidates)==40 and Counter(r['category']+'/'+r['input']['target_lang'] for r in candidates)==Counter({c+'/'+d:4 for c in ('daily','travel','food','academic','hard') for d in ('en','zh-CN')})
    report=dict(at=now(),draft_hash=fingerprint(draft),rows=40,source_groups=20,old_unique_source_texts=len(old),
        old_input_file_hashes=file_hashes,rows_hash=fingerprint(candidates),audit_hash=fingerprint(audit),audit=audit,
        exact_text_group_overlap=False,context_rows=40,glossary_rows=40,
        scope='TRAIN-only original synthetic supplement; no independent quality estimate, no confirmation data, no fabricated negatives.',
        ready_for_training=False,stage_goal_complete=False,release_approved=False)
    if args.approve_readings:
        prior=json.loads(Path(args.audit_reference).read_text(encoding='utf-8'))
        assert fingerprint(audit)==prior['audit_hash'] and fingerprint(draft)==prior['draft_hash']
        readings=read_jsonl(args.approve_readings)
        assert len(readings)==len({r['id'] for r in readings})==40
        decisions={r['id']:r for r in readings}
        for item,row in zip(audit,candidates):
            reading=decisions[item['id']]
            assert reading['audit_row_hash']==fingerprint(item) and reading['decision']=='distinct_source'
            assert reading['reviewer']=='Codex AI' and reading['note'].strip()
            row['source_overlap_review']=dict(reading_hash=fingerprint(reading),audit_row_hash=fingerprint(item))
        target=Path('data/prepared/qwen35-v3-context-supplement')
        assert not target.exists()
        write_jsonl(target/'train.jsonl',candidates)
        report.update(ready_for_training=True,train_path=str(target/'train.jsonl'),train_hash=fingerprint(candidates),
                      readings_hash=fingerprint(readings))
    write_json(dest,report)
    print(dict(output=str(dest),rows=40,old_sources=len(old),ready=report['ready_for_training']))


if __name__=='__main__':
    main()
