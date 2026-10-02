"""Freeze genuine training context-family inputs before teacher labeling."""
from pathlib import Path
from data.v7_context_pairs import PAIRS
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.data import validate_record

def main():
    destination=Path('data/prepared/v7-context')
    if destination.exists():
        raise ValueError('Context pool already frozen')
    old={r['input']['text'].strip().casefold() for p in Path('data/prepared').glob('**/*.jsonl') for r in read_jsonl(p) if 'input' in r}
    excluded={family for family,en,zh,_ in PAIRS if en.strip().casefold() in old or zh.strip().casefold() in old}
    rows=[]
    for i,(family,en,zh,context) in enumerate(PAIRS,1):
        if family in excluded:
            continue
        for target,text,answer in (('zh-CN',en,zh),('en',zh,en)):
            row={'id':f'v7-context-{i:03}-{target}','group_id':f'v7-context-{family}','category':'hard',
                'input':{'text':text,'target_lang':target,'context':context,'glossary':{}},'output':{'translation':answer},
                'source':{'name':'Original Codex-authored same-source training context contrasts','license':'Original synthetic project data',
                    'training_allowed':True,'external_labeling_allowed':True}}
            reviewed(row,'Both context senses individually checked; source shared across senses and reverse direction remains in same family; not copied from development errors')
            validate_record(row,True,True)
            rows.append(row)
    write_jsonl(destination/'references.jsonl',rows)
    write_jsonl('data/v7_context_label_inputs.jsonl',[{k:v for k,v in r.items() if k not in ('output','review')} for r in rows])
    write_json(destination/'manifest.json',{'at':now(),'rows':len(rows),'groups':len({r['group_id'] for r in rows}),'hash':fingerprint(rows),
        'excluded_duplicate_families':sorted(excluded),'scope':'Training-only contrasts; English target direction has explicit Chinese sense, context-dependence evaluated separately by direction; exact source exclusion, not semantic near-duplicate guarantee'})
    print({'rows':len(rows),'groups':len({r['group_id'] for r in rows}),'excluded_families':sorted(excluded)})

if __name__=='__main__':
    main()
