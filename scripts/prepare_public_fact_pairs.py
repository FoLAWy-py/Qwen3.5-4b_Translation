"""Freeze reviewed single-fact derivatives; whole linked family remains train-only."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from data.public_fact_variants import PUBLIC_TRAIN_HASH, VARIANTS
from scripts.audit_v10_sources import grams, normalized
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.data import encode_example,validate_record

DEST = Path('data/prepared/v14-public-fact-pairs')


def main():
    if DEST.exists():
        raise ValueError('Preserve frozen fact-pair evidence')
    sources = read_jsonl('data/prepared/v12-public-short/train.jsonl')
    if fingerprint(sources) != PUBLIC_TRAIN_HASH:
        raise ValueError('Previously read public training sources changed')
    parent = {int(row['public_provenance']['candidate_id'].rsplit('-',1)[1]):row
              for row in sources if row['input']['target_lang']=='zh-CN'}
    if set(VARIANTS)-set(parent):
        raise ValueError('A variant parent is not train-authorized')
    prior, snapshots = {},[]
    for path in sorted(Path('data/prepared').glob('**/*.jsonl')):
        records = read_jsonl(path)
        snapshot = []
        for row in records:
            inp = row.get('input')
            if not isinstance(inp,dict) or not inp.get('text'):
                continue
            heldout = any(word in path.name.casefold() for word in ('dev','test','acceptance')) or row.get('source',{}).get('training_allowed') is False
            if not heldout:
                continue
            key = normalized(inp['text'])
            item = prior.setdefault(key,{'text':inp['text'],'ids':set(),'groups':set()})
            item['ids'].add(row['id']); item['groups'].add(row['group_id'])
            snapshot.append({'id':row['id'],'group_id':row['group_id'],'text':inp['text']})
        if snapshot:
            snapshots.append({'path':str(path),'source_fields_hash':fingerprint(snapshot)})
    entries = list(prior.values())
    features = [grams(row['text']) for row in entries]
    postings = defaultdict(list)
    for index,feature in enumerate(features):
        for gram in feature:
            postings[gram].append(index)
    packets, positives, excluded, token_lengths = [],[],[],[]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B',local_files_only=True)
    for number,(changed_en,changed_zh,axis,note) in VARIANTS.items():
        original = parent[number]
        en = original['input']['text']; zh = original['output']['translation']
        texts = [en,zh,changed_en,changed_zh]
        matches = []
        for text in texts:
            feature = grams(text)
            counts = Counter(index for gram in feature for index in postings[gram])
            for index,common in counts.items():
                score = common/math.sqrt(len(feature)*len(features[index])) if feature and features[index] else 0
                if score >= .45 or normalized(text)==normalized(entries[index]['text']):
                    matches.append({'score':round(score,6),'ids':sorted(entries[index]['ids']),
                                    'groups':sorted(entries[index]['groups'])})
        if matches:
            excluded.append({'parent_id':original['id'],'matches':matches,
                             'reason':'Conservative whole-family heldout source retrieval exclusion'})
            continue
        provenance = {**original['source'],
                      'construction':'Codex-reviewed one-fact derivatives of credited public TRAIN pair; not teacher/model errors or DEV derivatives',
                      'modifications':original['source'].get('modifications','none')+'; '+note,
                      'parent_record_id':original['id'],'parent_record_hash':fingerprint(original),
                      'training_allowed':True}
        for direction, source_texts, translations in (('zh-CN',[en,changed_en],[zh,changed_zh]),
                                                       ('en',[zh,changed_zh],[en,changed_en])):
            inputs = [{'text':text,'target_lang':direction,'context':'','glossary':{}} for text in source_texts]
            outputs = [{'translation':text} for text in translations]
            pair_id = f'public-fact-{number:04d}-{direction}'
            packet = {'id':pair_id,'group_id':original['group_id'],'category':original['category'],
                      'fact_axis':axis,'inputs':inputs,'outputs':outputs,'source':provenance,
                      'review':{'status':'approved','reviewer':'Codex (user-authorized AI acceptance)',
                                'at':now(),'content_hash':fingerprint({'inputs':inputs,'outputs':outputs}),
                                'notes':note,'cross_pair_check':'Both off-diagonal answers change the explicitly identified source fact; not model-generated negatives.'}}
            for i in range(2):
                positive = {'id':pair_id+f'-{i}','group_id':original['group_id'],'category':original['category'],
                            'input':inputs[i],'output':outputs[i],'source':provenance,
                            'review':{'status':'approved','reviewer':'Codex (user-authorized AI acceptance)',
                                      'at':now(),'content_hash':fingerprint({'input':inputs[i],'output':outputs[i]}),
                                      'notes':note}}
                validate_record(positive,True,True)
                positives.append(positive)
                for j in range(2):
                    combination = {**positive,'output':outputs[j]}
                    encoded = encode_example(tokenizer,combination,1024)
                    token_lengths.append(len(encoded['input_ids']))
            packets.append(packet)
    if not packets:
        raise ValueError('All fact families excluded')
    write_jsonl(DEST/'pairs.jsonl',packets)
    write_jsonl(DEST/'positives.jsonl',positives)
    report = {'at':now(),'parent_training_hash':PUBLIC_TRAIN_HASH,'explicitly_reviewed_parent_families':len(VARIANTS),
              'retained_families':len({packet['group_id'] for packet in packets}),'pair_matrices':len(packets),
              'positive_records':len(positives),'pairs_hash':fingerprint(packets),'positives_hash':fingerprint(positives),
              'fact_axes':dict(Counter(packet['fact_axis'] for packet in packets)),
              'directions':dict(Counter(packet['inputs'][0]['target_lang'] for packet in packets)),
              'all_four_combination_max_tokens':max(token_lengths),'truncated_combinations':0,
              'excluded':excluded,'scanned_heldout_source_snapshots':snapshots,
              'training_allowed':True,'training_started':False,'release_approved':False,
              'scope':'Train-only reviewed factual derivative pilot. Keep parent family across replay/variants/directions. No DEV targets used. Conservative source retrieval is not exhaustive semantic decontamination. Not a model quality result.'}
    write_json(DEST/'manifest.json',report)
    print({key:value for key,value in report.items() if key not in ('scanned_heldout_source_snapshots','fact_axes')},flush=True)


if __name__ == '__main__':
    main()
