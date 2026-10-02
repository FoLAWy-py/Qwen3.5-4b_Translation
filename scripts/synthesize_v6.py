"""Create pending original source pairs with32B; approval remains Codex's job."""
import concurrent.futures
import argparse
import json
from pathlib import Path
import httpx
from data.v6_topics import TOPICS, HARD_GROUPS
from witrans_tools.common import TEACHER_ID, fingerprint, now, read_jsonl, secret, write_json, write_jsonl
from witrans_tools.label import ENDPOINT
from witrans_tools.data import validate_record

SYSTEM = '''Create original fictional bilingual training source pairs. Do not copy published text. Return a JSON object with items, a list matching the supplied cases in order. Each item has exactly case_id, english, chinese, context. Chinese must faithfully translate all English content without adding facts. Prefer short natural dialogue or instructions. For non-hard cases use two connected sentences totaling20-40 English words and empty context. For hard cases use a single ambiguous English sentence (8-15 words), and separate concise English context that resolves the listed word's sense. Do not spell out the sense in the ambiguous sentence. The Chinese translation must use the resolved sense but not translate background context as added facts. Use no real personal data, current events, unsafe instructions or specialized medical advice. Each case must express a different situation, not mere name/number variants. These outputs are unreviewed candidates, never automatically accepted.'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('v6','v10'),default='v6')
    args = parser.parse_args()
    version = args.profile
    topics, hard_groups, system = TOPICS, HARD_GROUPS, SYSTEM
    if version == 'v10':
        from data.v10_topics import TOPICS as topics
        hard_groups = [f'v10-hard-{i+1:02}' for i in range(len(topics['hard']))]
        system = '''Create original fictional bilingual training paragraphs, never copied published text. Return a JSON object with items, a list matching supplied cases in order. Each item has exactly case_id, english, chinese, context. English should have2-3 connected sentences totaling40-65 words. Chinese must faithfully translate all English content, preserving actors, negation, conditions, time, quantities, technical meaning, and degree of certainty. Use no real personal data or current events. Non-hard context is empty. For hard cases write natural actual dialogue using the stated idiom and a short separate context resolving it. Do not put a parenthetical explanation or dictionary definition in the source text. Do not translate context as extra facts. Vary sentence structure, speaker roles and voice across cases; do not reuse the same instruction template with names or numbers changed. Do not answer questions in the Chinese translation. Any literal formula or identifier in English must remain intact in Chinese. These are pending training candidates; Codex individual review remains necessary.'''
    destination = Path(f'data/prepared/{version}-source')
    if destination.exists():
        raise ValueError('Pending pool already frozen')
    cases = []
    for category, category_topics in topics.items():
        for i, topic in enumerate(category_topics):
            cases.append({'case_id': f'{version}-{category}-{i+1:02}', 'category': category, 'topic': topic,
                'group_id': hard_groups[i] if category == 'hard' else f'{version}-{category}-{i+1:02}'})
    batches = [cases[i:i+4] for i in range(0,len(cases),4)]
    parameters = {'temperature': .7, 'max_tokens': 3072, 'response_format': {'type':'json_object'}, 'chat_template_kwargs': {'enable_thinking':False}}
    plan = {'system':system,'cases':cases,'parameters':parameters,'model':TEACHER_ID}
    plan_path = Path(f'runs/{version}-source-plan.json')
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf-8'))
        if previous['plan_hash'] != fingerprint(plan):
            raise ValueError('Immutable synthesis plan changed')
    else:
        write_json(plan_path, {'at':now(), 'plan_hash':fingerprint(plan), 'plan':plan,
            'scope':('40 source-pair candidates only; teacher-authored original fictional inputs, not held-out tests; same hard sense-family joins v5 train groups; Codex review mandatory' if version=='v6' else '100 individually specified source-pair candidates; original fictional training only. Candidate semantic families and labels pending individual Codex review, not100 independently verified groups or release test.')})
    with httpx.Client(headers={'Authorization': f"Bearer {secret('DEEP_INFRA_APIKEY')}"}, timeout=120) as client:
        def request(index):
            path = Path(f'data/generated/{version}-source/batch-{index+1:02}.json')
            if path.exists():
                return json.loads(path.read_text(encoding='utf-8'))
            batch = batches[index]
            response = client.post(ENDPOINT, json={'model':TEACHER_ID,'messages':[{'role':'system','content':system},
                {'role':'user','content':json.dumps(batch,ensure_ascii=False)}], **parameters})
            if response.status_code != 200:
                raise RuntimeError(f'DeepInfra HTTP {response.status_code}')
            payload = response.json()
            choice = payload['choices'][0]
            record = {'at':now(),'plan_hash':fingerprint(plan), 'batch':batch,'raw':choice['message']['content'],
                'finish_reason':choice.get('finish_reason'),'usage':payload.get('usage',{}),'response_model':payload.get('model')}
            write_json(path,record)
            return record
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(request,range(len(batches))))
    old = {r['input']['text'].strip().casefold() for p in Path('data/prepared').glob('**/*.jsonl') for r in read_jsonl(p) if 'input' in r}
    rows, seen, exclusions = [], {}, []
    for index, response in enumerate(responses):
        if response['plan_hash'] != fingerprint(plan) or response['finish_reason'] != 'stop':
            raise ValueError('Incomplete or mismatched provider response retained for inspection')
        items = json.loads(response['raw'])['items']
        batch = batches[index]
        if len(items) != len(batch):
            raise ValueError('Incorrect source count')
        for case, item in zip(batch,items):
            if set(item) != {'case_id','english','chinese','context'} or item['case_id'] != case['case_id'] or any(not isinstance(item[k],str) for k in item):
                raise ValueError('Unexpected source schema')
            for lang, source, translation in (('zh-CN',item['english'],item['chinese']),('en',item['chinese'],item['english'])):
                key = source.strip().casefold()
                if key in old or (key in seen and seen[key] != case['group_id']):
                    exclusions.append({'case_id':case['case_id'],'group_id':case['group_id'],'reason':'Exact source duplicates old corpus or another source family'})
                seen[key] = case['group_id']
                row = {'id':f"{case['case_id']}-{lang}",'group_id':case['group_id'],'category':case['category'],
                    'input':{'text':source,'target_lang':lang,'context':item['context'],'glossary':{}},'output':{'translation':translation},
                    'source':{'name':f'Original fictional32B-generated {version} project sources','license':'Original model-generated project data; no copied source requested',
                        'training_allowed':True,'external_labeling_allowed':True},'review':{'status':'pending'},
                    'label_metadata':{'provider':'DeepInfra','requested_model':TEACHER_ID,'response_model':response['response_model'],
                        'raw_batch':f'data/generated/{version}-source/batch-{index+1:02}.json','construction':'Teacher authored both languages; no independent reference, Codex must check each direction'}}
                validate_record(row,True)
                rows.append(row)
    excluded = {r['group_id'] for r in exclusions}
    rows = [r for r in rows if r['group_id'] not in excluded]
    write_jsonl(destination/'candidates.jsonl',rows)
    write_json(destination/'manifest.json',{'at':now(),'plan_hash':fingerprint(plan),'candidate_rows':len(rows),'source_groups':len({r['group_id'] for r in rows}),
        'candidate_hash':fingerprint(rows),'exclusions':exclusions,'quality_status':'Pending source, translation, near-duplicate and provenance review; not training-ready',
        'usage':{key:sum(r['usage'].get(key,0) for r in responses) for key in ('prompt_tokens','completion_tokens')}})
    print({'pending_rows':len(rows),'source_groups':len({r['group_id'] for r in rows}),'excluded_groups':len(excluded)})

if __name__ == '__main__':
    main()
