"""Root AI's source-grounded review of every actual fixed24 timing translation."""
from pathlib import Path
from witrans_tools.common import read_jsonl, write_jsonl, write_json, now
from witrans_tools.rag import digest
from witrans_tools.independent_quality import counts

JUDGMENTS={
    "v8-multi-001-zh-CN":("major",True,"neither of us 的我们两人错为你们俩，改变取钥匙条件中的角色；与原无RAG实测输出相同。"),
    "v8-multi-018-zh-CN":("major",False,"普通动作修饰语loosely未译，松盖的操作要求没有完整中文表达；与原无RAG实测输出相同。"),
    "v8-multi-011-zh-CN":("major",False,"普通路线对象trail marker未译；保持左侧的对象表达也不明确，关键导航句不完整；与原无RAG实测输出相同。"),
    "v8-multi-026-zh-CN":("minor",True,"物质损失且不能识别是哪种物质的含义保留，但哪一种物质离开了措辞不自然；与原无RAG相同。"),
    "v8-multi-019-en":("pass",True,"mustard为芥末，调味汁有而烤蔬菜无，单独盛放供各人选择的条件完整；sauce与dressing在给定中文中均合理。"),
    "v8-multi-019-zh-CN":("pass",True,"酱汁含芥末、烤蔬菜不含，酱汁分别提供让每人决定是否添加，在句内保持独立提供含义。"),
}


def main():
    root=Path("runs/takeover-20261002/rag/performance")
    formal=read_jsonl(root/"performance.jsonl")
    warm=read_jsonl(root/"warmup.jsonl")
    if len(formal)!=144 or len(warm)!=48:
        raise ValueError("Both complete24x3 profiles and whole-round warmups required")
    baseline=[r for r in read_jsonl("runs/takeover-20261002/decode-final/performance.jsonl") if r['repeat']==0]
    by_id={r['id']:r for r in baseline}
    sources={r['id']:r for r in read_jsonl('runs/qwen35-v3-diagnostic-recovery1/performance-inputs.jsonl')}
    all_reviews={}
    for profile in ('local_terms','semantic_terms'):
        rows=[r for r in warm if r['profile']==profile]
        if len(rows)!=24 or len({r['id'] for r in rows})!=24:
            raise ValueError("Fixed24 source coverage mismatch")
        reviews=[];base_reviews=[]
        for row in rows:
            old=by_id[row['id']]
            if row['original_input']!=old['input']:
                raise ValueError("Baseline source identity mismatch")
            # All24 source/context/output pairs have been read; explicit exceptions above.
            verdict,language,note=JUDGMENTS.get(row['id'],('pass',True,
                '原文与实际译文逐条读取：人物、否定、条件、数量、动作顺序和语境完整，合理等义表达可接受。'))
            if row['raw']!=old['raw'] and row['id']!='v8-multi-019-en':
                raise ValueError("Additional baseline change needs root adjudication")
            if any(r['raw']!=row['raw'] for r in formal if r['profile']==profile and r['id']==row['id']):
                raise ValueError("Measured-round changed output needs additional root reading")
            review={'id':row['id'],'group_id':sources[row['id']]['group_id'], 'verdict':verdict,
                'language_correct':language,'note':note,'json_valid':True,'ended':row['ended'],
                'source_input_hash':digest(row['original_input']),'output_hash':digest(row['raw']),
                'profile':profile,'reviewer':'Codex AI','human_acceptance':False}
            reviews.append(review)
            base_reviews.append({**review,'output_hash':digest(old['raw']),'profile':'frozen_no_RAG'})
        write_jsonl(root/(profile+'-semantic.jsonl'),reviews)
        all_reviews[profile]={'counts':counts(reviews),'baseline_counts':counts(base_reviews),
            'new_major':0,'new_language_errors':0,'review_hash':digest(reviews)}
    write_json(root/'semantic-summary.json',{'at':now(),'profiles':all_reviews,
        'method':'Root Codex AI read all24 original source/context/actual warmup translations. All144 formal raw outputs bound to those actually read predictions, same raw checked for binding only; not semantic string-match scoring.',
        'review_complete':True,'human_acceptance':False,
        'finding':'Three pre-existing major errors including two language-direction failures persist. Passing timing thresholds is not a translation-quality or release pass.',
        'independent_test':False,'release_approved':False})
    print(all_reviews)


if __name__=='__main__':main()
