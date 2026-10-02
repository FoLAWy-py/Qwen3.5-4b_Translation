"""Bind the completed Codex AI source/context/output reading to actual new raws.

Historical notes are retained only after rereading the complete 316 actual rows.
Changes below are explicit human-visible judgments, never string-match scoring.
"""
from pathlib import Path
from witrans_tools.common import fingerprint,now,read_jsonl,write_json,write_jsonl
from witrans_tools.independent_quality import counts,group_intervals
from witrans_tools.paired_stats import paired_pass_interval

ROOT=Path('runs/takeover-20261002/development-final')
CHANGES={
 'v4-dev-009-zh-CN':('pass','污渍变淡而未完全消失，状态与否定完整。'),
 'v4-dev-029-en':('pass','上层可乘电梯，地下室只能走楼梯；upper floors 合理表达楼上各层。'),
 'v4-dev-033-zh-CN':('pass','下车再次轻点卡片，tap 刷卡动作及否则最高票价完整。'),
 'v4-dev-033-en':('pass','vehicle 没有擅定为巴士；下车再刷卡与最高票价后果完整。'),
 'v4-dev-036-en':('pass','能见度持续低时可能推迟出游，条件与可能性保留。'),
 'v4-dev-042-zh-CN':('pass','碎羊肉与切丁牛肉的肉类和形态对比完整。'),
 'v4-dev-044-zh-CN':('minor','不加盖小火慢炖明确；之前表达可辨煮制时段，但 until 的终止点表达欠清晰。'),
 'v4-dev-046-en':('pass','面包切片而不完全切透，否定操作正确。'),
 'v4-dev-065-en':('pass','溶剂蒸发后原先溶解的盐留下，过去时合理。'),
 'v4-dev-089-zh-CN':('major','明确金属压制机械语境中的 press 被译为新闻部门，消歧错误。'),
 'public-short-0407-en':('pass','知道有些人看重我的工作，角色和 some 范围完整。'),
 'public-short-0366-zh-CN':('pass','询问我们需要等多久，角色和问句完整。'),
 'public-short-0099-en':('pass','汤姆认为300美元不能买齐所需物品，否定、金额及全部范围保留。'),
 # Source-only reassessment, applied to BOTH old and new outputs below.
 'v4-dev-008-en':('pass','split the bill 是平摊账单的合理惯用表达，配送费范围完整；不因缺 equal 字样判错。'),
 'v4-dev-021-zh-CN':('pass','无消歧语境的 coach 可指客车或铁路车厢；列车解释合理，夜间、后仰座椅和无卧铺完整。'),
 'v4-dev-022-zh-CN':('minor','无交通方式语境的 on board 可合理指船上；亭子未明确售票用途，措辞轻微不准。'),
 'v4-dev-040-zh-CN':('pass','return 未给返程消歧语境，退款凭证属合理解释；工作日有效且公共假日除外完整。'),
 'v4-dev-073-en':('minor','最快及温度范围关系正确，但 narrower 额外引入比较，应为 narrow。'),
}
REASSESS={'v4-dev-008-en','v4-dev-021-zh-CN','v4-dev-022-zh-CN','v4-dev-040-zh-CN','v4-dev-073-en'}

def main():
    report=dict(at=now(),reviewer='Codex AI',human_acceptance=False,
        method='All316 actual source/context/reference/prediction rows read individually; reasonable equivalent expressions accepted. Existing notes confirmed by actual reading, changed rows freshly adjudicated. Raw identity is diagnostic only, never semantic scoring.',
        source_only_reassessments=sorted(REASSESS),independent_test_outputs_read=False,release_approved=False,sets={})
    for label in ('known','public'):
        actual=read_jsonl(ROOT/(label+'.jsonl'))
        historical={r['id']:r for r in read_jsonl(f'runs/qwen35-finetuned-{label}-semantic.jsonl')}
        old={r['id']:r for r in read_jsonl(f'runs/qwen35-finetuned-{label}.jsonl')}
        reviews=[];legacy=[];comparable=[];changed=[]
        for row in actual:
            prior=historical[row['id']]
            verdict,note=CHANGES.get(row['id'],(prior['verdict'],prior['note']))
            review=dict(id=row['id'],group_id=row['group_id'],category=row['category'],target_lang=row['input']['target_lang'],
                verdict=verdict,note=note,language_correct=row['id']!='v4-dev-090-zh-CN',json_valid=row['json_valid'],ended=row['ended'],
                reviewer='Codex AI',human_acceptance=False,review_source='complete_actual_source_context_prediction_reading',
                input_hash=fingerprint(row['input']),output_hash=fingerprint(row['raw']),generation_hash=fingerprint(row),
                has_constraint=bool(row['input'].get('context') or row['input'].get('glossary')),
                constraint_preserved=row['id'] not in ('v4-dev-089-zh-CN','v4-dev-090-zh-CN','v4-dev-091-zh-CN'))
            reviews.append(review)
            legacy.append(dict(review,verdict=prior['verdict'] if row['id'] in REASSESS else verdict))
            oldverdict=CHANGES[row['id']][0] if row['id'] in REASSESS else prior['verdict']
            comparable.append(dict(review,verdict=oldverdict))
            if old[row['id']].get('raw')!=row['raw']:
                changed.append(dict(id=row['id'],previous_verdict=oldverdict,new_verdict=verdict,note=note))
        write_jsonl(ROOT/(label+'-semantic.jsonl'),reviews)
        contexts=[r for r in reviews if r['has_constraint']]
        report['sets'][label]=dict(total=counts(reviews),historical_reported=dict(pass_count=sum(r['verdict']=='pass' for r in historical.values()),minor=sum(r['verdict']=='minor' for r in historical.values()),major=sum(r['verdict'] in ('major','critical') for r in historical.values())),
            legacy_regression_counts=counts(legacy),old_outputs_same_source_only_reassessment=counts(comparable),
            source_group_intervals=group_intervals(reviews),
            paired_old_eager_vs_final=paired_pass_interval(reviews,comparable,iterations=10000,seed=20261002),
            strata={f'{c}/{d}':counts([r for r in reviews if r['category']==c and r['target_lang']==d]) for c in sorted({r['category'] for r in reviews}) for d in ('en','zh-CN')},
            context_rows=len(contexts),constraint_preserved=sum(r['constraint_preserved'] for r in contexts),
            actual_outputs_hash=fingerprint(actual),reviews_hash=fingerprint(reviews),changed_raw_count=len(changed),changed_rows=changed,
            newly_introduced_major=sum(r['verdict'] in ('major','critical') and b['verdict'] not in ('major','critical') for r,b in zip(reviews,comparable)),semantic_review_complete=True)
    report['old_set_regression_review_complete']=True
    report['confirmation_entry_semantic_prerequisite']=True
    write_json(ROOT/'semantic-summary.json',report)

if __name__=='__main__':main()
