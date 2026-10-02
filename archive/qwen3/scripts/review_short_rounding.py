"""Explicit root reading of every changed precision-rounding benchmark output."""
from collections import Counter
from pathlib import Path
from archive.qwen3.runtime import parse_translation
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

# Exact translations personally read in all three rounds; no unread defaults.
READ = {
 'v8-multi-027-zh-CN': ('调查记录了受访者是否选择了任一选项。它不记录他们对所选选项的偏好程度。',
   '调查记录了受访者是否选择了其中任何一个选项。它不记录他们对所选选项的偏好程度。',
   'pass','pass','任一选项/其中任何一个选项等义；仅记录是否选择、不记录偏好强度的否定范围保持。'),
 'v8-multi-018-zh-CN': ('这种面团还需要再休息一下，才能擀开。盖上松松的盖子，这样在我们准备馅料时，表面不会干掉。',
   '这种面团还需要再休息一下，才能擀开。盖上松一点，这样在我们准备馅料时，表面不会干掉。',
   'major','minor','eager额外指定盖子，来源仅要求松松地覆盖；新编译版去掉物件限定，但面团休息及盖上松一点搭配仍需润色。再次静置先于擀开、备馅期间防表面干燥均保持。'),
 'v8-multi-036-en': ("We'll put this aside for now. Once the numbers are verified, we can go back to it.",
   "We'll put this aside for now. Once the numbers are verified, we can revisit it.",
   'pass','pass','revisit与go back to it均保留搁置后再处理；核对数字之后与can可能性保持，语境不指物理冷藏。'),
 'v8-multi-009-zh-CN': ('小码头旁边就是仓库，船从那里出发。在线购买的票在那有效，所以不需要先兑换。',
   '小码头旁边就是仓库，船从那里出发。在线购买的票在那有效，所以你不需要先去兑换。',
   'pass','pass','显式补出原文you等义，先去兑换与先兑换保持不需要该动作；小码头自然表达此处smaller pier，没有新增码头或改发船地点。网上票在该地有效的原因保持。'),
}


def main():
    output = Path('runs/v7-short-eager-rounding-semantic-changes.jsonl')
    if output.exists():
        raise ValueError('Preserve root review evidence')
    path = Path('runs/v7-short-eager-rounding-comparison.changed.jsonl')
    rows = read_jsonl(path)
    frozen = {r['id']:r['input'] for r in read_jsonl('data/prepared/short-benchmark-v1/inputs.jsonl')}
    if len(rows)!=12 or {(r['round'],r['id']) for r in rows}!={(i,k) for i in (1,2,3) for k in READ}:
        raise ValueError('Changed scope differs from all12 individually read outputs')
    decisions = []
    for row in rows:
        before,after,bv,av,note = READ[row['id']]
        if (row['input']!=frozen[row['id']] or row['eager_prediction']!={'translation':before}
                or row['compiled_prediction']!={'translation':after}
                or parse_translation(row['eager_raw'])!=row['eager_prediction']
                or parse_translation(row['compiled_raw'])!=row['compiled_prediction']):
            raise ValueError('Actual source/output differs from personally read content')
        decisions.append({'id':row['id'],'round':row['round'],
            'content_hash':fingerprint({k:row[k] for k in ('input','eager_raw','compiled_raw','eager_prediction','compiled_prediction')}),
            'eager_verdict':bv,'compiled_verdict':av,'note':note,
            'reviewer':'Codex (user-authorized AI acceptance)','at':now()})
    eager = {(r['round'],r['id']):r for r in read_jsonl('runs/v7-short-eager.jsonl')}
    compiled = {(r['round'],r['id']):r for r in read_jsonl('runs/v7-short-compiled-eager-rounding.jsonl')}
    deadline = 'v8-multi-010-zh-CN'
    if len(eager)!=81 or set(eager)!=set(compiled):
        raise ValueError('Complete corresponding benchmark required')
    for i in (1,2,3):
        a,b = eager[i,deadline],compiled[i,deadline]
        if a['raw']!=b['raw'] or '停止接待' not in b['prediction']['translation'] or not b['ended']:
            raise ValueError('Known deadline regression persists')
    write_jsonl(output,decisions)
    report = {'at':now(),'changes_hash':fingerprint(rows),'reviewed_changed_measurements':12,
        'unique_changed_inputs':4,'unchanged_raw_measurements':69,
        'before_unique':dict(Counter(r[2] for r in READ.values())),
        'after_unique':dict(Counter(r[3] for r in READ.values())),
        'new_major_ids':[],'improved_major_ids':['v8-multi-018-zh-CN'],
        'known_deadline_regression_fixed_all_three_rounds':True,
        'known_short_changed_output_regression_passed':True,
        'runtime_deployment_approved':False,'release_approved':False,
        'scope':'All12 changed outputs read against their actual frozen sources, repeated signatures explicitly bound.69 unchanged outputs provide consistency only. No new independent model accuracy or600-row release evidence; default runtime unchanged.'}
    write_json('runs/v7-short-eager-rounding-semantic-review.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
