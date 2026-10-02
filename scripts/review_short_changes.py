"""Root's source-grounded audit of all changed compiled translations."""
import hashlib
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

DECISIONS = {
    'v8-multi-026-en': ('pass','pass','省略of material后，质量损失与后句which substance仍共同明确物质损失；称重前后、不能确定种类都保留，未出现新的语义问题。'),
    'v8-multi-026-zh-CN': ('pass','pass','材料损失改为材料流失，仍指样品物质离开；种类不能确定、前后称重和较低质量均保留。'),
    'v8-multi-002-en': ('pass','pass','take/bring the receipt home均可表达带回家；邻居已取包裹和不必去取件点保留。'),
    'v8-multi-017-zh-CN': ('minor','minor','干燥的锅改干锅改善搭配；两版发热时均不如趁热自然，冷却后研磨与不趁热加水的时序保持，未新增严重错译。'),
    'v8-multi-018-zh-CN': ('major','minor','普通版指定盖子，原文只说轻盖未指定覆盖物；编译版去掉该物件限定，但盖上松一点和面团休息仍需润色。额外静置、擀开之前、备馅时防干保持。'),
    'v8-multi-010-zh-CN': ('pass','major','原文until明确停止接待入场的截止时点。普通版正确保留停止，编译版改为开放时间是闭馆前一小时，丢失截止含义且可误作开始时点；后句预约不足不能补救第一句。'),
}


def main():
    path = Path('runs/v7-short-runtime-comparison.changed.jsonl')
    output = Path('runs/v7-short-compiled-semantic-changes.jsonl')
    if output.exists():
        raise ValueError('Preserve completed source-grounded review')
    if hashlib.sha256(path.read_bytes()).hexdigest()!='91a4976c545d40ef6ceda8cac4b76b901a826b193ce7e58468d2b3305bab632f':
        raise ValueError('Individually read changes differ')
    rows = read_jsonl(path)
    if len(rows)!=18 or {r['id'] for r in rows}!=set(DECISIONS):
        raise ValueError('Complete changed-output scope required')
    signatures = {}
    results = []
    for row in rows:
        identity = fingerprint({k:row[k] for k in ('input','eager_raw','compiled_raw','eager_prediction','compiled_prediction')})
        if row['id'] in signatures and signatures[row['id']]!=identity:
            raise ValueError('Repeated rounds have distinct, unreviewed translations')
        signatures[row['id']] = identity
        before,after,note = DECISIONS[row['id']]
        results.append({'id':row['id'],'round':row['round'],'content_hash':identity,
            'eager_verdict':before,'compiled_verdict':after,'note':note,
            'reviewer':'Codex (user-authorized AI acceptance)','at':now()})
    write_jsonl(output,results)
    write_json('runs/v7-short-compiled-semantic-review.json',{'at':now(),
        'reviewed_changed_measurements':18,'unique_changed_inputs':6,
        'before_unique':dict(Counter(v[0] for v in DECISIONS.values())),
        'after_unique':dict(Counter(v[1] for v in DECISIONS.values())),
        'new_major_ids':['v8-multi-010-zh-CN'],
        'improved_major_ids':['v8-multi-018-zh-CN'],
        'runtime_deployment_approved':False,'release_approved':False,
        'scope':'Changed outputs only, known performance inputs; not full model accuracy, release testing, or new independent semantic development. Numeric speed gate passed but deadline mistranslation introduced; full runtime quality acceptance absent.'})
    print('All18 changed measurements/6 unique translations read; numeric speed success is not deployment approval.',flush=True)


if __name__=='__main__':
    main()
