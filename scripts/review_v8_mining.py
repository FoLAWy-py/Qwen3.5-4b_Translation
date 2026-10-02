"""Persist root's explicitly read training-pool outputs; never approve unread rows."""
import hashlib
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

READ_PAIRS = range(1, 41)
RAW_SHA = '6b86acddda4d35a6e811eb592dea872af80b208c85bb94e87930678580236607'
ISSUES = {
    '001-zh-CN': ('major', 'neither can get home before you arrive变为我们都不在，返回与到达的时间条件丢失；only if限制也未明确。'),
    '001-en': ('minor', 'only if只有才泛化为if then条件，显式限制强度不够明确；主要条件分支保留，不作为确定错误偏好负例。'),
    '003-en': ('minor', '保留日历原定时间的安排变为日历会保留的未来状态表达，言语行为不够明确。'),
    '004-zh-CN': ('minor', '手指远离缝隙变为手不伸进去，距离约束减弱但避免夹手的主意仍在，不当反转负例。'),
    '004-en': ('major', '我拧螺丝变成you tighten the screw，操作人角色改变。'),
    '005-zh-CN': ('minor', 'bottom drawer最下面的抽屉泛化为下面抽屉，damp无依据添加有点程度。'),
    '007-zh-CN': ('minor', '我重命名了文档为的词序生硬；指定术语静谧信标已保留。'),
    '008-zh-CN': ('minor', '检查一下没人需要它之后的句式不自然，没人需要的条件仍在。'),
    '009-zh-CN': ('minor', 'smaller比较级变成小码头，船从那里指代码头不够明确，未认定为明确地点反转。'),
    '010-zh-CN': ('major', 'until关门前一小时停止入场变成在关门前一小时开放参观，入场时间边界改变。'),
    '011-zh-CN': ('major', '保持路标左侧直到分岔变成只在分岔处；uphill通向坡上无依据限定通往山顶。'),
    '011-en': ('minor', '通向山上的窄路仅说go up the narrow one，山的地形对象未明确；不强制英语参考措辞。'),
    '012-zh-CN': ('major', 'extra night额外多住一晚变为保留房间过夜，延长住宿的请求丢失。'),
    '012-en': ('major', 'even if早餐可能更贵变为even though早餐已经更贵，假设条件变成已发生事实。'),
    '013-zh-CN': ('major', 'departures新增列车限定，核对日期变为查看当天时刻表，接续班次也只字面译连接。'),
    '013-en': ('major', '接续班次未说明航班，connecting flight新增航空方式。'),
    '014-zh-CN': ('major', 'guided walk导览步行活动变为导览步道设施，需预约对象改变。'),
    '014-en': ('minor', '日票泛化为ticket，单日有效的票种未明确。'),
    '015-zh-CN': ('major', 'following route沿路线行进无依据限定为行驶，新增车辆出行方式。'),
    '016-zh-CN': ('minor', '我们就下车在主广场的语序生硬；替代下车地点和条件仍在。'),
    '017-zh-CN': ('minor', 'while hot趁热译成发热时，热状态与产生热量的表述不自然。'),
    '017-en': ('minor', '炒香的气味完成标准省略，只说toast；冷却再研磨与不趁热加水保留。'),
    '018-zh-CN': ('minor', '面团休息和盖上松松的盖子搭配生硬，盖子类型原文未指定；擀开前静置顺序保留。'),
    '018-en': ('minor', '还需要静置一次未明确再一次，another rest的重复阶段信息弱化。'),
    '019-en': ('minor', '芥末未明确辣根原料，horseradish限定原料；原中文名称口语混用有歧义，不作为确定负例。'),
    '020-zh-CN': ('minor', '用铲子轻轻提起的搭配不自然，铲起米饭的主意与避免刮涂层仍在。'),
    '021-zh-CN': ('minor', '如果你想让它更久缺少持续等谓语，延长馅料使用的意图表达生硬。'),
    '022-zh-CN': ('minor', '松散混合物不自然，烹饪中调稀的黏稠度调整不够明确。'),
    '024-zh-CN': ('major', 'tart馅饼改成蛋糕；移动托盘之前可能损坏底部变成未冷却切开必然损坏，动作因果与程度改变。'),
    '031-zh-CN': ('major', '指标measure译成措施，报告定义的度量对象变成行动措施；指定术语虽正确，概念类别改变。'),
    '031-en': ('major', '如果可能则避免缩写新增if possible许可条件，原来在可能混淆时不得缩写的要求被弱化。'),
    '034-zh-CN': ('major', '等大家到齐新增再公布的承诺；原文只约定等待，语境用于消歧不应补写揭晓动作。'),
    '035-zh-CN': ('major', '责任在对方的语境已明确，ball in your court仍直译球在你这边，决策责任含义丢失。'),
    '036-zh-CN': ('minor', 'can return可以再处理变成再回来讨论的确定表述，可能性及处理范围不够准确，不作为确定负例。'),
    '037-zh-CN': ('minor', 'cold shoulder用冷眼相对表达生硬，主动忽视不够明确；两个问题的数量省略，未当作确定关系反转负例。'),
    '040-zh-CN': ('major', '保留意图的语境已明确，cards close to chest仍译成把牌藏在胸口，新增实际藏牌动作。'),
}

def main():
    path = Path('runs/v8-multisentence-v5-mining.jsonl')
    rows = read_jsonl(path)
    refs = {r['id']:r for r in read_jsonl('data/prepared/v8-multisentence/train.jsonl')}
    expected = {f'v8-multi-{i:03}-{target}' for i in READ_PAIRS for target in ('en','zh-CN')}
    if expected == set(refs) and RAW_SHA is None:
        raise ValueError('Complete review must bind finished raw SHA')
    if RAW_SHA is not None and hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Read output changed')
    indexed = {r['id']:r for r in rows}
    if len(indexed) != len(rows) or not expected.issubset(indexed):
        raise ValueError('Missing inspected output or duplicate ID')
    decisions = []
    for row in rows:
        if row['id'] not in expected:
            continue
        ref = refs[row['id']]
        if row['input'] != ref['input'] or row['reference'] != ref['output']:
            raise ValueError('Approved source changed')
        verdict, note = ISSUES.get(row['id'].removeprefix('v8-multi-'), ('pass', '逐条按实际原文和术语核对，接受等义措辞，未发现需修订问题。'))
        if row['id'] == 'v8-multi-010-en':
            note += ' 第一分句已经明确停止入场，第二分句even with reservation按完整两句读，不构造虚假的进入选择偏好。'
        if row['id'] == 'v8-multi-016-en':
            note += ' 中文停车允许park读法，不因原始英文stop强制否定另一方向的实际中文译法。'
        decisions.append({'id':row['id'], 'group_id':ref['group_id'], 'category':ref['category'],
            'target_lang':row['input']['target_lang'], 'verdict':verdict, 'note':note, 'reviewer':'Codex', 'at':now(),
            'format_valid':'prediction' in row, 'ended':row['ended'], 'language_correct':True,
            'output_hash':fingerprint({k:row[k] for k in ('input','raw','reference')}), 'purpose':'Training-pool mining; no accuracy or release claim'})
    summary = {'at':now(), 'reviewed':len(decisions), 'expected_total':len(refs), 'complete':len(decisions)==len(refs),
        'verdicts':dict(Counter(d['verdict'] for d in decisions)), 'unread_rows_are_not_approved':True,
        'scope':'Training-pool mining yield only. Minor or equivalent wording never harvested as definite semantic negative.'}
    write_jsonl('runs/v8-mining-semantic.jsonl', decisions)
    write_json('runs/v8-mining-semantic.summary.json', summary)
    print(summary)

if __name__ == '__main__':
    main()
