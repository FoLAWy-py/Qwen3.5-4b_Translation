"""Root explicitly expands inspected pairs only after reading actual v5 output."""
import hashlib
from collections import Counter
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

READ_PAIRS = range(1, 101)
RAW_SHA = '4a4215b29b811beb7843b34aa2fa8a8ea8b40824688b20287f1692ec512cda42'
ISSUES = {
    '001-en': ('major', '花盆底下仍译成in the pot，钥匙所在位置改变。'),
    '002-zh-CN': ('minor', 'lower放下仍泛化成调到一半，下降方向未明确。'),
    '002-en': ('minor', '百叶帘泛化为shade遮阳帘，帘子类型精度降低。'),
    '003-zh-CN': ('minor', '我还是有你的延长线搭配生硬，仍持有物品的主意保留。'),
    '005-zh-CN': ('minor', 'draining board沥水板泛化为沥水架，器具类型偏差。'),
    '007-en': ('major', '拉上背包译为pull the backpack up，关闭动作变为提起。'),
    '011-zh-CN': ('major', 'tap水龙头误译成丝锥，维修对象改变。'),
    '013-zh-CN': ('minor', 'ajar虚掩泛化成开着，开门程度未保留。'),
    '015-zh-CN': ('major', 'bin liners垃圾袋误译成纸箱。'),
    '015-en': ('minor', 'The garbage bags are out不够自然，用完的库存条件表述不明确。'),
    '017-zh-CN': ('major', 'unless提前取消的例外条件变为取消之前续费，条件关系改变。'),
    '018-zh-CN': ('major', '玻璃在保护膜下面变为划痕不在玻璃下面，位置关系改变。'),
    '020-en': ('minor', 'the smoke使用了特定烟雾的定冠词，而原文只说未闻到烟味；主要未闻到信息保留。'),
    '021-zh-CN': ('minor', 'reclining后仰调斜功能泛化成可调节，功能类型不够明确。'),
    '022-zh-CN': ('major', '车上购票更贵反转为柜台购票更贵，并无依据添加飞机。'),
    '022-en': ('major', '上车未说明bus类型，译文新增公交车辆限定。'),
    '025-zh-CN': ('major', 'drop box归还钥匙箱译成取车箱，用途改变。'),
    '028-zh-CN': ('major', 'last connection译为最后一班连接，关键接续交通服务未译清；不要求列车的参考额外限定。'),
    '031-zh-CN': ('minor', '瀑布之后及滑腻的表达生硬，经过瀑布后路段位置不够明确。'),
    '032-zh-CN': ('major', 'refundable押金可退款的条件漏译。'),
    '033-zh-CN': ('major', 'tap card乘车刷卡译成按一下卡，动作不正确。'),
    '033-en': ('major', '下车未明确train类型，译文新增列车限定。'),
    '035-en': ('major', '饮用水泛化为water，可饮用条件漏译。'),
    '036-zh-CN': ('minor', 'may可能推迟变为可以推迟，可能性与许可有偏差。'),
    '037-zh-CN': ('major', 'shuttle接驳交通服务误译成航班。'),
    '037-en': ('major', '接驳车未指定bus，译文无依据限定为巴士。'),
    '040-zh-CN': ('major', 'return voucher返程凭券译成报销单，用途改变。'),
    '040-en': ('minor', '返程凭券泛化成return tickets，凭券与车票类型未区分。'),
    '042-zh-CN': ('major', '羊肉馅中的羊肉物种遗漏，pastry无依据限定甜点，diced切丁也泛化成切碎。'),
    '043-zh-CN': ('major', 'zest柠檬外皮屑误译成果肉，刮取食材改变。'),
    '043-en': ('major', '刮取外层皮屑变为peel剥外层，未保留取得皮屑的动作。'),
    '044-zh-CN': ('minor', 'lentils泛化成豆子，具体食材类型未保留。'),
    '044-en': ('minor', '扁豆泛化为green beans；中文并未明确lentils，不强制参考词，保留参考歧义说明。'),
    '046-zh-CN': ('minor', 'loaf整条面包译成面包片，切片对象阶段表述偏差，未完全分离约束仍在。'),
    '048-zh-CN': ('minor', 'piping hot滚烫降为热，温度程度丢失。'),
    '048-en': ('minor', '滚烫泛化成hot，温度程度丢失。'),
    '049-zh-CN': ('major', 'capers刺山柑花蕾误译成沙葱，食材改变。'),
    '049-en': ('major', '刺山柑花蕾变为hibiscus buds木槿花蕾，青胡椒粒也变为green pepper flakes，食材改变。'),
    '047-zh-CN': ('major', 'before变稠之前的时序条件变为防止变稠的目的关系。'),
    '051-zh-CN': ('major', 'garnish装饰配料无依据限定为香料，配料种类和角色改变。'),
    '051-en': ('major', '分着喝这碗汤变为each have a bowl各自一碗，份量和分享方式改变。'),
    '052-zh-CN': ('major', 'bay leaf月桂叶误译香芹叶。'),
    '053-zh-CN': ('major', 'stock高汤泛化成这道菜，食物对象改变。'),
    '054-en': ('minor', '滤掉籽译为strain the seeds，缺少out，滤除方向不够明确。'),
    '055-zh-CN': ('minor', 'casserole dish炖菜烤盘表述为炖菜盘，术语不自然；塑料盖禁入烤箱正确。'),
    '056-zh-CN': ('minor', '瓶子是普通水主谓搭配不自然，瓶中水的含气对比仍在。'),
    '058-zh-CN': ('minor', '价格是每人两个饺子搭配不自然，包含份量的角色不明确；每人/每桌数量区别正确。'),
    '060-zh-CN': ('minor', 'drinking chocolate字面译为饮用巧克力，冲饮用品名称不自然。'),
    '060-en': ('minor', '冲饮巧克力粉未指定速溶，instant添加了速溶类型限定。'),
    '061-zh-CN': ('major', '正的分子条件漏译，分数单调结论被扩大至不成立的情况。'),
    '061-en': ('minor', '越大越小的比较结构变成逗号拼接谓语，表达生硬；正分子与固定条件保留。'),
    '064-en': ('major', '电阻元件误译resistance物理量，耗散能量的主语类型改变。'),
    '065-zh-CN': ('minor', '留下溶解的盐未明确原先溶解，蒸发后状态表述易混淆。'),
    '068-zh-CN': ('minor', 'silt泛化成泥沙，沉积物细分类型未保留。'),
    '069-zh-CN': ('minor', 'hindsight事后视角译成事后回顾，视角与回顾行为存在偏差。'),
    '074-zh-CN': ('major', 'quadruples变为四倍误译增加四倍，数值关系变成五倍。'),
    '082-zh-CN': ('minor', 'under tension拉伸受力译成拉伸状态，力学受力信息不够明确。'),
    '082-en': ('minor', '拉伸受力译成stretched state，受力信息未明确。'),
    '084-zh-CN': ('major', '火柴划擦变成泛泛点火，具体对象和动作未保留。'),
    '090-zh-CN': ('major', 'file金属加工锉刀仍误译文件，未使用手工工具语境。'),
    '099-zh-CN': ('minor', 'circulating文件传阅译成流传，文件供同事阅读的流转表述不自然。'),
}

def main():
    path = Path('runs/v5-selected-dev.jsonl')
    raw = read_jsonl(path)
    refs = {row['id']:row for row in read_jsonl('data/prepared/v5-mixed/dev.jsonl')}
    expected = {f'v4-dev-{i:03}-{target}' for i in READ_PAIRS for target in ('en','zh-CN')}
    if expected == set(refs) and RAW_SHA is None:
        raise ValueError('Full completed review requires frozen raw SHA')
    if RAW_SHA is not None and hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Read generation changed')
    indexed = {row['id']:row for row in raw}
    if len(indexed) != len(raw) or not expected.issubset(indexed):
        raise ValueError('Missing inspected output or duplicate ID')
    decisions = []
    for row in raw:
        if row['id'] not in expected:
            continue
        ref = refs[row['id']]
        if row['input'] != ref['input'] or row['reference'] != ref['output']:
            raise ValueError('Source changed')
        verdict, note = ISSUES.get(row['id'].removeprefix('v4-dev-'), ('pass', '逐条核对当前原文与译文，接受等义表达，未发现需修订问题。'))
        if row['id'] == 'v4-dev-047-en':
            note += ' 中文打蛋器未明确手动工具，mixer允许电动搅打器读法，不强制套用参考whisk。该来源不作为严格器具术语试题。'
        decisions.append({'id':row['id'], 'group_id':ref['group_id'], 'category':row['category'],
            'target_lang':row['input']['target_lang'], 'verdict':verdict, 'note':note, 'reviewer':'Codex', 'at':now(),
            'format_valid':'prediction' in row, 'ended':row['ended'], 'language_correct':True,
            'output_hash':fingerprint({k:row[k] for k in ('input','raw','reference')}), 'purpose':'Known-development screening, not release'})
    summary = {'at':now(), 'reviewed':len(decisions), 'expected_total':len(refs), 'complete':len(decisions)==len(refs),
        'verdicts':dict(Counter(row['verdict'] for row in decisions)), 'unread_rows_are_not_approved':True}
    write_jsonl('runs/v5-selected-dev-semantic.jsonl', decisions)
    write_json('runs/v5-selected-dev-semantic.summary.json', summary)
    print(summary)

if __name__ == '__main__':
    main()
