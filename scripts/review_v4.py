"""Persist only v4 rows whose source and current output Codex has read."""
import argparse
import hashlib
from pathlib import Path
from collections import Counter
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl

# These ranges are expanded by the reviewer only after reading both directions.
READ_PAIRS = {'selected': range(1, 101), 'start': range(1, 101)}
RAW_SHA = {'selected': '02e542834579c58b7e0d7aa5ef30a1b187ab03a693078765147d45c3eb350c29'}
RAW_SHA['start'] = 'fd88241cb4fdb67d124e420739f80a29890822b582212cb34eee5587bb61266d'
ISSUES = {'selected': {
    '001-en': ('major', '花盆底下译为in the pot，粘贴位置改变。'),
    '002-zh-CN': ('minor', 'lower放下变成调到一半，帘子的下降方向不明确。'),
    '003-zh-CN': ('minor', '我还是有你的延长线表达生硬，但仍持有对方物品的含义保留。'),
    '005-zh-CN': ('minor', 'draining board沥水板泛化为沥水架，器具形状有偏差。'),
    '006-zh-CN': ('minor', 'watch照看译成看，照管请求不够明确；条件保留。'),
    '007-en': ('major', '拉上背包关闭动作译为pull the backpack up提起背包。'),
    '011-zh-CN': ('major', 'tap水龙头译为活塞，gasket另加入活塞限定，维修对象改变。'),
    '013-zh-CN': ('minor', 'ajar虚掩泛化为开着，开门程度细节未保留。'),
    '015-zh-CN': ('major', 'bin liners垃圾袋译为纸箱。'),
    '015-en': ('minor', 'The garbage bags are out表达不自然，out of bin liners用完的信息不够明确。'),
    '018-zh-CN': ('major', 'not on the glass underneath译为不是玻璃下面，玻璃相对保护膜位置与划痕所在表面的关系改变。'),
    '021-zh-CN': ('minor', 'reclining可调斜座椅泛化为可调节座椅，后仰功能不明确。'),
    '022-zh-CN': ('major', '车上购买更贵反转成柜台购买更贵，并无依据加入飞机。'),
    '022-en': ('major', '原文只说上车，新增bus限定交通工具类型。'),
    '025-zh-CN': ('major', '车位译成停机位，钥匙drop box归还箱译成取车箱，对象与用途改变。'),
    '028-zh-CN': ('major', '最后一班连接不是自然的接续交通服务表述，关键换乘对象未译清；按英文原文审核，不要求参考额外限定的列车。'),
    '031-zh-CN': ('minor', '瀑布之后与滑腻表达生硬，经过瀑布后的那段路的空间位置不够明确。'),
    '032-zh-CN': ('major', '押金可退还的关键财务条件漏译。'),
    '032-en': ('major', '可退还的押金只译deposit，遗漏退款条件。'),
    '033-zh-CN': ('major', 'tap card刷卡译成按一下卡，乘车刷卡动作错误。'),
    '033-en': ('major', '未说明交通工具，新增train限定。'),
    '035-en': ('major', '饮用水泛化为water，营地提供水的可饮用条件丢失。'),
    '036-zh-CN': ('minor', 'may be postponed可能推迟变成可以推迟，事件可能性与许可表述有偏差。'),
    '037-zh-CN': ('major', 'shuttle接驳车误译成航班。'),
    '037-en': ('major', '接驳车未限定bus，无依据加入公交类型。'),
    '040-zh-CN': ('major', 'return voucher返程凭券误译报销单。'),
    '040-en': ('minor', '凭券泛化为ticket，凭券与车票类型区别不明确。'),
    '042-zh-CN': ('major', '羊肉馅pastry无依据译成甜点；beef diced切丁也泛化为切碎。'),
    '043-zh-CN': ('major', '刮柠檬皮屑变成挤柠檬汁，烹饪动作和材料改变。'),
    '043-en': ('major', '刮取皮屑变成rub外层，未表达取得zest食材的动作。'),
    '044-zh-CN': ('minor', 'lentils泛化为豆子，食材具体类别未保留。'),
    '044-en': ('minor', '中文扁豆并未明确小扁豆lentils，不强制参考词；green beans比原文更泛，记录术语偏差及参考歧义。'),
    '046-zh-CN': ('minor', '整条loaf表述成面包片，切片对象与阶段表述不准确，未切断约束仍在。'),
    '047-zh-CN': ('major', '变稠之前的时序条件改成防止变稠的目的，过程关系改变。'),
    '048-zh-CN': ('minor', 'piping hot滚烫降为热，温度程度信息漏译。'),
    '048-en': ('minor', '滚烫泛化为hot，温度程度信息漏译。'),
    '049-zh-CN': ('major', 'capers刺山柑花蕾译成沙葱，食材改变。'),
    '049-en': ('major', '刺山柑花蕾译成prickly pear仙人掌花蕾，食材改变。'),
    '051-zh-CN': ('minor', 'garnish装饰配料泛化为配菜，配料角色不明确。'),
    '051-en': ('major', '分着喝同一碗汤变成各自一碗汤，无依据改变份量与分享方式。'),
    '052-zh-CN': ('major', 'bay leaf月桂叶译成香芹叶，食材改变。'),
    '053-zh-CN': ('major', 'stock高汤译成这道菜，食物对象改变。'),
    '054-en': ('minor', '滤掉籽译为strain the seeds，滤除方向不够明确，宜strain out。'),
    '055-zh-CN': ('minor', 'casserole dish炖菜烤盘表述成炖菜盘，容器名称不自然；塑料盖不能入烤箱的禁令正确。'),
    '056-zh-CN': ('minor', '瓶子是普通水的主谓搭配不自然，应说瓶中是水；含气与不含气对比仍在。'),
    '058-zh-CN': ('minor', '价格是每人两个饺子搭配生硬，包含份量的表达不准确；每人/每桌区别正确。'),
    '060-zh-CN': ('minor', 'drinking chocolate字面译饮用巧克力，冲饮用巧克力粉术语不自然。'),
    '061-zh-CN': ('major', '正的分子条件漏译，数学结论被扩大到负分子等不成立的情况。'),
    '061-en': ('minor', '比较关系使用逗号连接两个谓语，表达生硬；正分子、不变与大小关系仍保留。'),
    '064-en': ('major', '电阻器部件译成resistance电阻物理量，主语对象改变。'),
    '065-zh-CN': ('minor', '留下溶解的盐未明确原先溶解，蒸发后盐的状态表达易混淆。'),
    '068-zh-CN': ('minor', 'silt淤泥泛化为泥沙，沉积物术语精度降低。'),
    '069-zh-CN': ('minor', 'hindsight事后视角译成事后回顾，视角与回顾行为表述有偏差。'),
    '074-zh-CN': ('major', 'quadruples变为四倍译成增加四倍，数学上变成五倍。'),
    '078-en': ('major', '显微镜主语漏译，仅说magnification，原文具体仪器信息消失。'),
    '082-zh-CN': ('major', 'under tension拉伸受力泛化为受力，拉力与压力等的力学区别丢失。'),
    '082-en': ('minor', '拉伸受力译为stretched state，只说明伸长状态，受力信息不明确。'),
    '084-zh-CN': ('major', '火柴划得太早译成蜡烛点得太早，翻译了背景对象而非原文动作。'),
    '090-zh-CN': ('major', '金属加工手工具语境下file锉刀仍译成文件。'),
    '095-zh-CN': ('major', 'chord和弦译成一条弦，吉他指法对象改变。'),
    '099-zh-CN': ('minor', '草案circulating传阅译流传，文件供同事阅读的流转方式表述不自然。'),
}, 'start': {}}
ISSUES['start'] = {
    '001-en': ('major', '花盆底下译成in the bottom of the flowerpot，位置表述成花盆内部底部。'),
    '002-zh-CN': ISSUES['selected']['002-zh-CN'],
    '002-en': ('minor', '百叶帘泛化为curtain，帘子类型不明确。'),
    '003-zh-CN': ISSUES['selected']['003-zh-CN'],
    '005-zh-CN': ISSUES['selected']['005-zh-CN'],
    '005-en': ('minor', '沥水板译draining rack沥水架，器具形状泛化。'),
    '007-en': ISSUES['selected']['007-en'],
    '011-zh-CN': ('major', 'tap水龙头误用丝锥词义，维修对象改变。'),
    '013-zh-CN': ISSUES['selected']['013-zh-CN'],
    '015-zh-CN': ISSUES['selected']['015-zh-CN'],
    '015-en': ('major', '垃圾袋用完译为garbage bags are empty垃圾袋是空的，库存条件改变。'),
    '017-zh-CN': ('major', '除非提前取消的例外条件译为提前取消之前续费，条件与时间关系改变。'),
    '018-zh-CN': ISSUES['selected']['018-zh-CN'],
    '019-zh-CN': ('minor', 'finish explaining解释完泛化为说完，话语行为类别不明确。'),
    '021-zh-CN': ISSUES['selected']['021-zh-CN'],
    '022-zh-CN': ISSUES['selected']['022-zh-CN'],
    '022-en': ISSUES['selected']['022-en'],
    '025-zh-CN': ('major', '钥匙drop box归还箱译成取车箱，用途改变。'),
}
for key in ('028-zh-CN','031-zh-CN','032-zh-CN','032-en','033-zh-CN','033-en','036-zh-CN','037-zh-CN','037-en','040-zh-CN','040-en'):
    ISSUES['start'][key] = ISSUES['selected'][key]
ISSUES['start'].update({
    '042-zh-CN': ('minor', 'diced牛肉丁泛化为切碎，羊肉绞碎与牛肉切丁的加工形态对比不完整；糕点是pastry的可接受译法。'),
    '043-zh-CN': ('major', 'zest外皮屑误译柠檬果肉，食材和刮取对象改变。'),
    '043-en': ISSUES['selected']['043-en'],
    '044-zh-CN': ISSUES['selected']['044-zh-CN'],
    '044-en': ('major', '中文扁豆译为broad beans蚕豆，食材种类改变；不强制原参考lentils。'),
    '046-zh-CN': ISSUES['selected']['046-zh-CN'],
    '047-zh-CN': ('major', 'before mixture thickens的时序条件完全漏译。'),
    '048-zh-CN': ISSUES['selected']['048-zh-CN'],
    '048-en': ISSUES['selected']['048-en'],
    '049-zh-CN': ISSUES['selected']['049-zh-CN'],
    '049-en': ('major', '刺山柑花蕾译为Sichuan pepper buds花椒芽，食材改变。'),
    '051-zh-CN': ISSUES['selected']['051-zh-CN'],
    '051-en': ISSUES['selected']['051-en'],
})
ISSUES['start'].update({
    '052-zh-CN': ISSUES['selected']['052-zh-CN'],
    '053-zh-CN': ('major', 'stock高汤译成这股味道，食物主语对象丢失。'),
    '054-en': ISSUES['selected']['054-en'],
    '055-zh-CN': ('major', 'casserole dish无依据限定为意式炖菜盘，新增料理来源事实。'),
    '056-zh-CN': ('minor', '气水不是自然的气泡水术语，瓶子是普通水的主谓搭配也生硬，含气对比仍可理解。'),
    '060-zh-CN': ISSUES['selected']['060-zh-CN'],
    '061-zh-CN': ISSUES['selected']['061-zh-CN'],
    '061-en': ('major', '分子保持不变译为分子保持正值，固定大小的数学条件漏译。'),
    '064-en': ISSUES['selected']['064-en'],
    '065-zh-CN': ISSUES['selected']['065-zh-CN'],
})
for key in ('068-zh-CN','069-zh-CN','082-zh-CN','082-en'):
    ISSUES['start'][key] = ISSUES['selected'][key]
ISSUES['start']['084-zh-CN'] = ('major', 'match火柴与struck划擦动作均泛化为点火，具体对象与动作未保留。')
for key in ('090-zh-CN','095-zh-CN','099-zh-CN'):
    ISSUES['start'][key] = ISSUES['selected'][key]

def persist(role):
    path = Path(f'runs/v4-{role}-dev.jsonl')
    if role in RAW_SHA and hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA[role]:
        raise ValueError('Reviewed generation changed; read again before approving')
    raw = read_jsonl(path)
    refs = {r['id']: r for r in read_jsonl('data/prepared/v4/dev.jsonl')}
    rows = {r['id']: r for r in raw}
    expected = {f'v4-dev-{i:03}-{target}' for i in READ_PAIRS[role] for target in ('en', 'zh-CN')}
    if not expected.issubset(rows):
        raise ValueError('Inspected range includes a missing output')
    decisions = []
    for row in raw:
        if row['id'] not in expected:
            continue
        ref = refs[row['id']]
        if row['input'] != ref['input'] or row['reference'] != ref['output']:
            raise ValueError('Source/reference changed')
        verdict, note = ISSUES[role].get(row['id'].removeprefix('v4-dev-'), ('pass', '逐条核对当前原文、语境与译文，接受等义措辞；未发现需修订问题。'))
        decisions.append({'id': row['id'], 'group_id': ref['group_id'], 'category': row['category'],
            'target_lang': row['input']['target_lang'], 'verdict': verdict, 'note': note, 'reviewer': 'Codex', 'at': now(),
            'format_valid': 'prediction' in row, 'ended': row['ended'], 'language_correct': True,
            'output_hash': fingerprint({k: row[k] for k in ('input', 'raw', 'reference')}), 'purpose': 'development screening, not release'})
    summary = {'at': now(), 'reviewed': len(decisions), 'expected_total': len(refs), 'complete': len(decisions) == len(refs),
        'verdicts': dict(Counter(r['verdict'] for r in decisions)), 'reviewer': 'Codex', 'unread_rows_are_not_approved': True}
    write_jsonl(f'runs/v4-{role}-dev-semantic.jsonl', decisions)
    write_json(f'runs/v4-{role}-dev-semantic.summary.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('selected', 'start'), required=True)
    print(persist(parser.parse_args().role))

if __name__ == '__main__':
    main()
