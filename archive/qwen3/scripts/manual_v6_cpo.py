"""Persist only explicit changed outputs read by Codex, with exact raw binding."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl
from witrans_tools.review_reuse import indexed

READ = {
    'v4-dev-005-en':('minor', '沥水板译成draining rack沥水架，器具形态不够准确；冲洗和放置动作保留。'),
    'v4-dev-006-zh-CN':('pass', '周二看管狗与携带狗粮的条件均保留；您为可接受礼貌措辞。'),
    'v4-dev-007-en':('minor', '拉上关闭变为pull the zipper泛指拉动，关闭方向未明确；内衬被夹住与拉链卡住属于同一夹挂事件，不按操作人反转判严重错误。'),
    'v4-dev-008-en':('minor', '平分用split未明确equally，均分的比例约束表达不够明确；配送费包括在账单中仍保留。'),
    'v4-dev-013-zh-CN':('minor', 'ajar半开泛化成开着，门的开启程度未明确；即使门半开也须先敲门的约束仍在。'),
    'v4-dev-015-en':('pass', '垃圾袋已用完与不要扔空购物袋的因果和否定均保留。'),
    'v4-dev-017-zh-CN':('pass', '提前取消作为自动续订的例外条件正确，接受您和重复订阅的表达。'),
    'v4-dev-018-en':('pass', '划痕在保护膜而非下方玻璃正确；中文未限定单复数，英语复数可接受。'),
    'v4-dev-019-zh-CN':('major', '由你决定是否借给我反转为由我决定是否借给你，决定人和借出方向均改变。'),
    'v4-dev-022-zh-CN':('major', '车上购买更贵反转成柜台购买更贵；on board无依据具体化为飞机上，售票亭泛化为柜台。'),
    'v4-dev-022-en':('major', '原文上车未限定公交车或列车，on the bus新增公交方式；比较方向正确，不把它当费用关系反转。'),
    'v4-dev-026-zh-CN':('pass', '寄存外套但不能存行李箱的对比正确，接受寄存处的功能性译法。'),
    'v4-dev-030-en':('pass', '可携带折叠自行车且须装袋的条件均保留；实际中文携带允许carry，不强制另一方向措辞。'),
    'v4-dev-041-zh-CN':('pass', '孜然籽短暂干热处理再研磨的先后正确；toast未指定锅具，烤一下为可接受读法。'),
    'v4-dev-043-zh-CN':('major', '刮取柠檬表层皮屑变成用柠檬挤出果肉，动作和食材部分均改变。'),
    'v4-dev-049-en':('major', '刺山柑花蕾误为citrus柑橘花蕾，青胡椒粒误为green pepper flakes胡椒碎片，食材和形态改变。'),
    'v4-dev-051-zh-CN':('major', 'garnish配料无依据限定为香草；on the side也仅说放在一边，未明确单独盛放。'),
    'v4-dev-061-en':('pass', '正的分子固定与分母越大分数越小的原文关系均保留，不自行补数学前提。'),
    'v4-dev-066-zh-CN':('pass', '蒸腾期间通过气孔失水的机制正确，接受失去水分的措辞。'),
    'v4-dev-071-en':('major', '校验和checksum改成parity checks奇偶校验，错误检测方法类别改变，不能因功能相近判等义。'),
    'v4-dev-074-zh-CN':('pass', '半径加倍与圆面积变成四倍正确保留。'),
    'v4-dev-079-en':('pass', '化石给出岩层年代下限的实际原文正确译出，lower limit与lower bound等义。'),
    'v4-dev-080-zh-CN':('pass', '加入溶剂稀释且保留全部溶质的操作与条件均正确。'),
    'v4-dev-082-zh-CN':('pass', '金属弹簧语境下under tension译受拉状态正确，未误用精神紧张词义。'),
    'v4-dev-094-zh-CN':('pass', '被侵蚀小路旁树木的root正确译成树根，暴露在外为可接受等义表述。'),
    'v4-dev-100-zh-CN':('pass', '屋顶木支撑语境下beam正确译横梁，下垂保留受力弯垂状态。'),
}

def main():
    raw = indexed(read_jsonl('runs/v6-cpo-dev.jsonl'))
    refs = indexed(read_jsonl('data/prepared/v4/dev.jsonl'))
    path = Path('runs/v6-cpo-manual-semantic.jsonl')
    previous = indexed(read_jsonl(path)) if path.exists() else {}
    for key,(verdict,note) in READ.items():
        row, ref = raw[key], refs[key]
        if row['input'] != ref['input'] or row['reference'] != ref['output'] or not row['ended'] or 'prediction' not in row:
            raise ValueError('Explicitly read structural evidence changed')
        digest = fingerprint({k:row[k] for k in ('input','raw','reference')})
        if key in previous and previous[key]['output_hash'] != digest:
            raise ValueError('Previously read output changed')
        previous[key] = {'id':key, 'group_id':ref['group_id'], 'category':ref['category'],
            'target_lang':ref['input']['target_lang'], 'verdict':verdict, 'note':note,
            'reviewer':'Codex', 'at':now(), 'format_valid':True, 'ended':True, 'language_correct':True,
            'output_hash':digest, 'purpose':'Known-development individual reading; no release approval'}
    write_jsonl(path, list(previous.values()))
    print({'manual_rows':len(previous)})

if __name__ == '__main__':
    main()
