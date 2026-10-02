"""Explicit individual decisions for changed v8 known-development outputs."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl
from witrans_tools.review_reuse import indexed

READ = {
 'v4-dev-006-zh-CN':('pass','周二照顾狗及provided带狗食条件保留；您/你仅礼貌程度变化，没有新增承担或遗漏条件。'),
 'v4-dev-013-zh-CN':('minor','ajar半掩译成一般开着，部分开启程度丢失；敲门先于进来和even if让步保留。'),
 'v4-dev-016-en':('pass','没关灯的否定、理由为电工仍在干活、说话者角色保留；中文灯未指定数量，单数light可接受。'),
 'v4-dev-019-en':('pass','先让说话者解释完，听者随后决定是否借给说话者，时序、角色和是否选择完整；explaining it在此对话可指待解释的事情。'),
 'v4-dev-022-zh-CN':('major','on board未指定交通方式，无依据限定飞机；车票/飞机搭配也冲突。反转比较语序仍等义，主要问题是新增模式；kiosk译柜台较泛不抵消该严重问题。'),
 'v4-dev-023-en':('pass','人行天桥关闭、行人必须改走地下通道均保留；pedestrian bridge与underground passage为可接受等义表达。'),
 'v4-dev-031-zh-CN':('minor','瀑布之后指经过瀑布后的路段，中文搭配不够自然，宜在经过瀑布后；路面变滑与即使干燥日子也如此的让步仍可理解。'),
 'v4-dev-038-zh-CN':('minor','information counter译成信息柜台不够自然，宜咨询台；兑换处更早关门的比较方向完整保留。'),
 'v4-dev-040-zh-CN':('pass','返程凭证有效条件为工作日且排除节假日，整体范围与原文相符；不从未指定国家推断调休规则或强制某一参考措辞。'),
 'v4-dev-041-en':('pass','孜然籽、先短时烘炒再研磨保持；roasted可用于炒香香料，不因未加pan一词就判成不同步骤。'),
 'v4-dev-042-zh-CN':('minor','diced切丁译成一般切碎，特定形状细节变弱；绞羊肉与非牛肉的否定对比保留。用羊肉填的表达也不够自然。'),
 'v4-dev-043-zh-CN':('major','from the lemon误成用柠檬，柠檬从被刮取对象变成工具；应刮取柠檬表层皮且不带苦味白瓤。'),
 'v4-dev-051-en':('major','分着喝这碗汤变成each have a bowl，每人一碗，丢失共分同一碗的数量关系并新增碗数；装饰配料另放虽保持仍不能补救。'),
 'v4-dev-057-zh-CN':('minor','serve宜上桌或供食用，当前仅食用表达略泛；持续冷藏直至准备供应的主时序保留，取出是上桌的自然隐含动作不单独算新增事实。'),
 'v4-dev-059-zh-CN':('pass','奶酪表皮firm与中心soft的对比保留，紧实/柔软是可接受性质表述。'),
 'v4-dev-063-zh-CN':('pass','磁场方向变化而大小保持，向量方向与量值两项关系正确。'),
 'v4-dev-066-zh-CN':('pass','叶片在蒸腾中通过气孔失水，结构术语、通道与过程保持。'),
 'v4-dev-070-zh-CN':('pass','调查遗漏已经搬走的人，受调查人群与已完成搬离状态保持；未添加遗漏原因。'),
 'v4-dev-071-en':('pass','校验和只能检测某些传输错误而不能纠正，检测/纠正和能力范围均保留。'),
 'v4-dev-081-en':('pass','植物组织语境下cell为细胞，细胞壁异常厚正确；不把显微镜背景添加到正文。'),
 'v4-dev-090-zh-CN':('major','明确的file锉刀被泛化为工具，sharpening磨利也变成一般打磨；短句两项核心对象/动作都失去特定含义，金属加工语境未落实具体词义。'),
}


def main():
    raw = indexed(read_jsonl('runs/v8-selected-dev.jsonl'))
    refs = indexed(read_jsonl('data/prepared/v4/dev.jsonl'))
    path = Path('runs/v8-selected-manual-semantic.jsonl')
    previous = indexed(read_jsonl(path)) if path.exists() else {}
    for key,(verdict,note) in READ.items():
        row,ref = raw[key],refs[key]
        if row['input']!=ref['input'] or row['reference']!=ref['output'] or not row['ended'] or 'prediction' not in row:
            raise ValueError('Read source or structure changed')
        digest = fingerprint({k:row[k] for k in ('input','raw','reference')})
        if key in previous and previous[key]['output_hash']!=digest:
            raise ValueError('Previously read output changed')
        previous[key] = {'id':key,'group_id':ref['group_id'],'category':ref['category'],
            'target_lang':ref['input']['target_lang'],'verdict':verdict,'note':note,
            'reviewer':'Codex','at':now(),'format_valid':True,'ended':True,'language_correct':True,
            'output_hash':digest,'purpose':'Known development screening only; no release approval'}
    write_jsonl(path,list(previous.values()))
    print({'manual_rows':len(previous)},flush=True)


if __name__=='__main__':
    main()
