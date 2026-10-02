"""Only changed SFT outputs actually read by Codex; no blanket pass fallback."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl
from witrans_tools.review_reuse import indexed

READ = {
    'v4-dev-021-en':('pass', '夜间长途客车、可调斜座椅及没有卧铺均正确，berths为可接受卧铺译法。'),
    'v4-dev-023-zh-CN':('pass', '行人桥关闭因而必须走地下通道的因果与要求正确，步行桥可接受。'),
    'v4-dev-033-zh-CN':('minor', 'tap刷卡译成按一下卡，读卡交互动作表达不自然；下车再次使用卡与否则收最高票价的条件保留，不按费用反转判严重。'),
    'v4-dev-043-zh-CN':('major', '刮取柠檬表层皮屑误为用柠檬挤出果肉，动作和食材部分均改变。'),
    'v4-dev-043-en':('major', '刮取外层皮屑改成peel outer layer剥掉外层皮，刮屑动作及细屑形态未保留；避免白色内皮的条件正确。'),
    'v4-dev-047-zh-CN':('minor', '让混合物打散结块的句式不自然，使用打蛋器打散的施事关系不清晰；变稠之前的时序与工具保留。'),
    'v4-dev-049-zh-CN':('major', 'capers刺山柑花蕾误为沙葱，食材类别改变。'),
    'v4-dev-049-en':('major', '刺山柑花蕾误为prickly pear仙人掌花蕾；青胡椒粒也误成pepper flakes碎片。'),
    'v4-dev-055-zh-CN':('minor', 'casserole dish炖菜盘的器皿名称不自然；可入烤箱但塑料盖不可的关键限制正确保留。'),
    'v4-dev-059-en':('pass', '奶酪外层硬而内部软正确；crust可以指硬外层，不强制使用参考rind。'),
    'v4-dev-070-zh-CN':('pass', '调查遗漏已经搬走的人正确，搬离的已完成时态保留。'),
    'v4-dev-074-zh-CN':('minor', '面积四倍增加语序不自然，容易混淆终值与增量，应明确变为原来四倍；未判定为确定五倍数值反转。'),
    'v4-dev-078-zh-CN':('pass', '显微镜放大图像而非标本本身的技术区分正确。'),
}

def main():
    raw = indexed(read_jsonl('runs/v6-sft-control-dev.jsonl'))
    refs = indexed(read_jsonl('data/prepared/v4/dev.jsonl'))
    path = Path('runs/v6-sft-control-manual-semantic.jsonl')
    previous = indexed(read_jsonl(path)) if path.exists() else {}
    for key,(verdict,note) in READ.items():
        row, ref = raw[key], refs[key]
        if row['input'] != ref['input'] or row['reference'] != ref['output'] or not row['ended'] or 'prediction' not in row:
            raise ValueError('Read structural evidence changed')
        digest = fingerprint({k:row[k] for k in ('input','raw','reference')})
        if key in previous and previous[key]['output_hash'] != digest:
            raise ValueError('Previous reading changed')
        previous[key] = {'id':key, 'group_id':ref['group_id'], 'category':ref['category'],
            'target_lang':ref['input']['target_lang'], 'verdict':verdict, 'note':note,
            'reviewer':'Codex', 'at':now(), 'format_valid':True, 'ended':True, 'language_correct':True,
            'output_hash':digest, 'purpose':'Known-development individual reading; no release approval'}
    write_jsonl(path, list(previous.values()))
    print({'manual_rows':len(previous)})

if __name__ == '__main__':
    main()
