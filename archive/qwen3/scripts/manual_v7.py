"""Persist only changed span-ablation translations explicitly read by Codex."""
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_jsonl
from witrans_tools.review_reuse import indexed

READ = {
    'v4-dev-033-zh-CN':('minor','tap刷卡译成按一下卡，交互动作表达不自然；下车再次使用卡及否则收最高票价的条件保留，与SFT相同问题用相同严重度。'),
    'v4-dev-047-zh-CN':('minor','打破结块的烹饪搭配不自然，宜打散结块；打蛋器工具、主动作与变稠之前的时序保留。'),
    'v4-dev-049-zh-CN':('major','capers刺山柑花蕾误为沙葱，食材类别改变；青胡椒粒的对比项正确。'),
    'v4-dev-050-zh-CN':('pass','涂油器皿、倒入面糊及抹平表面正确；tin未限定器皿形状深度，烤盘是可接受烘焙译法，不强制参考模具。'),
    'v4-dev-065-en':('pass','溶剂蒸发后原先溶解的盐留下，时态和物质状态均正确。'),
    'v4-dev-069-zh-CN':('minor','hindsight用事后回顾表达不自然，基于后来认识看待事件的角度不够明确；事件后写作及可能性仍保留。'),
    'v4-dev-079-zh-CN':('pass','化石给出岩层年代下限正确，年龄可用于岩层形成时间，不强制参考用词。'),
}

def main():
    raw = indexed(read_jsonl('runs/v7-critical-dev.jsonl'))
    refs = indexed(read_jsonl('data/prepared/v4/dev.jsonl'))
    path = Path('runs/v7-critical-manual-semantic.jsonl')
    previous = indexed(read_jsonl(path)) if path.exists() else {}
    for key,(verdict,note) in READ.items():
        row,ref = raw[key],refs[key]
        if row['input']!=ref['input'] or row['reference']!=ref['output'] or not row['ended'] or 'prediction' not in row:
            raise ValueError('Read source or structure changed')
        digest = fingerprint({k:row[k] for k in ('input','raw','reference')})
        if key in previous and previous[key]['output_hash']!=digest:
            raise ValueError('Previously inspected translation changed')
        previous[key] = {'id':key,'group_id':ref['group_id'],'category':ref['category'],
            'target_lang':ref['input']['target_lang'],'verdict':verdict,'note':note,
            'reviewer':'Codex','at':now(),'format_valid':True,'ended':True,'language_correct':True,
            'output_hash':digest,'purpose':'Known-development individual reading, no release approval'}
    write_jsonl(path,list(previous.values()))
    print({'manual_rows':len(previous)})

if __name__ == '__main__':
    main()
