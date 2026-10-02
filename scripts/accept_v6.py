"""Codex's completed source/translation audit of all80 synthesis candidates."""
import hashlib
from pathlib import Path
from collections import Counter
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = '1c56ce59fd9cd79aecc6b40d1b41ded3534103c5ed47978e69f197adc9a8f944'
EXCLUDED = {'v6-daily-01':'团队进入储物柜的来源表述不清，英文时间未说明上午而中文新增上午；不使用这组来源。',
    'v6-travel-06':'after the market未说明逛完市场还是市场活动结束，中文选择了活动结束；来源有歧义。',
    'v6-academic-02':'自生成学术来源把位移定义为直线距离而未保留方向，避免训练不严谨定义。'}
FIXES = {
    'v6-daily-02-zh-CN': ('我把上周末你借给我的耙子和手套带回来了。希望它们都完好无损。','brought back归还方向漏译。'),
    'v6-daily-02-en': ('I brought the rake and gloves you lent me last weekend. I hope they are all in one piece.','中文只说带来，不补回到原处。'),
    'v6-daily-03-zh-CN': ('我认为我们应该把预算讨论移到议程靠前的部分，以便更早敲定。','first part不强制限定前半。'),
    'v6-daily-03-en': ('I think we should move the budget discussion to the first half of the agenda so we can finalize it earlier.','中文前半保留first half。'),
    'v6-daily-06-zh-CN': ('有人能负责报名台，再由另一个人负责餐饮区吗？','sign-up报名不能改签到。'),
    'v6-daily-06-en': ('Can someone handle the check-in desk and someone else manage the food station?','实际中文签到译check-in，不强制原英语sign-up。'),
    'v6-travel-03-zh-CN': ('能把我的行李直接送到酒店吗？','恢复have luggage delivered的配送请求。'),
    'v6-travel-04-zh-CN': ('我想安排明天离开之前早点吃早餐。','early breakfast较早吃早餐，不能只用早饭替代early。'),
    'v6-travel-04-en': ('I would like to arrange breakfast before I leave tomorrow.','实际中文只有早饭，不能再添加early的时间限定。'),
    'v6-travel-05-zh-CN': ('这条小路有些陡峭的路段。我不确定我的自行车能否应付。','修正不确定加吗的混合句式。'),
    'v6-food-04-zh-CN': ('顾客点了一份正常份量的汤和一份试吃样品，以便之后比较味道。','明确regular soup portion的份量与试吃份量。'),
    'v6-food-05-zh-CN': ('把面粉拌进融化的油脂，直到它变成金黄并散发香味。','厨房fat使用油脂，改善融化的脂肪字面表述。'),
    'v6-academic-03-zh-CN': ('如果物质回到原始状态，这种相变就是可逆的。','returns保持实际返回，不添加can。'),
    'v6-academic-03-en': ('A phase change is reversible if the substance can return to its original state.','实际中文能回到保留can return。'),
    'v6-academic-05-zh-CN': ('侵蚀会削蚀土地，沉积则会把沉积物堆积在土地上。','修正沉积则通过沉积物堆积起来的对象与语法缺失。'),
    'v6-hard-03-zh-CN': ('这个账户的利息按月复利计算。','账户量词与按月复利表述自然化。'),
    'v6-hard-08-zh-CN': ('该证明依靠归纳步骤来确立这一规律。','数学pattern使用规律，不使用建立模式的建模含义。'),
}
RECLASSIFY = ['academic','daily','daily','daily','academic','daily','daily','academic']

def main():
    destination = Path('data/prepared/v6-source/train.jsonl')
    if destination.exists():
        raise ValueError('Accepted corpus immutable')
    path = Path('data/prepared/v6-source/candidates.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Candidates changed; inspect again')
    raw = read_jsonl(path)
    if len(raw) != 80:
        raise ValueError('Unexpected reviewed pool size')
    rows, decisions = [], []
    for candidate in raw:
        row = dict(candidate)
        case = row['id'].rsplit('-',1)[0] if row['input']['target_lang']=='en' else row['id'].removesuffix('-zh-CN')
        if case in EXCLUDED:
            decisions.append({'id':row['id'],'action':'exclude_source','note':EXCLUDED[case], 'raw_record_hash':fingerprint(candidate)})
            continue
        fixed = FIXES.get(row['id'])
        if fixed:
            text, note = fixed
            row['output'] = {'translation':text}
        else:
            note = '逐条核对实际原文、语境和译文，接受等义措辞。'
        category_changed = row['category']=='hard'
        if category_changed:
            row['category']=RECLASSIFY[int(case.rsplit('-',1)[1])-1]
            note += ' 原文自身已提示词义，不称仅靠语境消歧难例；重新按实际来源分类。'
        reviewed(row,'Codex read all80 teacher-generated source/translation candidates; per-direction repairs and exclusions bound to immutable raw; no independent teacher reference')
        validate_record(row,True,True)
        rows.append(row)
        decisions.append({'id':row['id'],'action':'correct' if fixed else 'accept_teacher','note':note,'category_reclassified':category_changed,
            'raw_record_hash':fingerprint(candidate),'reviewed_hash':fingerprint({'input':row['input'],'output':row['output']})})
    assert set(FIXES).issubset({r['id'] for r in raw})
    write_jsonl(destination,rows)
    write_jsonl('data/generated/v6-source/decisions.jsonl',decisions)
    report={'at':now(),'raw_rows_read':80,'accepted_rows':len(rows),'accepted_groups':len({r['group_id'] for r in rows}),
        'actions':dict(Counter(d['action'] for d in decisions)),'reclassified_rows':sum(d.get('category_reclassified',False) for d in decisions),
        'categories':dict(Counter(r['category'] for r in rows)),'context_rows':sum(bool(r['input']['context']) for r in rows),
        'raw_sha256':RAW_SHA,'train_hash':fingerprint(rows),
        'limitations':['Teacher did not follow requested two-sentence length consistently; this pool does not supply the promised40 multi-sentence pairs',
            'All requested hard cases self-disambiguated, so none counted as genuine context-only hard coverage',
            'Model authored source and translation together; Codex checked both, not independent professional human labels',
            'Exact prior-source exclusion verified in synthesis; no broad semantic near-duplicate guarantee. Same sense families retained in v5 group IDs'],
        'scope':'Training-only extension; original API batches, excluded sources and corrected outputs retained; no release evidence'}
    write_json('runs/v6-data-acceptance.json',report)
    print(report)

if __name__=='__main__':
    main()
