"""Root's complete source-grounded reading of80 paragraph labels."""
import hashlib
from pathlib import Path
from collections import Counter
from scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = '4d24635ffd32a517f2e9bc5a8a80270a9be11e555290da2582ef5a2ab6685a00'
FIXES = {
    '001-zh-CN': ('我把一把备用钥匙交给管理员了。只有我们两个人都不能在你到达之前回到家时，你才去找管理员要钥匙。', '恢复only if限制；不将neither of us擅自改成我和他或添加管理员性别。'),
    '003-en': ('If everyone agrees, we can move the meeting to the afternoon. Until then, keep the original time in the shared calendar.', '保留共享日历里原定时间的安排使用keep，不改为对未来状态的预测。'),
    '004-zh-CN': ('铰链又松了。我拧紧螺丝时，请把门扶稳，并让手指远离门缝。', 'keep away保持距离，不减弱为只不伸进门缝。'),
    '005-zh-CN': ('请把毯子叠好，不要直接塞进柜子。搁板可以放得下，但最下面的抽屉仍然潮湿。', 'bottom drawer保留最下面，不泛化成下面任一抽屉。'),
    '007-zh-CN': ('我已经把文档重新命名为静谧信标了。虽然文件夹仍用旧名称，请在会议纪要里使用这个名称。', '落实显式术语表Quiet Beacon→静谧信标，教师留下英文名称。'),
    '008-zh-CN': ('地板现在已经干了，但清洁提示牌还放在外面。确认没人需要它之后，你能把它拿进来吗？', '保留checking that nobody needs it的无人需要条件，不只说检查是否有人需要。'),
    '010-en': ('The museum stops admitting visitors one hour before closing. If you arrive later, your reservation alone will not allow you to enter.', '保留仅凭预约不足，而非扩展为晚到者无论其他条件一律不能入内。'),
    '011-zh-CN': ('在小路分岔之前，让路标一直位于你的左侧。在分岔处走较宽的那条，不要走通向坡上的窄路。', '小路branch使用岔路，不用不自然的分支；uphill不擅自限定山。'),
    '012-en': ('Our room faces the courtyard, not the street. Could we keep the same room for an additional night, even if breakfast is more expensive?', 'even if即使的让步不能只译if条件。'),
    '013-zh-CN': ('时刻表上列了两班出发时间，但星期日只有较晚的那班运行。选择接续班次前，请核对一下日期。', '未说明交通方式，不添加列车；修正请选择连接班次前请的语法。'),
    '016-en': ('If the road is clear, the driver can stop near the east gate. If it is blocked, we will get off at the main square instead.', '保留当前司机和临时停车，不变为所有司机停车泊车的泛化许可。'),
    '018-zh-CN': ('这团面团还需要静置一次才能擀开。我们准备馅料时，把它轻轻盖住，以免表面变干。', '恢复擀开之前的时序条件；面团静置不用休息，不添加盖子类型。'),
    '019-en': ('The dressing contains mustard, but the roasted vegetables do not. Please serve the dressing separately so each person can decide whether to add it.', '无山葵限定的芥末按常规mustard译，不特指wasabi；记录中文口语中名称存在混用，不作为严格术语测试。'),
    '020-zh-CN': ('底部的米饭略微发脆，没有烧焦。用锅铲轻轻铲起米饭，不要刮到锅的涂层。', 'lift铲起不变成翻动；不添加涂层只在锅底的位置限定。'),
    '020-en': ('The rice at the bottom is slightly crisp but not burnt. Use a spatula to gently lift the rice without scraping the coating of the pan.', '涂层未指定non-stick材料，不擅自添加。'),
    '023-zh-CN': ('菜单把这种酱叫作金色果园。酱的名称请使用这个译名，但配料表按通常方式翻译。', '落实Golden Orchard→金色果园的显式术语约束。'),
    '024-zh-CN': ('馅饼冷却后就会变得足够结实，可以切片。在那之前，把它从托盘上移走可能会弄碎底部。', 'tart是整个馅饼，不只是酥皮；tray不强制限定烤盘。'),
    '028-zh-CN': ('这个函数接受空列表，但拒绝缺少参数的情况。这两种情况在文档和测试中必须明确区分。', 'cases remain separate指区别处理，不误作情况保持独立的关系属性。'),
    '030-en': ('Due to the small sample size, the uncertainty interval of the estimate is wide. Collecting more independent observations may narrow it, but repeating the same observation does not increase independence.', '不确定区间不能擅自限定为频率学派confidence interval。'),
    '031-zh-CN': ('报告把这个指标称为边界保留度。摘要中请始终使用约定的术语；如果可能与另一个指标混淆，就不要把它缩写成保留度。', '保留when could be confused条件，不改为无条件禁止缩写。'),
    '033-zh-CN': ('这是一项艰巨的任务。我们不能保证在星期五之前完成。', 'tall order是艰巨要求或任务，改善一个很高的要求的字面表达。'),
    '034-zh-CN': ('不要泄露秘密。我们说好了等所有人到齐。', 'wait保留等待，不把语境补成到齐后揭晓的行动。'),
    '035-zh-CN': ('现在该由你来决定了。我们会等你作出决定后再进行另一项改动。', 'ball in your court指轮到对方决策，不按球在你这边字面翻译。'),
    '036-zh-CN': ('我们先把这件事搁置一下。等数字核对好之后，我们可以再回过头来处理它。', '保留can可以，不把回头处理变为确定承诺。'),
    '037-zh-CN': ('他冷落了我们。我们的两个问题都没有得到回答。', '改善给我们冷遇搭配，并保留neither限定的两个问题。'),
    '038-zh-CN': ('我们不应该偷工减料。即使截止日期变了，每个阶段也必须接受同样的检查。', '保留should not建议强度和工作质量含义，不泛化为禁止所有捷径。'),
}

def main():
    destination = Path('data/prepared/v8-multisentence/train.jsonl')
    if destination.exists():
        raise ValueError('Accepted paragraph labels immutable')
    path = Path('data/generated/v8_multisentence_candidates.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Reviewed teacher labels changed')
    rows = read_jsonl(path)
    refs = {r['id']:r for r in read_jsonl('data/prepared/v8-multisentence/references.jsonl')}
    if len(rows) != 80 or {r['id'] for r in rows} != set(refs):
        raise ValueError('Reviewed pool incomplete')
    decisions = []
    for row in rows:
        ref = refs[row['id']]
        if any(row[k] != ref[k] for k in ('input', 'group_id', 'category', 'source')):
            raise ValueError('Frozen source changed')
        raw_hash = fingerprint(row)
        key = row['id'].removeprefix('v8-multi-')
        note = '逐条核对实际原文、上下文与术语约束；接受等义措辞，双向各按实际来源审核。'
        if key in FIXES:
            answer, note = FIXES[key]
            row['output'] = {'translation': answer}
        if key == '011-en':
            note += ' 中文实际原文已含山上，不强制套用英语参考uphill；作者两种来源各自冻结。'
        if key == '024-en':
            note += ' 中文馅饼可译pie，不因原英语为tart强制改词。'
        for original, required in row['input']['glossary'].items():
            if original in row['input']['text'] and required not in row['output']['translation']:
                raise ValueError('Explicit glossary repair still missing')
        reviewed(row, 'Codex individually read all80 teacher paragraph labels; condition and glossary repairs documented against immutable source; no release use')
        validate_record(row, True, True)
        decisions.append({'id':row['id'], 'action':'correct' if key in FIXES else 'accept_teacher', 'note':note,
            'raw_record_hash':raw_hash, 'reviewed_hash':fingerprint({'input':row['input'], 'output':row['output']})})
    assert set(FIXES).issubset({r['id'].removeprefix('v8-multi-') for r in rows})
    write_jsonl(destination, rows)
    write_jsonl('data/generated/v8_multisentence_decisions.jsonl', decisions)
    report = {'at':now(), 'reviewed_rows':len(rows), 'groups':len({r['group_id'] for r in rows}),
        'actions':dict(Counter(d['action'] for d in decisions)), 'raw_sha256':RAW_SHA, 'train_hash':fingerprint(rows),
        'glossary_rows':sum(bool(r['input']['glossary']) for r in rows), 'multi_sentence_source_rows':80,
        'reference_caveats':['011Chinese source explicitly mentions mountain; English only uphill', '019Chinese mustard name has colloquial ambiguity; not a strict term test', '024Chinese pie name broader than English tart'],
        'scope':'Future training-only extension; not added to frozen running678-row experiment; no accuracy or release claim'}
    write_json('runs/v8-multisentence-data-acceptance.json', report)
    print(report)

if __name__ == '__main__':
    main()
