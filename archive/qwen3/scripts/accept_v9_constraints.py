"""Root's individual reading and correction of all60 frozen teacher labels."""
import hashlib
from collections import Counter
from pathlib import Path
from archive.qwen3.scripts.build_v2 import reviewed
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record

RAW_SHA = '6c534b85b2b842e3e9adc54f404d0c4efc75f543a82b7f90878ed68c9c8840f6'
FIXES = {
    '003-zh-CN': ('我让俊给维修店打电话，还没有让他安排上门。如果他们需要先检查电器，请先告诉我，再答应具体时间。', 'call具体为打电话；asked not yet不改为新的一条禁止安排命令，保留先告知后答应时序。'),
    '003-en': ('I asked Jun to call the repair shop, but I have not asked him to arrange a home visit yet. If they need to inspect the appliance first, please let me know before agreeing to a specific time.', '修正重复first；保留原文只是未让俊安排上门及先告知的限定。'),
    '004-zh-CN': ('替换坐垫比旧的更软，但并没有更厚。如果旧套子合适，就继续用它；附送的新套子只是备用的。', 'keep的条件指令不减弱为可以继续使用。'),
    '005-zh-CN': ('我们已经商定读书小组继续叫银灯。请在邀请函上使用这个名称，但街道地址保留原来的写法。', '落实Silver Lantern→银灯，教师违背显式术语表留下英文；保留地址不译。'),
    '006-zh-CN': ('洗好的衣服还没有取回来，我只是收到了一条可以取衣服的通知。如果你今天去那里，请在取你的衬衫时，也把我的外套取回来。', '修正洗衣物和准备好了的不自然表达；仍未取回且仅收到取件通知，不新增已取状态。'),
    '006-en': ('The washed clothes have not been picked up yet; I only received a notification that they are ready for collection. If you go there today, please pick up my coat along with your shirts.', '外套用coat，中文衬衫未限定单数；取件对象的my/your及条件保留。'),
    '008-zh-CN': ('我们订了三晚的小木屋，星期四退房。如果把抵达时间改到星期二，住宿就会缩短为两晚，除非退房日期也改变。', 'nights不是天；语境原周一到达，改周二不是提前。保留unless条件。'),
    '009-zh-CN': ('接驳车停在市场对面，不是在市场入口。上车之前，请问司机这一班是否开往北侧航站楼。', '司机/上车语境的shuttle不是航天飞机；保留对面与入口区别。'),
    '010-en': ('Only the outdoor performance has been cancelled. The exhibition remains open, but the original ticket does not automatically include admission to the exhibition.', '中文未明示多个演出或多张票，使用与此具体场景相符的单数；只取消演出且展览入场非自动包含。'),
    '011-zh-CN': ('我们的步行路线叫柳渡线。地图上标出的桥禁止车辆通行，但行人仍可以使用。', '修正对车辆关闭的搭配；术语、车与行人许可差异保留。'),
    '012-zh-CN': ('导游说，如果水位下降，返程船可能会提前出发。她并没有说时间已经改好了，所以改变我们的计划之前，请先核实。', 'rescheduled明确为出发时间调整，改善它已经被重新安排的搭配；might与尚未证实状态保留。'),
    '014-en': ('Both jars contain pickled onions, but only the smaller jar has chilli in it. Please use the onions from the larger jar for this dish and put the lid back on the original jar.', '恢复教师遗漏的这道菜范围；小罐辣椒、大罐取料、盖子归原罐均保留。'),
    '015-zh-CN': ('给豆子沥水之前，先留出两汤匙煮豆子用的液体。只有混合物太干时才加入留出的液体，这并不是额外的一份酱汁。', '改善煮豆液体的表达；英语cooking liquid不额外限定清水或高汤；两汤匙、only if、非额外酱汁保留。'),
    '016-zh-CN': ('菜单把这款甜点叫作琥珀云。请保留这个名称，并翻译下面的描述，不要添加未列出的配料。', 'ingredients明确为配料；术语与只译正文不新增事实保留。'),
    '018-zh-CN': ('标签说这款酸奶没有添加糖，并没有说它完全不含糖。请分别翻译这两种说法，不要把它们当作同一个意思。', '修正声明和视为等同的翻译腔；无添加糖与完全无糖不混淆。'),
    '019-en': ('For every accepted sample, there must be at least one valid reading. This does not require a single reading that is valid for all samples.', '读数是reading，不是read；forall-exists与exists-forall的作用域区别完整保留。'),
    '022-en': ('The model assigns probability 0.8 to the observed event. This value is not a statement that eighty percent of the event happened.', '教师无依据添加of the time，变为发生频率；恢复事件发生百分之八十的原命题，不将错误命题改成另一命题。'),
    '023-zh-CN': ('这个区间在零处是开端点，在一处是闭端点：(0, 1]。零不包含在内，一则包含在内，请保留这个记号和端点的区别。', '修正边界区别的区分赘语；开闭端点、公式及保留两项的要求完整保留。'),
    '023-en': ('This interval has an open endpoint at zero and a closed endpoint at one: (0, 1]. Zero is excluded, while one is included; please preserve both the notation and the distinction between the endpoints.', '教师把保留记号和端点区别误作记号与端点之间的区别，恢复both范围。'),
    '024-zh-CN': ('仪器是在第一次测量之后校准的。因此，后来仪器之间的测量结果一致，并不能反过来证明第一次测量有效。', '修正追溯地验证该第一次测量；校准发生在第一次之后且不能反证有效，时序不变。'),
    '025-zh-CN': ('她在签字之前临阵退缩了。她要求再考虑一天，但并没有拒绝这个提议。', 'cold feet明确为临阵胆怯而非物理冷；修正再有一天时间且保留未拒绝。'),
    '026-zh-CN': ('维修店这个月终于收支平衡了。这意味着收入弥补了成本，并不是赚了一大笔钱。', '改善覆盖了成本的搭配；收支相等与巨额盈利区别保留。'),
    '027-zh-CN': ('你替我顶班，我欠你一个人情。我的意思是我很感激你的帮助，并不是说我们已经约定了报酬。', '教师我欠你一个人情，替我顶班可变成请求将来顶班；恢复已帮忙的原因关系。'),
}


def main():
    destination = Path('data/prepared/v9-constraints/train.jsonl')
    if destination.exists():
        raise ValueError('Preserve approved training labels')
    path = Path('data/generated/v9_constraint_candidates.jsonl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != RAW_SHA:
        raise ValueError('Individually read candidates changed')
    rows = read_jsonl(path)
    references = {r['id']:r for r in read_jsonl('data/prepared/v9-constraints/references.jsonl')}
    if len(rows) != 60 or len({r['id'] for r in rows}) != 60 or set(references) != {r['id'] for r in rows}:
        raise ValueError('Reviewed candidate pool incomplete')
    decisions = []
    for row in rows:
        if any(row[key] != references[row['id']][key] for key in ('input','source','group_id','category')):
            raise ValueError('Frozen source or provenance changed')
        raw_hash = fingerprint(row)
        key = row['id'].removeprefix('v9-constraint-')
        note = '逐条核对实际原文、语境、角色、否定、条件、数值和术语；译文等义，未添加正文外内容。'
        if key in FIXES:
            answer, note = FIXES[key]
            row['output'] = {'translation':answer}
        for original, required in row['input']['glossary'].items():
            if original in row['input']['text'] and required not in row['output']['translation']:
                raise ValueError('Explicit glossary still missing')
        reviewed(row, 'Codex individually read all60 actual teacher labels against immutable source; source-bound corrections documented, training only')
        validate_record(row, True, True)
        decisions.append({'id':row['id'], 'action':'correct' if key in FIXES else 'accept_teacher',
            'note':note, 'raw_record_hash':raw_hash,
            'reviewed_hash':fingerprint({'input':row['input'],'output':row['output']})})
    if not set(FIXES).issubset({r['id'].removeprefix('v9-constraint-') for r in rows}):
        raise ValueError('Correction ID absent')
    write_jsonl(destination, rows)
    write_jsonl('data/generated/v9_constraint_decisions.jsonl', decisions)
    report = {'at':now(), 'reviewed_rows':len(rows), 'groups':len({r['group_id'] for r in rows}),
        'actions':dict(Counter(r['action'] for r in decisions)), 'raw_sha256':RAW_SHA,
        'train_hash':fingerprint(rows), 'context_rows':sum(bool(r['input']['context']) for r in rows),
        'glossary_rows':sum(bool(r['input']['glossary']) for r in rows), 'multi_sentence_rows':len(rows),
        'scope':'Future training-only60-row extension. Not added to already frozen758-row v8 trial. No semantic accuracy or release claim; Codex AI acceptance, not independent professional human labels.'}
    write_json('runs/v9-constraints-data-acceptance.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
