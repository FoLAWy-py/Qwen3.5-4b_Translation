"""Bind root Codex AI's complete actual-output reading to frozen source rows.

The REVIEWED_COUNT and judgments are updated only after personal reading.
No reference equality or automated semantic classifier is used.
"""
import json
from pathlib import Path
from witrans_tools.common import read_jsonl,write_jsonl,write_json,fingerprint,now
from witrans_tools.independent_quality import evaluate

REVIEWED_COUNT=400
# Zero-based actual row index: verdict, source-based judgment, constraint retained.
JUDGMENTS={
0:('minor','跪下译为蹲下，且遗漏散热器热的状态；避蒸汽、耳环归属及术语保留。',True),
7:('minor','same mugs 容易读作同一批杯子，原文是相同款杯子；贴纸与大小关系正确。',True),
13:('major','餐桌椅在此指餐椅，译文额外把 dining table 也列为稳定对象；毛毡垫约束保留。',True),
14:('minor','一起折叠表达成把床笠叠在一起，合作者关系不够清楚；其余事实及床笠术语保留。',True),
25:('minor','搜查靠垫改为靠垫后方，增加位置限定；另外两处正确。',True),
31:('minor','master 在此不自然，原文主人是被子主人；询问拍摄许可与被子描述完整。',True),
32:('minor','分类、国家时代及文件夹完整，遗漏封好。',True),
33:('minor','年代缩窄为year，时代分类粒度不准；封好文件夹完整。',True),
34:('minor','toast只写致辞，祝酒性质不明确；对镜练习与语调节奏正确。',True),
38:('minor','cousin无性别年龄约束，堂弟额外指定男与年幼；送植物和便条完整。',True),
39:('minor','gave our cousin\'s apartment表达收件位置和收件人不自然；植物、卡片、新公寓可辨。',True),
50:('minor','伞的catch是锁扣机构，伞钩名称不准；室内测试顺畅打开的目的完整。',True),
52:('major','look for a safe place 寻找安全位置，被改为已经把镇纸放好，主要动作与完成状态改变。',True),
60:('minor','whistling口哨方式未明确，吹奏可以指乐器；旋律、逐音反复和终点正确。',True),
62:('minor','message改为一封信，postcard只写感谢卡，载体表达不准；感谢与加画目的保留。',True),
63:('minor','postcard的明信片载体缩为一般card；留言与加画目的正确。',True),
66:('minor','cousin无父系限定，堂亲缩窄了亲缘范围；新昵称与使用请求完整。',True),
68:('major','passes around传阅改为围着肖像，家人的主要动作改变；指出熟悉面孔完整。',True),
69:('major','传阅旧照片改为looked through，失去成员之间传递的主要动作；脸庞单复数也不准。',True),
72:('major','单一sibling增加为兄弟姐妹俩，笑点punchline未翻译；笑与努力回忆尚可辨。',True),
81:('minor','遗漏ferry渡轮限定；婴儿车、头顶舱、毯子和膝上位置正确。',True),
90:('minor','有遮阳吊床的位置不够清楚表达有树荫的可用营位；吊床术语与编号位置保留。',True),
92:('minor','下一次低潮的时间范围写成用于低潮，时刻表日期措辞生硬；核心检查及术语仍可辨。',True),
93:('minor','the date of the morning 未明确当天早上，日期表达不自然；下次低潮堤道状态正确。',True),
99:('major','步行者写为专名The Pacers，改变主体身份；饮水设施开放与洁净正确。',True),
108:('minor','upper上层遗漏，渡轮甲板的遮阳检查正确。',True),
112:('major','boot-drying rack误成普通公共鞋架，核心烘干功能遗漏。',True),
113:('minor','靴子烘干架泛化为boot dryer，架体形式不明确；入口附近与寻找正确。',True),
114:('minor','mountain pass地形山口变为山道，地点性质不准；进入前检查隧道长度可辨。',True),
119:('minor','空地缩为spot，开阔空地性质不明确；河边、阴凉和野餐保留。',True),
139:('minor','早已废弃的时间跨度未明确，只写abandoned；一家人探索铁路月台正确。',True),
144:('major','salt flats盐滩改为盐湖，旅行地点的地貌改变；确认日出之旅细节正确。',True),
156:('minor','wheelchair mat明确垫体只写轮椅道，材料设备形式不明；救生员及位置问询正确。',True),
157:('major','轮椅通行垫改为ramp坡道，通行设施类型改变。',True),
158:('minor','coin-operated投币方式遗漏，只写自助洗衣店；最近地点及向酒店员工询问正确。',True),
160:('minor','shaped into a braid写揉成辫子，成形操作措辞不准；先静置和毛巾覆盖正确。',True),
161:('major','辫子形改为rope绳条形，面团最终形状改变；酸种术语、静置顺序和毛巾正确。',True),
164:('major','rice pudding米布丁两处变为椰奶布丁，主要菜品与食材改变；豆蔻及四位客人保留。',True),
166:('minor','盛放碗额外限定为小碗，黄色宽碗沿表达不自然；栗子保湿与术语正确。',True),
168:('major','fold into烘焙翻拌误作把蛋白折叠到蛋糕，主要操作不正确；蛋白术语与烘焙纸正确。',True),
170:('minor','削皮刀泛化小刀，fibrous choke花蕊写刺球不准；分离洋蓟心及蓝盘保留。',True),
171:('major','削皮刀改为peeler削皮器，主要工具类型改变；洋蓟、分离和蓝盘正确。',True),
173:('minor','壶写为pot，储液容器表达不够明确；香菇梗浸泡软化、高汤和保留浸泡水正确。',True),
182:('minor','芳香的糊涂抹一串容易误读，糊状物表达需整理；香草、鳟鱼腹腔和调味事实完整。',True),
188:('minor','快速手腕翻转表达不自然且没直接写可丽饼翻面，动作可由边缘语境辨认。',True),
190:('minor','segment果瓣写每个段不自然；锋利刀具、葡萄柚苦膜与分离完整。',True),
196:('major','brittle酥糖变成抽象脆性，主要被盛装食物丢失，人物They也写成它们。',True),
198:('major','nutmeg肉豆蔻误为肉桂粉，主要调味食材替换。',True),
202:('minor','chives细香葱泛化为葱，具体葱类不明确；未明确替换为另一种菜，按轻微精度损失判定，厨房剪刀正确。',True),
204:('minor','leek韭葱泛化为葱，具体葱类不明确；未明确替换为另一种菜，按轻微精度损失判定，切细长条正确。',True),
205:('major','韭葱误为scallions普通青葱，主要食材替换。',True),
209:('minor','烤好的甜菜只写cooked，烤制方式遗漏；静置位置及剥皮先后正确。',True),
210:('major','saffron藏红花误为姜黄，主要酱料食材替换。',True),
211:('minor','壶写为pot，盛放器形不明确；藏红花、温热和上桌正确。',True),
216:('minor','馅料wrapper叫包装纸不自然，实际仍用卷心菜叶包馅，主要动作可辨。',True),
218:('minor','near the stem柄附近变为柄部本身，按压位置范围缩窄；轻压测试软硬正确。',True),
220:('major','haddock黑线鳕误成鲑鱼，flaked也写敲碎，主要食材及形态操作不准。',True),
221:('major','黑线鳕误为black cod黑鳕，鱼种不同；烟熏、小块和轻拌入鱼派馅正确。',True),
230:('minor','loosening skins表皮松脱只写变软，去皮准备效果不够准；冷却后剥皮正确。',True),
234:('major','juniper berries杜松子译为迷迭果，主要香料无法正确辨认。',True),
238:('minor','cheese wheel译奶酪轮，圆形整块奶酪表达不自然；线切与避免压实正确。',True),
246:('minor','第二句arrived送达事件遗漏，仅保留装在编号纸箱；矿物解理与面方向完整。',True),
256:('major','生态学plant recruitment植物补充更新误为植物招募，核心生态过程术语不正确。',True),
260:('major','权重对象应为与未回复者相似的已回复受访者，译文把未回复者与已回复者关系和赋权对象倒置。',True),
263:('minor','in the alternative orchestration未明确是否存在的差别，但也未断言两个版本都有替代编排；差异范围表达欠明确，接受其保留主要比较而判轻微问题。',True),
264:('major','rhizome根状茎误为块茎，植物器官类型改变。',True),
286:('major','age-standardized年龄标准化遗漏年龄，核心死亡率标准化方法不完整。',True),
294:('major','ergative argument作格论元写成格标记，论元与标记实体混淆且术语留英文。',True),
313:('minor','遗漏原文对feedback反馈进行分类的范围，只写区分两种评估；学习与评分用途正确。',True),
322:('major','印刷工printer误成打印机，固定滚筒误成固定印版；术语印版保留但主体与操作对象错误。',True),
336:('major','装订语境signature书帖误为签名，折叠缝入的主要对象改变。',True),
338:('major','不能移动否则王受攻击，被反转为不移动就暴露王，否定条件错误。',True),
343:('major','糖化醪mash写为mashing liquor，混淆糖化醪与糖化用液体；稠度检查虽保留但材料对象不准。',True),
346:('major','法律service送达误为服务通知，actual notice实际知悉通知也不明确，核心法律区分错误。',True),
348:('minor','舞蹈动作段落phrase写动作短语，术语不自然；排练中描述其节奏正确。',True),
350:('minor','warp end指一根经纱，写经线末端增加了位置理解；备用线修补断经的动作仍清楚。',True),
352:('major','forme印版误为字模，活字放入的对象改变，字肩也写成泛指肩。',True),
353:('major','印版forme误为composing stick排字盘，排字步骤和容器对象不同。',True),
356:('major','pontil mark铁棒痕被造词点托痕迹，玻璃工艺中所打磨的主要痕迹不能准确识别。',True),
358:('critical','攀岩cam凸轮塞误为快挂，lobes误为叶瓣，充分接触岩面变为嵌入岩石；将保护器材及安置条件改错，可能直接导致使用不合适器材或错误建立攀岩保护。',True),
359:('major','凸轮塞写cam locks，保护器材与一般凸轮锁混淆；各凸轮瓣写all cams，部件名称不准。充分接触岩面条件保留。',True),
360:('major','frog弓根组件未翻译留英文，核心调节部件不明确；调节弓毛张力目的保留。',True),
361:('major','弓根装置误为bow grip device弓握把装置，张力调节组件不准确。',True),
362:('major','foundation sheet巢础板误为巢框，替换的主要蜂箱组件不同。',True),
372:('minor','lost motion写运动损失不够明确，应为空程/失动；齿轮间隙造成反向间隙的关系可辨。',True),
374:('major','crawling缩釉误为爬裂，混淆釉面退缩露胎与开裂的缺陷种类；露胎斑块仍保留。',True),
376:('major','reduced part简化声部误为简谱记谱形式，主要乐谱概念不同。',True),
379:('major','帆脚索sheet误为forestay前支索，将控制帆角度的绳索错换为支撑桅杆的索。原文未要求实施具体索具操作，不推定critical。',True),
388:('major','unless both fail除非两个都失效，否则用备用方案，反转成两个都失效才用备用；明确条件反转。未给系统安全应用背景，不推定critical。',True),
397:('minor','有一段缺失写成a gap，具体段落单位不明；老师接受修改稿与让步关系正确。',True),
}

def main():
    root=Path('runs/takeover-20261002/confirmation')
    sources=read_jsonl('data/independent-20261002-frozen/confirmation.jsonl')
    outputs=read_jsonl(root/'outputs.jsonl')
    if len(outputs)<REVIEWED_COUNT:raise ValueError('Unread/not generated outputs cannot be reviewed')
    reviews=[]
    for index,(source,out) in enumerate(zip(sources,outputs)):
        if index>=REVIEWED_COUNT:break
        if source['id']!=out['id'] or out['source_row_hash']!=fingerprint(source):raise ValueError('Source binding changed')
        verdict,note,retained=JUDGMENTS.get(index,('pass','Codex AI逐条读过实际原文、术语及译文：动作/主体、修饰范围、否定条件、数值和方向符合；合理等义表达接受。',True))
        review=dict(id=out['id'],group_id=source['group_id'],category=source['category'],target_lang=source['input']['target_lang'],
            verdict=verdict,note=note,json_valid=out['json_valid'],language_correct=True,ended=out['ended'],
            has_constraint=bool(source['input'].get('context') or source['input'].get('glossary')),constraint_preserved=retained,
            multisentence=source['multisentence'],source_kind=source['source']['kind'],reviewer='Codex AI',human_acceptance=False,
            source_row_hash=fingerprint(source),generation_hash=fingerprint(out),input_hash=fingerprint(out['input']),output_hash=fingerprint(out['raw']),
            actual_reading=True,method='Personal source/context/actual prediction reading, no semantic string matching')
        reviews.append(review)
    write_jsonl(root/'semantic.jsonl',reviews)
    write_json(root/'review-progress.json',dict(at=now(),reviewed=len(reviews),total=len(sources),reviewer='Codex AI',human_acceptance=False,
        status='complete' if len(reviews)==len(sources) else 'reviewing'))
    if REVIEWED_COUNT==len(sources):
        if len(outputs)!=len(sources):raise ValueError('Full output count required')
        result=evaluate(reviews,stage='confirmation',performance=json.loads(Path('runs/takeover-20261002/decode-final/summary.json').read_text(encoding='utf-8')))
        result.update(at=now(),reviewer='Codex AI',human_acceptance=False,all_gates_passed=result['release_entry_passed'],
            output_hash=fingerprint(outputs),data_hash=fingerprint(sources),full_item_review_complete=True,
            method='All actual source/context/prediction rows personally read; semantic verdicts separate from JSON/direction/EOS; reasonable equivalents accepted; constraints manually assessed')
        write_json(root/'semantic-summary.json',result)
        write_json(root/'review-calibration.json',dict(at=now(),reviewer='Codex AI',human_acceptance=False,
            preliminary_counts=dict(pass_count=310,minor=45,major_including_critical=45,critical=1),
            final_counts=result['counts'],reconsidered_indices=[202,204,263],
            reason='Final source-based review distinguishes generic terms or underspecified existence from explicit substitution/contradiction; accept reasonable equivalents. Three preliminary major judgments become minor.',
            thresholds_changed=False,source_data_changed=False,model_or_runtime_changed=False))
        if not result['all_gates_passed']:
            write_json('runs/takeover-20261002/release-not-executed.json',dict(at=now(),reason='Frozen confirmation entry gates did not pass',
                failed_gates=[k for k,v in result['gates'].items() if not v],confirmation_summary_hash=fingerprint(result),
                release_data_frozen=True,release_outputs_exist=False,no_candidate_changes_after_test=True,release_approved=False))

if __name__=='__main__':main()
