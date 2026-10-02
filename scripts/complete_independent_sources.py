"""Bind the root Codex AI's pre-output source edits; never reads model outputs."""
import json
from pathlib import Path
from witrans_tools.common import read_jsonl,write_json,write_jsonl,fingerprint,now
from scripts.freeze_independent_tests import normalize,retrieve_neighbors

HARD={
'confirmation-hard-000':{'zh':'钢琴家注意到按下琴键后击弦机回弹缓慢，这影响了断奏的清晰度。'},
'confirmation-hard-002':{'zh':'水手检查了系在小艇船头的缆绳结，并确认其牢固。'},
'confirmation-hard-003':{'en':"The carpenter checked that the plane's sole was flat before cutting the wood.",'zh':'木匠在切削木料前检查了刨子的底面是否平整。'},
'confirmation-hard-005':{'zh':'裁缝调整了衣物的松量，以改善腰线部位的合身度。'},
'confirmation-hard-007':{'en':'The surveyor held the staff beside the benchmark and read its graduations.','zh':'测量员在基准点旁扶着水准尺，读取尺上的刻度。'},
'confirmation-hard-008':{'zh':'装订工解释了书册的书帖如何折叠并缝入封皮。'},
'confirmation-hard-009':{'zh':'国际象棋棋手讨论了被牵制的棋子为何不能移动，否则王会暴露在攻击下。'},
'confirmation-hard-011':{'en':'The brewer takes a sample of the mash to check its consistency.','zh':'酿酒师取了一份糖化醪样品，检查其稠度。'},
'confirmation-hard-013':{'zh':'律师向客户解释送达与实际知悉通知之间的区别。'},
'confirmation-hard-016':{'en':"The typographer inspected the metal type's shoulder before placing it in the forme.",'zh':'排字工在将金属活字放入印版前检查了它的字肩。'},
'confirmation-hard-017':{'en':"The clockmaker listened to the movement's beat through a stethoscope to check its regularity.",'zh':'钟表匠用听诊器听机芯的滴答声，检查其节奏是否规律。'},
'confirmation-hard-018':{'en':"The glassblower polished the pontil mark on the base without changing the rim's contour.",'zh':'玻璃吹制工打磨了底部的铁棒痕，却没有改变口沿的轮廓。'},
'confirmation-hard-019':{'zh':'攀岩者确认凸轮塞的各个凸轮瓣都充分接触岩面。'},
'confirmation-hard-020':{'zh':'小提琴手解释弓根装置调节弓毛张力，从而改善控制。'},
'confirmation-hard-024':{'en':"The mathematician identifies the unit element as the ring's multiplicative identity, not a numerical measurement.",'zh':'数学家指出，这个环的单位元是乘法单位元，而不是数值度量。'},
'confirmation-hard-026':{'zh':'机械师解释，齿轮传动中的反向间隙是齿轮间的间隙造成的运动空程。'},
'confirmation-hard-027':{'en':'The ceramic artist noticed crawling in the glaze, leaving bare patches on the fired surface.','zh':'陶艺家注意到釉面发生缩釉，在烧成后的表面留下了露胎斑块。'},
'confirmation-hard-029':{'zh':'水手解释，帆脚索控制帆的角度，而升降索用于升起或降下帆。'},
'confirmation-hard-031':{'en':"Smith rejected Jones's amendment, not the amendment Smith himself had submitted.",'zh':'史密斯拒绝的是琼斯的修正案，而不是史密斯自己提交的修正案。'},
'confirmation-hard-034':{'en':'The technician uses the fallback unless both sensors fail.','zh':'除非两个传感器都失效，否则技术人员使用备用方案。'},
'release-hard-040':{'zh':'图书管理员数着剩余的书帖，解释说有些部分被拆成了更小的部分。'},
'release-hard-041':{'en':"The dentist recorded the crown's margin position without changing the restoration.",'zh':'牙医记录了牙冠边缘的位置，却没有改变修复体。'},
'release-hard-044':{'zh':'水手演示了风如何把船横向推移，产生风致侧漂。'},
'release-hard-045':{'zh':'金属工人沿着焊趾查看，确保焊缝与基材正确结合。'},
'release-hard-046':{'en':'The photographer notices a slight change in the field of view while focusing and calls it lens breathing.','zh':'摄影师注意到对焦时视野略有变化，称之为镜头呼吸效应。'},
'release-hard-048':{'en':'The musician marked the pickup notes before the first full bar.','zh':'音乐家标记了第一个完整小节之前的弱起音符。'},
'release-hard-053':{'zh':'一位家长提醒孩子别人曾许下的诺言。'},
'release-hard-055':{'zh':'尽管没有得到支持背书，主管仍批准了草案。'},
'release-hard-059':{'zh':'乘客澄清了谁与谁交换了票。'},
'release-hard-060':{'en':'The conservator documents the uneven ground layer and refuses to apply fresh paint over the original.','zh':'修复师记录了不均匀的底层，并拒绝在原作上覆盖新颜料。'},
'release-hard-061':{'en':'The angler adds split shot and records how much lower the float sits in the water.','zh':'钓鱼人加上开口铅坠，并记录浮漂在水中降低了多少。'},
'release-hard-063':{'zh':'鞋匠用修边刀小心切去靴子沿条上多余的皮革，确保修边光滑。'},
'release-hard-065':{'zh':'石匠检查墙体的组砌方式，确保它符合所要求的结构与美观标准。'},
'release-hard-066':{'zh':'养蜂人注意到换王王台正在形成，表明蜂群正在自然更换蜂王。'},
'release-hard-068':{'zh':'锁匠发现钥匙与锁内固定隔挡之间的间隙太小，无法转动。'},
'release-hard-070':{'en':'The caretaker separately tells Anna and Ben that the other has not been granted access.','zh':'看管员分别告诉安娜和本，对方尚未获得进入许可。'},
'release-hard-073':{'en':"The shopkeeper refunded the jacket's price but kept the deposit for the headphones.",'zh':'店主退还了夹克的价款，却保留了耳机的押金。'},
'release-hard-080':{'en':'The potter adds a measured amount of deflocculant to the slip and records the water content separately.','zh':'陶艺家向泥浆中加入定量的解絮凝剂，并单独记录含水量。'},
'release-hard-081':{'zh':'乐手调整小号的吹口管，以改善音准和气流。'},
'release-hard-082':{'en':"The falconer checks the bird's jesses for wear before the hunt.",'zh':'驯鹰人在狩猎前检查鸟的脚绊是否磨损。'},
'release-hard-083':{'zh':'制表师辨认出擒纵机构中的擒纵瓦，它对计时精度至关重要。'},
'release-hard-084':{'en':'The theater technician explains how the masking legs hide the offstage crew.','zh':'剧场技术员解释侧幕如何遮住台侧的工作人员。'},
'release-hard-085':{'zh':'裁缝增加了袖山的松量，以确保穿着舒适、合身平顺。'},
'release-hard-087':{'en':'The surveyor compares the true bearing with the magnetic bearing and records the declination separately.','zh':'测量员比较真方位角与磁方位角，并单独记录磁偏角。'},
'release-hard-090':{'en':'The researcher says that replication requires following the study protocol, not merely repeating the outcome.','zh':'研究员说，重复实验需要遵循研究方案，而不只是重复结果。'},
'release-hard-093':{'zh':'读者注意到引语中暗藏的话里带刺的恭维。'},
'release-hard-094':{'en':'The musician clarifies whether to mute the left-hand or right-hand part.','zh':'音乐家澄清应将左手声部还是右手声部静音。'},
'release-hard-096':{'zh':'图书管理员在手稿中寻找缺失的一张书叶，而不是整本书。'},
'release-hard-099':{'zh':'裁判报告的是一分被判无效，而不是球员退赛。'},
}

# Explicit bilingual additions written and checked by root Codex AI, before outputs.
# Each line: term English | Chinese | additional EN sentence | its ZH reference.
SUPPLEMENTS={
'daily':'''radiator|散热器|The silver earring belonged to her aunt.|那枚银耳环属于她的姨妈。
sewing machine|缝纫机|The first repair was scheduled for Monday.|第一次修补安排在周一。
video audition|视频试镜|He put the camera on the upper shelf.|他把摄像机放在上层架子上。
mugs|马克杯|The blue sticker marked the larger one.|蓝色贴纸标记了较大的那个。
barber|理发师|He showed the barber a photograph from last summer.|他给理发师看了一张去年夏天的照片。
doorbell|门铃|The tune reminded him of his former apartment.|那段曲调让他想起以前的公寓。
felt pads|毛毡垫|She tested the chair beside the window.|她在窗边试了试那把椅子。
fitted sheet|床笠|They placed it beside the clean pillowcases.|他们把它放在干净的枕套旁。
knitting circle|编织聚会|The new meeting will take place in Room Four.|新的聚会将在四号房间举行。
large-print instructions|大字版说明书|The box also contained six wooden counters.|盒子里还有六枚木制计数筹码。
wall clocks|挂钟|One clock hung above the kitchen doorway.|其中一座钟挂在厨房门口上方。
secondhand record|二手唱片|The sleeve still carried the previous owner's initials.|唱片封套上仍有前任主人的姓名首字母。
ringtone|铃声|Her sister had changed it the previous evening.|她姐姐是在前一天晚上更换它的。
framed drawing|装裱的画|The drawing showed a small red boat.|画上是一艘红色小船。
chestnuts|栗子|The basket was left beside the garden gate.|篮子被放在花园门旁。
cactus|仙人掌|Her friend wrote the request on a yellow card.|她的朋友把这个请求写在一张黄色卡片上。
hedgehog|刺猬|The feeding area was behind the shed.|喂食区在棚屋后面。
silkworms|蚕|The enclosure remained on the classroom table.|饲养箱一直放在教室的桌上。
puppet show|木偶戏|Ben chose to play the forest messenger.|本选择扮演森林信使。
treasure hunt|寻宝活动|The final clue was inside an empty hat.|最后一条线索在一顶空帽子里。''',
'travel':'''stroller|婴儿车|The passenger kept the child's blanket on their lap.|乘客把孩子的毯子放在自己膝上。
floating bridge|浮动桥|The sign was attached to the southern railing.|标志牌装在南侧栏杆上。
binoculars|望远镜|The trail marker was shaped like a white triangle.|步道标记是一个白色三角形。
cable car|缆车|A green ribbon identified the traveler's suitcase.|一条绿色丝带标明了旅行者的行李箱。
lighthouse|灯塔|The visitor arrived shortly before sunset.|游客在日落前不久抵达。
hammock|吊床|The campsite number was written on a wooden post.|营位编号写在一根木柱上。
causeway|堤道|The printed timetable was dated that morning.|印好的时刻表标着当天早上的日期。
visor|面屏|The skier requested a blue helmet rather than a red one.|滑雪者要求蓝色头盔，而非红色头盔。
berth|床位|A narrow curtain separated it from the aisle.|一幅窄帘将它与过道隔开。
oasis|绿洲|The guide would meet the group beside a stone well.|导游将在一口石井旁与团队会合。
peat-bog boardwalk|泥炭沼泽木栈道|The hiker photographed the sign before leaving.|徒步者在离开前拍下了标志牌。
audio tour|音频导览|The clerk offered a small pair of headphones.|职员提供了一副小耳机。
shrine keeper|神社管理员|The visitor carried a pencil but no paints.|游客带着铅笔，却没有带颜料。
spiral stairway|螺旋楼梯|The ramp ended beside the eastern entrance.|坡道的尽头在东侧入口旁。
chain-fitting|安装防滑链|The marked area was beyond the last fuel station.|标出的区域在最后一座加油站之后。
herb terrace|香草露台|Two empty benches stood beside the planters.|种植箱旁有两张空长椅。
dry bag|防水袋|The attendant checked its yellow buckle.|工作人员检查了它的黄色搭扣。
sketchbook|素描本|A drawing of the harbor appeared on its first page.|它的第一页上有一幅港口画。
headlamp|头灯|The guide counted three visitors at the entrance.|导游在入口处数了三位游客。
saddlebag|鞍袋|The locker key was tied to a red cord.|储物柜钥匙系在一根红绳上。''',
'food':'''sourdough|酸种面团|The baker covered it with a striped towel.|烘焙师用一条条纹毛巾盖住它。
leeks|韭葱|She reserved the green ends for another dish.|她把绿色的末端留作另一道菜。
cardamom|豆蔻|The rice pudding was intended for four guests.|米布丁是为四位客人准备的。
chestnuts|栗子|The serving bowl had a wide yellow rim.|盛放用的碗有一道宽宽的黄色碗沿。
egg whites|蛋白|The cake tin was already lined with parchment.|蛋糕模已经铺上了烘焙纸。
artichoke|洋蓟|He placed the trimmed heart in a blue dish.|他把处理好的洋蓟心放进蓝色盘子。
shiitake|香菇|The cook kept the soaking water in a separate jug.|厨师把浸泡水留在另一个壶里。
lotus leaves|荷叶|The parcel was tied with a short cotton string.|包裹用一根短棉线扎好。
risotto|烩饭|She set aside a portion for her neighbor.|她给邻居留了一份。
radishes|萝卜|The serving plate was oval rather than round.|装盘用的盘子是椭圆形，而非圆形。
meat filling|肉馅|He wrote the batch number on the container.|他把批次号写在容器上。
cores|果核|The empty centers were filled with raisins.|挖空的中心填入了葡萄干。
onion tart|洋葱挞|She used the smaller of the two baking trays.|她用了两只烤盘中较小的一只。
zest|皮屑|The mixture was stored in a jar with a white lid.|混合物装在一个白盖罐子里。
almond flour|杏仁粉|The sieve rested across a large mixing bowl.|筛子横放在一个大搅拌碗上。
dough|面团|The cook marked the resting time on the board.|厨师把静置时间标在板上。
quinoa|藜麦|A purple cloth lay beside the empty bowl.|空碗旁放着一块紫色布。
eggplant|茄子|The cook waited beside the oven with a metal tray.|厨师拿着金属托盘，在烤箱旁等候。
fish fillet|鱼柳|The removed bones went into a small dish.|取出的鱼骨被放进小碟里。
platter|餐盘|The guests were already seated near the fireplace.|客人们已经在壁炉附近就座。''',
'academic':'''pollen|花粉|The first core came from the northern trench.|第一个岩芯来自北侧探沟。
lexical stress|词汇重音|The recording contained twelve short words.|录音包含十二个短词。
radial velocity|径向速度|The spectrum was recorded on Tuesday night.|光谱是在周二晚上记录的。
cleavage|解理|The sample arrived in a numbered cardboard box.|样本装在一个编号纸盒中送达。
recognition memory|再认记忆|The second task used photographs instead of sounds.|第二项任务使用照片，而非声音。
authorship|作者身份|The diary's cover carried no name.|日记封面上没有姓名。
selectivity|选择性|The laboratory assigned a separate label to each product.|实验室给每种产物分配了单独的标签。
group velocity|群速度|The diagram used blue arrows for the second wave packet.|图中用蓝色箭头表示第二个波包。
sediment|沉积物|The sample was collected from the western slope.|样本采集自西侧坡面。
spectrometer|光谱仪|The baseline file was saved under a new name.|基线文件以一个新名字保存。
unconformity|不整合面|The field notebook included a sketch of the contact.|野外笔记本里有一幅接触面的草图。
sufficiency|充分性|The example used three numbered propositions.|这个例子使用了三个编号命题。
recessive allele|隐性等位基因|The pedigree chart contained five generations.|系谱图包含五个世代。
unique solution|唯一解|The proof occupied the final page of the appendix.|证明写在附录的最后一页。
marginal note|旁注|The note was written in a darker ink.|旁注用的是颜色更深的墨水。
species richness|物种丰富度|The survey covered two neighboring valleys.|调查覆盖了两个相邻山谷。
opportunity-cost|机会成本|The workshop would reopen on Thursday.|工坊将在周四重新开放。
damping coefficient|阻尼系数|The matrix occupied the second table in the report.|矩阵位于报告中的第二张表。
Mandarin|普通话|The shift occurred just after the third speaker's question.|切换发生在第三位说话者提问之后。
enantiomeric excess|对映体过量|The sample label listed the measurement date.|样本标签列出了测量日期。''',
'hard':'''action|击弦机|The pianist wrote the observation in the repair log.|钢琴家把这一观察写进维修记录。
forme|印版|The bed had a small scratch near its left edge.|印床左边缘附近有一道小划痕。
painter|缆绳|The knot was beside a faded blue stripe.|绳结旁有一道褪色的蓝条纹。
sole|底面|The plane belonged to the carpenter's former teacher.|刨子属于木匠以前的老师。
table|台面|The stone was placed on a black cloth.|宝石放在一块黑布上。
ease|松量|The garment would be fitted again on Friday.|这件衣物将在周五再次试穿。
sport|变异枝条|The gardener attached a white label to that branch.|园丁给那根枝条挂上了白色标签。
staff|水准尺|The reading was entered in the second column.|读数填在第二栏中。
gatherings|书帖|The count was recorded on a separate slip.|数量记在一张单独的纸条上。
margin|边缘|The patient asked for a copy of the record.|患者要求一份记录副本。
dormant buds|休眠芽|The branch was labeled with a green ribbon.|枝条用一条绿色丝带标记。
normally closed contact|常闭触点|The diagram showed the contact in its resting state.|图中显示了触点的静止状态。
leeway|风致侧漂|The demonstration took place inside the harbor.|演示在港内进行。
toe|焊趾|The worker circled one spot with a blue marker.|工人用蓝色记号笔圈出一个位置。
lens breathing|镜头呼吸效应|The test used a stationary subject near the window.|测试使用了窗边一个静止的拍摄对象。
recovery|回收率|The drilling log contained the recovered length.|钻探记录中列出了取回的长度。
pickup notes|弱起音符|The penciled marks appeared above the staff.|铅笔标记位于五线谱上方。
saddle compensation|琴码补偿|The luthier compared two recorded notes afterward.|制琴师随后比较了两个录下的音符。
record|记录|The omitted record was dated June second.|被省略的记录标着六月二日的日期。
repairs|维修|The tenant kept the promise in a signed letter.|租户把这一承诺写在一封签名信中。''',
}

def main():
    root=Path('data/independent-20261002-v2-drafts')
    entries=read_jsonl(root/'local-source-review-packet.jsonl')
    if len(entries)!=500:raise ValueError('All 500 reviewed groups required')
    changes=dict(HARD)
    for name in ('local','travel','food','academic'):
        changes.update(json.loads(Path(f'data/independent-source-{name}-corrections-20261002.json').read_text(encoding='utf-8'))['changes'])
    supplement={category:[line.split('|') for line in text.splitlines()] for category,text in SUPPLEMENTS.items()}
    sets={'confirmation':[],'release':[]}
    for entry in entries:
        card=entry['card'];scene=dict(entry['scene']);scene.update(changes.get(entry['id'],{}))
        category=card['category'];index=card['index'];stage=card['stage'];added=None
        if index<8:added=supplement[category][index]
        elif 40<=index<52:added=supplement[category][8+index-40]
        if added:
            term,zhterm,en2,zh2=added
            if term.casefold() not in scene['en'].casefold() or zhterm not in scene['zh']:
                raise ValueError('Inapplicable pre-output term: '+entry['id']+' '+term+'/'+zhterm)
            scene['en']+=' '+en2;scene['zh']+=zh2
        for direction,key,other in (('zh-CN','en','zh'),('en','zh','en')):
            inp=dict(text=scene[key],target_lang=direction)
            if added:inp['glossary']={term:zhterm} if key=='en' else {zhterm:term}
            row=dict(id=entry['id']+'-'+direction,group_id=entry['id'],category=category,input=inp,
                output={'translation':scene[other]},multisentence=len([x for x in __import__('re').split(r'[.!?。！？]',scene[key]) if x.strip()])>=2,
                source=dict(kind='synthetic_AI',name='Original scene cards with blind bilingual AI drafting and Codex AI corrections',
                    license='Original synthetic material for local evaluation; no third-party quoted text; DeepInfra terms checked 2026-10-02, section 7',
                    attribution='Original scenario cards: Codex AI; bilingual draft/advisory review: DeepInfra Qwen/Qwen3-32B; source/reference corrections and final review: Codex AI',
                    origin_id=entry['id'],evaluation_allowed=True,training_allowed=False,human_authored=False,
                    card=card,author_metadata=entry['author_metadata'],advisory_audit_metadata=entry['audit_metadata'],provider_revision='not exposed',
                    local_revision=changes.get(entry['id'],{}),added_bilingual_fact=added,candidate_outputs_seen=False))
            content={k:row[k] for k in ('input','output','source','group_id','category')}
            row['source_review']=dict(reviewer='Codex AI',human_acceptance=False,status='approved',at=now(),
                content_hash=fingerprint(content),license_checked=True,attribution_checked=True,reference_checked=True,
                context_checked=True,ambiguity_resolved=True,historic_family_checked=True,overlap_adjudications={},
                note='Root Codex AI personally read both languages, card and closest historical candidates for this group before outputs. '+
                     'Generated contexts and claimed multisentence labels discarded; actual sentences counted. '+
                     ('Specific equivalence/ambiguity correction recorded. ' if entry['id'] in changes else '')+
                     ('Applicable bidirectional term and additional bilingual fact personally authored and checked.' if added else 'No unreviewed extra context.'))
            sets[stage].append(row)
    for stage,rows in sets.items():write_jsonl(root/(stage+'-reviewed.jsonl'),rows)
    old=json.loads(Path('runs/takeover-20261002/historic-source-catalog.json').read_text(encoding='utf-8'))['texts']
    prior={};matches=[]
    for stage,rows in sets.items():
        for row in rows:
            for text in (row['input']['text'],row['output']['translation']):
                key=normalize(text)
                if key in old:raise ValueError('Exact historical overlap '+row['id'])
                candidates=retrieve_neighbors(key,old,{k:v for k,v in prior.items() if v!=row['group_id']})
                if candidates:matches.append(dict(id=row['id'],text=text,retrieval_hash=fingerprint(candidates),candidates=candidates))
                prior[key]=row['group_id']
    write_jsonl(root/'final-overlap-review.jsonl',matches)
    write_json(root/'root-source-review-receipt.json',dict(at=now(),groups=500,rows=1000,reviewer='Codex AI',human_acceptance=False,
        candidate_outputs_seen=False,terms_url='https://deepinfra.com/terms',changes=changes,all_category_packets_read=True,
        contextual_terms={k:sum(bool(r['input'].get('glossary')) for r in v) for k,v in sets.items()},
        actual_multisentence={k:sum(r['multisentence'] for r in v) for k,v in sets.items()},unresolved_retrieval_rows=len(matches),
        synthetic_limitation='Independent source families and held-out candidate use, not independent human population sampling or human acceptance'))

if __name__=='__main__':main()
