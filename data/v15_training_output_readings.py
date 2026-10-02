"""Source-based TRAIN audit, not heldout accuracy or automatic training authorization."""
READINGS = '''
public-short-0451-zh-CN|pass|你们办公室几点下班是询问单位下班时间的自然口语表达，与让你回家的普通工作语境相符。
public-short-0054-zh-CN|minor|找钥匙地点的意思保留，但now的回想语气变为现在时间，丢失我把钥匙放哪的回忆口吻。
v10-daily-19-zh-CN|major|专业检查后再粉刷的条件保留，但water damage水造成的损坏被缩成水渍，损害范围与检查目的不准确。
public-short-0208-zh-CN|minor|浴缸中让身体暖和的意思正确，但暖和了自己不合中文自然搭配。
public-short-0066-zh-CN|pass|承诺会来而实际没来，承诺与实际行为的对照准确。
public-short-0062-zh-CN|pass|下雨导致不能在户外打网球，因果、否定和地点完整。
public-short-0173-en|pass|Many flowers bloom in March自然准确；与参考come out不同不构成错误，高NLL是措辞差异的例子。
public-short-0357-en|pass|教师恢复健康译has recovered from illness在普通恢复健康语境下成立，未因与参考措辞不同而判错。
public-short-0337-en|pass|He can't come准确表达不可能来，不要求匹配冗长的possibility参考句式。
public-short-0275-en|pass|他的问题使我非常困惑，施受关系与程度准确。
public-short-0510-en|pass|我认为汤姆不太可能当选，判断主体和概率程度正确。
public-short-0345-en|pass|今天太热导致没心情学习，时间、原因与意愿完整。
'''

REMAINING_READINGS = '''
v10-travel-04-zh-CN|minor|较短更陡、仅依团队偏好且非封路的条件保留，但仅当团体更喜欢它的后置结构生硬。
v5-seed-005-zh-CN|major|funicular地面缆车误为人行道，交通设施种类错误，即使轨道说明仍保留也不正确。
public-short-0411-zh-CN|pass|一周后到达准确表达从今天起一周的时间。
v10-travel-09-zh-CN|minor|大箱寄存和携带小背包正确，但单数traveler译他们，smaller的比较意味也弱化。
v10-travel-20-zh-CN|major|额外一英里的数值一遗漏成多走了英里，距离信息不完整；无障碍也泛化可达性。
v8-multi-013-zh-CN|minor|两班只有晚班周日运行及先核对日期准确，但connection译连接，接续班次表达不自然。
public-short-0456-en|pass|还有数间空房准确，中文未标过去时，are成立。
public-short-0462-en|minor|三天后再来准确，但请的礼貌措辞遗漏。
v6-travel-02-en|pass|礼貌确认本次短途旅行行李限额，baggage allowance和short trip正确。
public-short-0503-en|pass|女性乘出租车去博物馆的行为、交通工具和目的地准确。
public-short-0463-en|pass|认为取得驾驶证不会有困难，判断主体和否定程度自然表达。
v4-mining-005-en|pass|换乘站按指示去北行有轨电车且不出站，方位、行动、否定完整。
v2-train-058-zh-CN|major|查询高汤stock的成分被改为这道菜，成品没有贝类也改成没有添加，询问对象和条件改变。
v10-food-13-zh-CN|major|diner食客误成餐厅，chicken stir-fry炒鸡肉误成鸡肉炒饭，角色和食物错误。
v10-food-15-zh-CN|minor|后句保留糖与任何甜味剂的区别，但unsweetened译无糖，未准确表达未增甜的状态。
v4-mining-011-zh-CN|minor|先留煮面水再沥面顺序正确，但松散酱汁搭配不自然，应表达将酱汁调稀。
v10-food-02-zh-CN|critical|严重过敏场景must not be served at all反转为不能完全不提供、只放一边，禁令反转；diner也误成餐厅。
v4-mining-010-zh-CN|minor|轻拌蛋白入面糊的动作正确，但knock out air译破坏空气不自然，没有清楚表达避免消泡。
v10-food-20-en|pass|泡发干豆与罐装熟豆两选项、分别处理和不混用的要求保留，当前时态在中文无明确过去标记下成立。
public-short-0210-en|minor|需要打包容器的意思保留，但packing box偏包装箱，不是食品打包盒的自然说法。
public-short-0398-en|pass|Can I have the bill please自然保留索要账单的礼貌请求，不要求check措辞。
public-short-0081-en|pass|她不想吃午饭准确，中文未标过去时，无须复制参考was。
v10-food-16-en|minor|先冷却酥皮再加入冷馅、防潮保酥顺序正确，但果酱jam泛化成fruit filling果馅。
v4-mining-009-en|pass|漏勺可以指网状滤勺，strainer用于捞饺子成立，微沸与捞取动作正确。
v10-academic-01-zh-CN|major|空缺和零值区别保留，但could lead to incorrect analysis的可能性改成就会导致的确定性。
v9-constraint-023-zh-CN|pass|区间(0,1]的记号及零排除、一包含准确。
public-short-0021-zh-CN|pass|美国许多州废除死刑的过去事实完整，译文未新增现时法律结论。
public-short-0309-zh-CN|pass|革命带来许多变化的因果和数量范围准确。
v8-multi-031-zh-CN|major|词表边界保留度使用正确，但数值指标measure误为措施，研究对象类型改变。
public-short-0455-zh-CN|pass|个人因富裕或贫穷而有不同看法，条件关系完整。
v2-train-105-en|pass|先去重再分数据、测试集全程独立，动作顺序和隔离要求正确。
public-short-0504-en|pass|一英里约1600米，单位、数值与近似性准确。
public-short-0083-en|pass|询问4乘6是多少，两个数字和运算正确。
public-short-0048-en|pass|简单英语使孩子也能理解，原因和递进主体完整。
v10-academic-08-en|pass|充分条件不等于唯一条件、条件失败而定理成立不反驳充分性，逻辑关系准确。
v6-academic-01-en|pass|样本均值可能不同于总体均值，但用于估计总体均值，可能性和用途正确。
v10-hard-10-zh-CN|minor|假设失败需重新开始、旧笔记有帮助的大意保留，但重复起点使从头再来与重新着手依据的区别不自然。
public-short-0080-zh-CN|major|frog in my throat的嗓音沙哑习语误为喉咙内有实体青蛙，含义错误。
public-short-0573-zh-CN|minor|钱很快离开傻瓜的寓意仍可辨，但直译傻瓜和他钱分开生硬，未自然表达守不住钱。
v5-seed-020-zh-CN|major|明确戏剧演员语境下cast演员阵容误成布景，忽略语境消歧。
v8-multi-035-zh-CN|major|明确决策而非体育语境下ball in your court误成球归你所有，决策责任含义丢失。
v10-hard-13-zh-CN|pass|sitting on it根据语境译拖延，不回复、不转交主管及阻碍项目完整。
public-short-0175-en|pass|暴风雨后恢复平静的时序准确，未强求参考谚语式句型。
v10-hard-09-en|pass|一致不向外界共享内部稿、界限维护安全和专注，否定和用途正确。
v10-hard-03-en|pass|新员工学习、导师仍最终审批、经验者掌握决策，角色边界完整。
public-short-0513-en|pass|歉意与现在很忙准确表达，busy代替hands full属合理意译。
v10-hard-06-en|minor|spot on表达一针见血、指出主要问题正确，但下一步关注目标泛为next step。
v2-replay-v1-hard-008-02-en|pass|4条消息、翻译引用短语而不执行的区别完整，引用内容作为数据保留。
v9-constraint-006-zh-CN|minor|尚未领取、仅收到就绪通知和代取物品的条件保留，但洗衣尚未被领取是生硬的衣物名词表达。
public-short-0168-zh-CN|pass|提前致谢的意思正确，正式措辞可以接受。
public-short-0356-zh-CN|pass|可能发生交通事故的推测保留，中文可能在本句是常见对应表达。
public-short-0447-zh-CN|pass|老人一只眼睛失明，主体、数量和状态准确。
'''
