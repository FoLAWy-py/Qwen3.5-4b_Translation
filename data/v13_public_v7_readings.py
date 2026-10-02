"""Individual source-based readings of the 116 frozen auxiliary DEV outputs.

These are evaluation decisions, never training examples or automatic labels.
Both directions were read separately; reference wording is not a pass criterion.
"""

READINGS = """
0193|zh-CN|pass|房间内有电视机的存在关系正确，量词自然。
0193|en|pass|this room和一台电视机均完整保留。
0407|zh-CN|pass|知道、部分人和重视自己的工作均准确。
0407|en|pass|some people和value my work准确对应看重工作。
0085|zh-CN|pass|来此表示感谢的目的完整，中文略直白但可接受。
0085|en|pass|I'm here to thank you准确表达来此感谢的目的。
0015|zh-CN|minor|上错火车的事实正确，但上了错的火车是生硬中文。
0015|en|pass|got on the wrong train准确表达上错火车。
0424|zh-CN|pass|整天和很忙均保留，中文无需显式过去时。
0424|en|pass|very busy all day完整保留一整天都很忙。
0487|zh-CN|pass|汤姆为男人、说话者为女人两个主体性别都正确。
0487|en|minor|两人性别正确，但两个独立分句仅用逗号连接，英语标点有误。
0395|zh-CN|major|没有零钱被改成找零不够，添加了找零场景且改变数量状态。
0395|en|pass|道歉和没有零钱准确，无额外场景。
0082|zh-CN|pass|清楚说话并让别人听清的要求保留。
0082|en|pass|说清楚以便别人听见的目的关系准确。
0366|zh-CN|pass|询问我们需要等待多久准确。
0366|en|pass|how long do we have to wait正确表达等待时长要求。
0527|zh-CN|pass|every other day准确译为每隔一天。
0527|en|pass|洗澡未指定浴缸或淋浴，shower可接受；隔一天频率正确。
0376|zh-CN|minor|站着等车可辨认，但站在等公交车搭配错误。
0376|en|pass|中文无时间标记，当前站着等公交的英文读法成立。
0278|zh-CN|pass|反对任何形式战争的全称范围正确。
0278|en|pass|oppose any form of war准确保留立场与范围。
0266|zh-CN|pass|足球比网球更受欢迎，比较方向正确。
0266|en|pass|football在英式用法对应足球，比较方向正确。
0391|zh-CN|major|before I get home误成你回家之前，回家者由我变为你。
0391|en|pass|我回家之前你先睡觉及不介意的态度准确。
0351|zh-CN|pass|business译为商业成立，巨大成功和女性主体准确。
0351|en|pass|中文事业译career成立，巨大成功与女性主体准确。
0031|zh-CN|pass|东京居住各种各样的人准确。
0031|en|pass|all kinds people living in Tokyo完整保留种类与地点。
0215|zh-CN|pass|盟友关系对称，德国与意大利顺序交换不改变关系，曾经保留。
0215|en|pass|was an ally表示过去盟友关系，两个国家正确。
0342|zh-CN|minor|两张嘴一只耳的条件结构保留，但说话胜过倾听和被赋予生硬，反事实措辞不够自然。
0342|en|pass|假设本应多说少听及两张嘴一只耳准确表达。
0183|zh-CN|pass|感觉不错时去散步的条件和动作完整。
0183|en|pass|散步与感觉良好的条件正确。
0597|zh-CN|pass|汤姆说、认为、可能及唯一必须做此事者均保留。
0597|en|minor|只有自己必须做事和不确定性正确，但it might be only himself句式生硬。
0228|zh-CN|pass|成千上万人想知道答案准确。
0228|en|pass|中文无过去时标记，want to know成立；人数范围正确。
0492|zh-CN|pass|place译你家为普通口语合理读法，保留请求前来。
0492|en|pass|come over to your place自然表达能去你那的请求。
0134|zh-CN|pass|从树上剪下一枝准确，cut允许剪切。
0134|en|pass|cut a branch from the tree保留砍下树枝的动作与来源。
0474|zh-CN|minor|打车请求可辨认，但near here的附近范围弱化为我这里。
0474|en|minor|附近打车被缩为here，地点范围细节弱化。
0254|zh-CN|pass|差点用自己的铅笔戳到说话者眼睛准确。
0254|en|pass|差点、铅笔和我的眼睛均保留，中文未指定铅笔归属。
0189|zh-CN|pass|郊区生活离不开车准确保留必需关系。
0189|en|pass|住郊区需要车的条件与必需性保留。
0588|zh-CN|minor|买冰淇淋换亲吻的条件正确，但买我冰淇淋不合中文搭配。
0588|en|pass|给我买冰淇淋则亲吻的条件、人物正确。
0190|zh-CN|pass|所说的话可在法庭成为不利证据的可能性正确。
0190|en|pass|说的话可用于法庭上对你不利的证据准确。
0292|zh-CN|pass|汤姆确实无任何人帮助完成该事的事实和强调正确。
0292|en|pass|汤姆报告自己无任何帮助做成事情，at all保留强调。
0257|zh-CN|pass|希望对方旅途愉快的祝愿准确。
0257|en|minor|顺利旅行祝愿可辨认，但Have a smooth trip较不自然。
0493|zh-CN|pass|士兵准备战斗准确，未改变主体和状态。
0493|en|pass|士兵已经准备好战斗完整。
0500|zh-CN|minor|好男孩译成好孩子，男性细节弱化。
0500|en|pass|看起来是好男孩的判断与性别正确。
0578|zh-CN|pass|我到达比汤姆稍早，比较主体和幅度正确。
0578|en|pass|here、earlier than Tom和a bit准确对应地点比较幅度。
0477|zh-CN|pass|礼貌询问想试试准确。
0477|en|pass|Would you like to try it自然表达试试的邀请。
0307|zh-CN|pass|他送我不错的礼物，施受关系和评价正确。
0307|en|pass|gave me a nice gift完整保留送礼关系。
0033|zh-CN|minor|再来一瓶的请求准确，但wine译泛称酒，葡萄酒类别弱化。
0033|en|pass|another bottle of wine保留再来一瓶葡萄酒的请求。
0230|zh-CN|pass|再也见不到他准确表达永不再相见。
0230|en|pass|never see him again准确保留否定与未来。
0133|zh-CN|pass|打扰及询问正在做什么保留。
0133|en|minor|正在做什么准确，但遗漏请问的礼貌缓和。
0099|zh-CN|pass|汤姆认为300美元买不齐全部所需物，否定范围和金额正确。
0099|en|minor|金额和买不齐所需物保留，但无时间标记的报告全部转为过去时。
0201|zh-CN|minor|每人一千日元金额和分配正确，但把一千日元每人给了他们语序错误。
0201|en|pass|each of them 1000 yen保留每人的分配及币种。
0072|zh-CN|minor|不肯加薪的拒绝意愿弱化为不会加薪的未来判断。
0072|en|pass|won't give me a raise可表达不肯涨工资的拒绝。
0025|zh-CN|pass|下一个路口左转的地点顺序和方向正确。
0025|en|pass|next corner和turn left准确。
0151|zh-CN|pass|每次来此点同样菜的频率和一致性正确。
0151|en|pass|every time he comes here及same dish准确，always重复但自然可接受。
0226|zh-CN|pass|想谈对方处境准确，礼貌意愿保留。
0226|en|pass|I'd like to talk about your situation准确。
0565|zh-CN|pass|看起来像刚失去最好的朋友，比较和时间准确。
0565|en|pass|look like just lost your best friend保留外观比喻。
0227|zh-CN|minor|桌上存在水果正确，但orange译橘子而非橙子，具体类别不精确。
0227|en|pass|桌上一个橙子的数量、类别、地点完整。
0011|zh-CN|pass|包裹上的地址错误准确。
0011|en|pass|package address is wrong准确保留地址归属。
0135|zh-CN|pass|告诉妻子不要冲动购物，人物和否定正确。
0135|en|pass|told his wife not to shop impulsively完整。
0306|zh-CN|major|If I were you被写成如果你是我在，假设角色方向错误且句子残缺。
0306|en|pass|If I were you, I would go准确保留假设与选择。
0402|zh-CN|pass|感谢对方来见面准确。
0402|en|pass|thank you for coming to see me准确表达来见我的谢意。
0104|zh-CN|pass|无家可归时期经常睡在那张长椅的时间关系完整。
0104|en|pass|无家可归时间、频率及指定长椅完整。
0442|zh-CN|pass|询问论文是否写完准确。
0442|en|pass|中文论文未指定学位论文，paper成立，完成状态保留。
0052|zh-CN|minor|不要对批评太敏感的意思可辨，但敏感于批评搭配生硬。
0052|en|pass|Don't be too sensitive to criticism准确。
0174|zh-CN|pass|定期看牙医因而很少牙疼的因果与频率保留。
0174|en|pass|regularly和rarely准确保留看牙医及牙痛的因果频率。
0205|zh-CN|pass|房子离公园很近，空间关系正确。
0205|en|pass|家离公园很近的程度和地点正确。
0507|zh-CN|minor|完全不感兴趣艺术中文语序有误，否定意思仍可辨。
0507|en|pass|not at all interested in art准确保留完全否定。
0170|zh-CN|pass|一整天在农场工作的时间与地点准确。
0170|en|pass|worked all day on the farm准确。
0248|zh-CN|pass|年轻但有经验的转折准确。
0248|en|pass|young but experienced保留年轻与有经验的转折。
"""
