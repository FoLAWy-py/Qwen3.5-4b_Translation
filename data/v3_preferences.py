"""New semantic contrasts. Every negative changes one explicitly stated fact."""

# category, English, Chinese, wrong Chinese, wrong English, error, context
TRAIN = [
("daily", "Maya lent the headphones to Leo.", "玛雅把耳机借给了利奥。", "利奥把耳机借给了玛雅。", "Leo lent the headphones to Maya.", "借出者与借入者颠倒", ""),
("daily", "I didn't promise to repair it; I promised to inspect it.", "我没答应修好它，我答应的是检查它。", "我答应修好它，没答应检查它。", "I promised to repair it, not to inspect it.", "承诺对象及否定颠倒", ""),
("daily", "The handle is loose, but it has not broken off.", "把手松了，但还没有断掉。", "把手松了，而且已经断掉了。", "The handle is loose and has broken off.", "断裂事实的否定丢失", ""),
("daily", "Please keep the old address until the new one has been confirmed.", "新地址确认之前，请保留旧地址。", "新地址确认之后，请保留旧地址。", "Please keep the old address after the new one has been confirmed.", "之前与之后颠倒", ""),
("daily", "She sent me a photo of the damaged hinge, not the whole cupboard.", "她发给我的是损坏的铰链的照片，不是整个橱柜的照片。", "她发给我的是整个橱柜的照片，不是损坏的铰链的照片。", "She sent me a photo of the whole cupboard, not the damaged hinge.", "照片对象颠倒", ""),
("daily", "I used the guest account because my own account was locked.", "我用了访客账号，因为我自己的账号被锁定了。", "我用了自己的账号，因为访客账号被锁定了。", "I used my own account because the guest account was locked.", "账号身份颠倒", ""),
("daily", "The cupboard is empty, although the drawer beneath it is full.", "柜子是空的，但下面的抽屉是满的。", "柜子是满的，但下面的抽屉是空的。", "The cupboard is full, although the drawer beneath it is empty.", "两个容器的状态颠倒", ""),
("daily", "You may read the note, but please don't forward it.", "你可以读这张便条，但请不要转发。", "你可以读这张便条，也可以转发。", "You may read the note and forward it.", "转发禁令变成许可", ""),
("travel", "Passengers must show their tickets before entering the waiting room.", "乘客进入候车室之前必须出示车票。", "乘客进入候车室之后必须出示车票。", "Passengers must show their tickets after entering the waiting room.", "检票顺序改变", ""),
("travel", "This pass covers the shuttle, but not the cable car.", "这张通行证包含接驳车，但不包含缆车。", "这张通行证包含缆车，但不包含接驳车。", "This pass covers the cable car, but not the shuttle.", "适用交通工具颠倒", ""),
("travel", "The platform is above the concourse, not below it.", "站台在大厅上方，不在下方。", "站台在大厅下方，不在上方。", "The platform is below the concourse, not above it.", "垂直位置颠倒", ""),
("travel", "If you cancel after departure, the fare cannot be refunded.", "如果你在出发之后取消，票款无法退还。", "如果你在出发之前取消，票款无法退还。", "If you cancel before departure, the fare cannot be refunded.", "退款条件时间改变", ""),
("travel", "The gate number has changed, but the departure time has not.", "登机口编号变了，但出发时间没有变。", "登机口编号没变，但出发时间变了。", "The gate number has not changed, but the departure time has.", "变化对象颠倒", ""),
("food", "Please serve the custard in a small jug, not on a plate.", "请把蛋奶酱装在小壶里，不要放在盘子上。", "请把蛋奶酱放在盘子上，不要装在小壶里。", "Please serve the custard on a plate, not in a small jug.", "盛装容器颠倒", ""),
("food", "The broth contains shellfish even if the visible pieces are removed.", "即使去掉看得见的食材块，汤里仍含有贝类成分。", "只要去掉看得见的食材块，汤里就不含贝类成分了。", "The broth contains no shellfish once the visible pieces are removed.", "去掉可见食材被误认为去除过敏成分", ""),
("food", "Could you give me a spoon for the dessert and a fork for the salad?", "能给我一把吃甜点的勺子和一把吃沙拉的叉子吗？", "能给我一把吃甜点的叉子和一把吃沙拉的勺子吗？", "Could you give me a fork for the dessert and a spoon for the salad?", "餐具用途颠倒", ""),
("food", "Only one of the three sandwiches should have mustard.", "三个三明治中只应有一个放芥末。", "三个三明治中应有两个放芥末。", "Two of the three sandwiches should have mustard.", "指定份数从一变二", ""),
("food", "The crust contains butter, but the filling does not.", "外皮含黄油，但馅料不含。", "馅料含黄油，但外皮不含。", "The filling contains butter, but the crust does not.", "成分所在部分颠倒", ""),
("food", "We need the larger saucepan, not the deeper frying pan.", "我们需要较大的深煮锅，不是较深的煎锅。", "我们需要较深的煎锅，不是较大的深煮锅。", "We need the deeper frying pan, not the larger saucepan.", "锅具类型颠倒", ""),
("academic", "The capacitor stores charge; it does not generate charge.", "电容器储存电荷，不产生电荷。", "电容器产生电荷，不储存电荷。", "The capacitor generates charge; it does not store charge.", "储存与产生颠倒", ""),
("academic", "The dependent variable is mass, while time is the independent variable.", "因变量是质量，自变量是时间。", "因变量是时间，自变量是质量。", "The dependent variable is time, while mass is the independent variable.", "自变量因变量身份颠倒", ""),
("academic", "The concentration doubled, but the total volume stayed constant.", "浓度加倍了，但总体积保持不变。", "总体积加倍了，但浓度保持不变。", "The total volume doubled, but the concentration stayed constant.", "变化物理量颠倒", ""),
("academic", "Each observation has equal weight, regardless of its numerical value.", "无论数值大小如何，每个观测值的权重都相同。", "观测值越大，其权重越大。", "Observations with larger numerical values have greater weight.", "等权重变成依赖数值的权重", ""),
("academic", "The minimum acceptable purity is 98 percent.", "可接受的最低纯度为98%。", "可接受的最高纯度为98%。", "The maximum acceptable purity is 98 percent.", "最低门槛变成最高门槛", ""),
("academic", "The measured current was 6 milliamperes, not 6 amperes.", "测得的电流是6毫安，不是6安。", "测得的电流是6安，不是6毫安。", "The measured current was 6 amperes, not 6 milliamperes.", "毫安与安单位颠倒", ""),
("academic", "Increasing the sample size reduces sampling error, not systematic bias.", "增大样本量减少的是抽样误差，不是系统性偏差。", "增大样本量减少的是系统性偏差，不是抽样误差。", "Increasing the sample size reduces systematic bias, not sampling error.", "两类误差颠倒", ""),
("academic", "The function returns a copy, leaving the original array unchanged.", "这个函数返回一个副本，原数组保持不变。", "这个函数返回一个副本，并修改原数组。", "The function returns a copy and modifies the original array.", "无原地修改变成修改原数组", ""),
("academic", "The valve opens only when the pressure falls below the threshold.", "只有压力降到阈值以下时，阀门才打开。", "只有压力升到阈值以上时，阀门才打开。", "The valve opens only when the pressure rises above the threshold.", "触发方向颠倒", ""),
("academic", "Absence of a visible precipitate does not prove that the solution is pure.", "没有可见沉淀，并不能证明溶液是纯净的。", "没有可见沉淀，证明溶液是纯净的。", "Absence of a visible precipitate proves that the solution is pure.", "不能证明变成证明", ""),
("hard", "The bond is strong.", "这种化学键很强。", "这种债券很强。", "The financial bond is strong.", "明确化学语境误用债券词义", "We are discussing the chemical bond between atoms."),
("hard", "The draft is ready.", "草稿准备好了。", "穿堂风准备好了。", "The air current is ready.", "文稿语境误用气流词义", "We are preparing a written proposal for review."),
("hard", "The pupil expanded.", "瞳孔扩大了。", "学生扩大了。", "The student expanded.", "眼部语境误用学生词义", "An eye examination is in progress."),
]

DEV = [
("daily", "Nora received the invoice from Sam.", "诺拉收到了萨姆发来的发票。", "萨姆收到了诺拉发来的发票。", "Sam received the invoice from Nora.", "发送接收角色颠倒", ""),
("daily", "You can move the chair, but don't remove the cushion.", "你可以挪动椅子，但不要拿掉坐垫。", "你可以挪动椅子，也可以拿掉坐垫。", "You can move the chair and remove the cushion.", "禁止变成许可", ""),
("travel", "The luggage fee is per bag, not per passenger.", "行李费按每件行李收取，不按每位乘客收取。", "行李费按每位乘客收取，不按每件行李收取。", "The luggage fee is per passenger, not per bag.", "收费单位颠倒", ""),
("food", "Could you bring a ladle for serving the stew?", "能拿一个盛炖菜的长柄汤勺吗？", "能拿一把吃炖菜的叉子吗？", "Could you bring a fork for eating the stew?", "盛菜餐具与进食餐具改变", ""),
("food", "The glaze contains honey, although the cake batter does not.", "糖衣含蜂蜜，但蛋糕面糊不含。", "蛋糕面糊含蜂蜜，但糖衣不含。", "The cake batter contains honey, although the glaze does not.", "成分位置颠倒", ""),
("academic", "The upper bound is inclusive, but the lower bound is exclusive.", "上界包含边界值，但下界不包含边界值。", "下界包含边界值，但上界不包含边界值。", "The lower bound is inclusive, but the upper bound is exclusive.", "边界包含关系颠倒", ""),
("academic", "The sensor measures speed rather than acceleration.", "这个传感器测量的是速率，而不是加速度。", "这个传感器测量的是加速度，而不是速率。", "The sensor measures acceleration rather than speed.", "测量物理量颠倒", ""),
("hard", "The cell divided.", "细胞分裂了。", "牢房分裂了。", "The prison cell divided.", "生物语境误用牢房词义", "We are observing a living cell under a microscope."),
]

TEST = [
("daily", "Ravi handed the notebook to Elena.", "拉维把笔记本交给了埃莱娜。", "埃莱娜把笔记本交给了拉维。", "Elena handed the notebook to Ravi.", "交付者接收者颠倒", ""),
("daily", "The latch is stuck, although the hinges still move freely.", "门闩卡住了，但铰链仍能灵活转动。", "铰链卡住了，但门闩仍能灵活转动。", "The hinges are stuck, although the latch still moves freely.", "部件状态颠倒", ""),
("travel", "The return journey requires a separate reservation.", "返程需要单独预订。", "去程需要单独预订。", "The outward journey requires a separate reservation.", "返程去程颠倒", ""),
("food", "Please put the olives in a separate ramekin, not in the soup bowl.", "请把橄榄放在单独的小陶瓷盅里，不要放进汤碗。", "请把橄榄放进汤碗，不要放在单独的小陶瓷盅里。", "Please put the olives in the soup bowl, not in a separate ramekin.", "容器要求颠倒", ""),
("food", "The noodles contain egg, even without the topping.", "即使不加配料，面条本身仍含鸡蛋。", "只要不加配料，面条就不含鸡蛋。", "The noodles contain no egg if the topping is omitted.", "配料与本体成分混淆", ""),
("academic", "The liquid absorbs energy as it evaporates.", "液体蒸发时吸收能量。", "液体蒸发时释放能量。", "The liquid releases energy as it evaporates.", "吸收释放颠倒", ""),
("academic", "The limit applies to the sum, not to each term separately.", "这个限制适用于总和，不是分别适用于每一项。", "这个限制分别适用于每一项，不适用于总和。", "The limit applies to each term separately, not to the sum.", "限制适用范围颠倒", ""),
("hard", "The current is weak.", "水流很弱。", "电流很弱。", "The electric current is weak.", "河流语境误用电流词义", "We are discussing the flow of water in a river."),
]
