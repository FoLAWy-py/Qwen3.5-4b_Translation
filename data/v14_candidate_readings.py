"""Individual content-bound decisions for changed fact-pilot outputs."""
BATCHES = {
    'public-02': {
        'stem':'runs/v14-matching-public',
        'snapshot_hash':'b32686867431c9c4d195444ccbae41875aa2056378d3560265d9aeee6c278630',
        'readings':'''
public-short-0189-zh-CN|pass|郊区生活必须有车的必要性、地点准确。
public-short-0190-en|pass|所说的话可能在法庭对你不利，可能性和证据方向完整。
public-short-0477-zh-CN|pass|愿意试试的礼貌询问准确。
public-short-0307-en|pass|他给我不错的礼物，施受关系与评价完整。
public-short-0151-zh-CN|pass|每次来此都点同样菜，频率和重复选择准确。
''',
    },
    'public-01': {
        'stem':'runs/v14-matching-public',
        'snapshot_hash':'1c1861fe6ad6c401ce54e057f4d1c05da0b1c4cc2ebb0b860da33fc35cbf8c7d',
        'readings':'''
public-short-0527-en|pass|洗澡频率为每隔一天，bath在未限定洗澡方式的中文下成立。
public-short-0391-en|pass|我回家之前对方先睡觉、我不介意，回家者与睡觉者正确区分。
public-short-0492-en|pass|Can I come to your place准确表达去对方那里的请求。
''',
    },
    'known-05': {
        'stem':'runs/v14-matching-known',
        'snapshot_hash':'367c80825be95fa8abe3d0ffe87de85652d824c4c0012b02ff3fda50b8f6694c',
        'readings':'''
v4-dev-078-zh-CN|pass|显微镜放大图像而非样品自身，否定对象与作用边界准确。
v4-dev-085-zh-CN|pass|种植地块语境下plot译地，尚未清理的否定与完成状态正确。
v4-dev-088-zh-CN|pass|化学烧杯语境下solution译溶液，且不稳定，消歧正确。
v4-dev-090-zh-CN|minor|未把金属工具file误成文件，但具体锉刀泛化工具，需要磨利也弱化为打磨。
''',
    },
    'known-04': {
        'stem':'runs/v14-matching-known',
        'snapshot_hash':'bbfa201531dc3e0af9e2b1ecb70b1c3f24acdb11384339a8b3ba53996641276d',
        'readings':'''
v4-dev-061-en|minor|正分子固定、分母越大分数越小的方向可辨，但缺少the larger...the smaller结构，逗号串接语法生硬。
v4-dev-062-en|pass|回归系数表达关联而未必因果效应，概念及限定完整。
v4-dev-065-en|minor|溶剂蒸发与溶解盐留下的关系正确，但中文已完成的事件译成现在时，完成性弱化。
v4-dev-066-zh-CN|pass|蒸腾作用中叶片经气孔散失水分，通路和过程正确。
v4-dev-069-zh-CN|minor|事件发生后写文献和可能性保留，但hindsight译事后回顾，事后知识视角的含义不够准确。
v4-dev-070-zh-CN|pass|调查遗漏已经搬走的人，对象和状态正确。
v4-dev-071-zh-CN|pass|校验和可检测某些错误而不能纠正，能力范围和否定准确。
v4-dev-071-en|major|校验和checksum误译parity checks奇偶校验，技术概念错误。
v4-dev-074-zh-CN|major|面积变为原来四倍误写增加四倍，增量与总量混淆。
''',
    },
    'known-03': {
        'stem':'runs/v14-matching-known',
        'snapshot_hash':'f7d257a584e4b921ce18498926921bdc6b25135f8f25a939772c04ab2386ce99',
        'readings':'''
v4-dev-040-en|minor|工作日有效且法定假日除外的逻辑完整，但凭券被具体译为ticket车票，文档类型不精确。
v4-dev-047-zh-CN|major|变稠之前打散结块的时间要求误为防止变稠的目的，操作逻辑改变。
v4-dev-049-zh-CN|major|capers刺山柑花蕾被译为沙葱，食材类别错误。
v4-dev-049-en|major|刺山柑花蕾误成citrus flower buds，青胡椒粒误成pepper flakes，食材错误。
v4-dev-050-zh-CN|pass|面糊倒入涂油烘焙容器并抹平表面的动作与顺序正确，tin译烤盘在此可接受。
v4-dev-051-en|major|分着喝同一碗汤被改成each have a bowl每人一碗，分食数量关系改变。
''',
    },
    'known-02': {
        'stem':'runs/v14-matching-known',
        'snapshot_hash':'8df26ffc6366db01a59f123c6ca64203266199f4814f59be18d80378ec61469a',
        'readings':'''
v4-dev-018-en|pass|划痕在保护膜而不在下面玻璃，位置和否定对象准确。
v4-dev-022-zh-CN|minor|比较方向正确为柜台比车上便宜，但on board未经语境限定被具体写成飞机，kiosk泛化柜台，地点表达不精确。
v4-dev-022-en|pass|上车票价比售票亭贵，比较方向和购票地点完整。
v4-dev-025-en|pass|租车停十二号位且钥匙放归还箱，两步骤完整；中文钥匙未指定单复数。
v4-dev-034-en|pass|仅最后一节车厢继续到终点，范围限制和行程准确。
''',
    },
}
