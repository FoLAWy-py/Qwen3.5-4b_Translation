"""Explicit individual decisions for changed outputs; frozen snapshot per batch."""
BATCHES = {
    'public-03': {
        'stem': 'runs/v13-candidate-public',
        'snapshot_hash': '27cf9f6260459c59321ef1b0a4fab4814c214f9bd1dcb374e5d531a3081247a2',
        'readings': '''
public-short-0099-zh-CN|pass|汤姆认为300美元买不齐全部所需物，金额及否定全称范围准确。
public-short-0072-zh-CN|minor|不肯加薪的拒绝意愿弱化为不会提高工资的未来判断。
public-short-0025-zh-CN|minor|下一个拐角左转的指令大意保留，但弯口搭配不自然。
public-short-0151-zh-CN|pass|每次来此都点同样菜，频率与重复选择完整。
public-short-0205-en|minor|家离公园近的地点关系正确，但遗漏很近的程度。
''',
    },
    'public-02': {
        'stem': 'runs/v13-candidate-public',
        'snapshot_hash': '19878315c4e86289a46f36a7dc55f37b67fb720b5c074a64f43730e6c4dd1727',
        'readings': '''
public-short-0376-en|minor|等待公交车正确，但遗漏站着的姿态。
public-short-0391-zh-CN|pass|我回家之前对方睡觉且我不介意，回家者与睡觉者准确。
public-short-0391-en|pass|I get home和you sleep保留主体区别与不介意态度。
public-short-0342-en|pass|多说少听的假设与两嘴一耳的反事实结果完整。
public-short-0597-en|minor|汤姆认为可能只有自己必须做事的内容正确，但only himself句式生硬。
public-short-0492-en|pass|Can I come to your place准确表达去对方那里的请求。
public-short-0189-en|pass|住郊区必须有车，必要性与条件完整。
public-short-0190-zh-CN|major|法庭用作证据保留，但遗漏against you对你不利，证据方向丢失。
public-short-0292-en|minor|汤姆无任何人帮助做成该事保留，但确实的明确强调弱化。
public-short-0477-zh-CN|pass|愿意试试的询问准确。
public-short-0307-en|pass|他给我不错的礼物，送礼施受关系和评价正确。
''',
    },
    'public-01': {
        'stem': 'runs/v13-candidate-public',
        'snapshot_hash': 'e31fef3ce6b5a030d47dcdddcb5fb0f0635c4659e3e1616a0e0b1c05634e554b',
        'readings': '''
public-short-0395-zh-CN|major|没有零钱被改为找不开零钱，添加找零用途，缺零钱的事实状态未被准确保留。
public-short-0527-en|pass|每隔一天洗澡译every other day，频率与动作准确，中文未限定浴缸或淋浴。
''',
    },
    'known-07': {
        'stem': 'runs/v13-candidate-known',
        'snapshot_hash': 'b7eec0250b582ff7ba0d771b86d0f45749559f7d916d0bd5a77be547c76ee884',
        'readings': '''
v4-dev-090-zh-CN|minor|手工金属工具语境未误成文件，但file具体锉刀被泛化工具，磨利也弱化打磨。
v4-dev-092-en|pass|铜导体语境下conductor译导体，损坏状态完整。
v4-dev-094-zh-CN|pass|侵蚀路径旁树木语境下root译树根，露出状态完整。
''',
    },
    'known-06': {
        'stem': 'runs/v13-candidate-known',
        'snapshot_hash': '36c92e32f9bc9c36d9cb8365c88852fcc2863923cc53eefd8c1f9101a9ef34e9',
        'readings': '''
v4-dev-074-zh-CN|major|quadruples变为原来四倍误写增加四倍，数学增量与总量混淆。
v4-dev-076-zh-CN|pass|区分两个定义后表面矛盾消失，条件关系完整。
v4-dev-078-zh-CN|pass|放大图像而非样品自身，否定对象和边界正确。
v4-dev-079-zh-CN|pass|化石给出岩层年代下限，界限方向正确。
v4-dev-082-zh-CN|pass|机械金属弹簧语境消歧正确，拉伸状态表达tension。
v4-dev-085-zh-CN|pass|种植地块语境下plot译地且尚未清理，语境与否定正确。
v4-dev-085-en|pass|种植地块仍未清理，plot消歧与not yet准确。
v4-dev-087-en|pass|眼睛对光反应的语境下pupil正确指瞳孔，反应缓慢准确。
v4-dev-088-zh-CN|pass|烧杯化学语境下solution译溶液且不稳定，消歧正确。
''',
    },
    'known-05': {
        'stem': 'runs/v13-candidate-known',
        'snapshot_hash': '2846dbc8c311ee369bc09c2817b38e71bbfe81c4cd91e22873920ae70c5b7897',
        'readings': '''
v4-dev-066-zh-CN|pass|蒸腾作用中叶片通过气孔散失水分，过程和通路准确。
v4-dev-070-zh-CN|pass|调查遗漏已搬走者，对象和完成状态正确。
v4-dev-071-zh-CN|pass|校验和只能检测某些错误不能纠正，能力边界与否定完整。
v4-dev-071-en|major|checksum校验和误译parity checks奇偶校验，技术方法错误。
''',
    },
    'known-04': {
        'stem': 'runs/v13-candidate-known',
        'snapshot_hash': 'a650ba356511e161b50301c7e4e8be3abbdfa795396c22019e210ed594aa7264',
        'readings': '''
v4-dev-047-zh-CN|major|before mixture thickens的时间要求误为防止变稠的目的，操作逻辑改变。
v4-dev-048-en|minor|蘸酱冷着上和馅料滚烫的对照保留，但served及piping程度弱化。
v4-dev-049-zh-CN|major|capers刺山柑花蕾误成沙葱，食材类别错误。
v4-dev-049-en|major|刺山柑花蕾误成citrus flower buds，青胡椒粒误成pepper flakes，两个食材均错误。
v4-dev-051-zh-CN|major|garnish泛指装饰配料被指定香草，未经原文支持改变食材。
v4-dev-051-en|major|分着喝这碗汤误成each have a bowl每人一碗，分食数量关系改变。
v4-dev-061-en|minor|正分子固定和分母大分数小的方向可辨，但缺少the larger...the smaller结构且逗号串接生硬。
''',
    },
    'known-03': {
        'stem': 'runs/v13-candidate-known',
        'snapshot_hash': '5c010e9b06676a18345e620e8bcc9762c548de199f953dbfc977c3a60c1500e5',
        'readings': '''
v4-dev-033-zh-CN|minor|下车再刷卡和否则收最高票价的逻辑正确，但按一下卡不清楚表达刷卡动作。
v4-dev-034-en|pass|仅最后车厢继续到终点的范围与方向正确。
v4-dev-040-zh-CN|pass|返程凭券工作日有效且排除节假日，适用范围完整。
v4-dev-040-en|minor|工作日及法定假日例外保留，但凭券voucher被改为ticket车票。
v4-dev-041-zh-CN|pass|孜然籽先短暂烘烤再磨碎，食材和顺序正确。
v4-dev-043-zh-CN|major|刮取柠檬外皮误为用柠檬刮去外皮，柠檬变成工具且取材动作错误。
''',
    },
}
