"""Bind 17 individually read changed candidate translations to their snapshot."""
from scripts.export_v13_first_readings import save
from witrans_tools.common import fingerprint, read_jsonl

READINGS = """
005-en|pass|冲洗海绵并放沥水板上的连续动作和礼貌请求完整。
006-zh-CN|minor|周二和带狗粮的条件正确，但看您的狗未清楚表达照看狗的照料意义。
007-en|major|拉上背包即合上拉链被译为pull the backpack up提起背包，触发动作错误。
009-en|pass|污渍变淡但尚未完全消失，程度和否定完整。
012-en|pass|误把椅子上外套当成对方的，位置和误认关系正确。
014-en|pass|脱水功能正常而目前不能排水，两项功能对照准确。
015-zh-CN|major|bin liners垃圾袋误成纸箱，物品类别错误。
015-en|minor|购物袋和否定保留，但The garbage bags are out不自然，缺少明确用完的表达。
017-zh-CN|pass|除非提前取消否则自动续费，逻辑条件完整。
018-en|pass|划痕在保护膜而不在下面玻璃，位置及否定范围完整。
022-zh-CN|major|车上比售票亭贵被倒成柜台比飞机上贵，价格比较方向反转。
022-en|pass|上车买票比售票亭贵，比较方向和购票地点正确。
023-zh-CN|pass|步行桥关闭和行人必须走地下通道完整。
024-zh-CN|pass|房间保留至午夜而非翌日早晨，截止时刻和否定完整。
025-zh-CN|major|drop box归还箱译取车箱，钥匙归还地点的功能错误。
025-en|pass|租车停十二号位和钥匙投入归还箱两步骤完整；中文钥匙未指定单复数。
031-zh-CN|minor|干燥天气仍滑保留，但瀑布之后的空间指示生硬且不够清楚。
"""


def main():
    rows = read_jsonl('runs/v13-candidate-known-unmatched.jsonl')
    if fingerprint(rows) != '54de8bc4202518a9eae367f5da2c0faffee54408ebca5632c3351dd6c838069d':
        raise ValueError('Read snapshot changed')
    annotations = {}
    for line in READINGS.strip().splitlines():
        suffix, verdict, note = line.split('|', 2)
        annotations['v4-dev-'+suffix] = (verdict, note)
    if set(annotations) != {row['id'] for row in rows}:
        raise ValueError('Read IDs do not match snapshot')
    save('runs/v13-candidate-known-manual.jsonl', [
        {'id': row['id'], 'reviewer': 'Codex', 'generation_hash': fingerprint(row),
         'verdict': annotations[row['id']][0], 'note': annotations[row['id']][1], 'language_correct': True}
        for row in rows
    ])
    print({'new_individual_readings': len(rows)})


if __name__ == '__main__':
    main()
