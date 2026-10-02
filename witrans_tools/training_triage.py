"""Source-family-safe stratified selection from training-only likelihood scores."""


def select_training_triage(scores,limit=64,quota=6):
    ranked = sorted([r for r in scores if r['short_decode_eligible']],key=lambda r:(-r['answer_eos_nll'],r['id']))
    selected,groups = [],set()
    for category in ('daily','travel','food','academic','hard'):
        for direction in ('zh-CN','en'):
            count = 0
            for row in ranked:
                if row['category']!=category or row['target_lang']!=direction or row['group_id'] in groups:
                    continue
                selected.append(row)
                groups.add(row['group_id'])
                count+=1
                if count==quota:
                    break
    for row in ranked:
        if len(selected)>=limit:
            break
        if row['group_id'] not in groups:
            selected.append(row)
            groups.add(row['group_id'])
    if len(selected)!=limit or len(groups)!=limit:
        raise ValueError('Distinct eligible families required for every selected record')
    if limit>=10*quota:
        for category in ('daily','travel','food','academic','hard'):
            for direction in ('zh-CN','en'):
                if sum(r['category']==category and r['target_lang']==direction for r in selected)<quota:
                    raise ValueError('Each planned category/direction needs its quota')
    return selected
