"""Preserve historical evidence; publish separate source-grounded TRAIN corrections."""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.report_qwen35_error_gradients import summarize
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl
from witrans_tools.data import validate_record


def main():
    report_path = Path('runs/qwen35-v3-lexical-negative-correction.json')
    assert not report_path.exists(), 'Preserve prior correction'
    at = now()
    original_path = 'runs/qwen35-v3-train-discovery-conservative-manual.jsonl'
    candidate_path = 'runs/qwen35-v3-reviewed34-recall-manual.jsonl'
    pool_path = 'data/prepared/qwen35-v3-reviewed-current-errors/pairs.jsonl'
    original = read_jsonl(original_path)
    candidate = read_jsonl(candidate_path)
    pool = read_jsonl(pool_path)
    assert len(original) == 416 and len(candidate) == 114 and len(pool) == 34
    ambiguous = 'v10-food-01-en'
    food_note = ('原文青豆无豌豆限定，context/glossary均未消歧；Cambridge green bean词条接受青豆/四季豆。'
                 'green beans是可接受词义；鹰嘴豆用于汤、青豆留另一道菜及唯有鹰嘴豆适用的关系均完整，判pass。'
                 '参考green peas是一种可接受解释，不能据此把green beans作负例。')
    slotted_note = ('漏勺、捞饺子和请求均正确，主要语言方向已修复；遗漏正在慢煮的汤这一介质和温度细节，'
                    '按附属细节遗漏判minor，不作下一轮负例。原起点该条英文方向错误仍是major。')
    condition_note = ('只有两人都未能在你到达之前回家时才取钥匙；译文删去在你到达之前这一限定，'
                      '改变取钥匙条件，判major。保留旧笔记但不沿用其扩大条件的方向断言。')
    changes = []

    def corrected(rows, path, updates):
        result = copy.deepcopy(rows)
        for row in result:
            if row['id'] not in updates:
                continue
            verdict, note = updates[row['id']]
            old = copy.deepcopy(row)
            row.update(verdict=verdict, note=note, at=at,
                       rereview=dict(original_review_path=path, original_review_hash=fingerprint(old),
                                     original_verdict=old['verdict'], original_note=old['note'],
                                     correction_report=str(report_path), reviewer='Codex AI; not independent human review'))
            # Generation/source/reference binding stays exact; no raw output repair.
            for key in ('generation_hash', 'binding_hash', 'chosen_content_hash', 'format_valid', 'ended', 'language_correct'):
                assert row[key] == old[key]
            changes.append(dict(view=path, id=row['id'], old=fingerprint(old), new=fingerprint(row),
                                old_verdict=old['verdict'], new_verdict=verdict, note=note))
        assert set(updates) <= {r['id'] for r in result}
        return result

    original_new = corrected(original, original_path, {ambiguous: ('pass', food_note)})
    candidate_new = corrected(candidate, candidate_path, {
        ambiguous: ('pass', food_note), 'v4-mining-009-zh-CN': ('minor', slotted_note),
        'v8-multi-001-zh-CN': ('major', condition_note)})
    assert dict(Counter(r['verdict'] for r in original_new)) == {'pass':328, 'minor':55, 'major':33}
    assert Counter(r['verdict'] for r in candidate_new) == {'pass':78, 'minor':24, 'major':12}
    new_pool = [copy.deepcopy(r) for r in pool if r['id'] != ambiguous]
    ids = {r['id'] for r in new_pool}
    assert len(new_pool) == 33 and len(ids) == 33
    assert ids == {r['id'] for r in original_new if r['verdict'] == 'major'}
    assert 'v4-mining-009-zh-CN' in ids
    original_lookup = {r['id']:r for r in original_new}
    for row in new_pool:
        validate_record(row, True, True, purpose='training')
        assert row['preference_review']['verdict'] == 'major'
        assert row['preference_review']['binding_hash'] == original_lookup[row['id']]['binding_hash']
        assert row['preference_review']['hash'] == fingerprint({k:row[k] for k in ('input','output','rejected','preference_issue')})
    raw_probe = read_jsonl('runs/qwen35-v3-gradient-fp64.jsonl')
    selected = [r for r in raw_probe if r['id'] in ids]
    assert len(selected) == 33 and {r['id'] for r in selected} == ids
    prior_gradient = json.loads(Path('runs/qwen35-v3-reviewed34-gradient-diagnosis.json').read_text(encoding='utf-8'))
    strata = defaultdict(list)
    for row in selected:
        pair = next(p for p in new_pool if p['id']==row['id'])
        strata[f"{pair['category']}/{pair['input']['target_lang']}"].append(row)
    gradient = dict(at=at, original_probe_report_hash=prior_gradient['original_probe_report_hash'],
                    original_executed_accounting=prior_gradient['original_executed_accounting'],
                    selected_real_error_rows=33, selected_rows_hash=fingerprint(selected),
                    clean_pairs_hash=fingerprint(new_pool), overall=summarize(selected),
                    strata={key:summarize(value) for key,value in strata.items()},
                    scope='CPU subset of existing actual FP64 probe; no new GPU calls or optimizer updates',
                    removed_ambiguous_negative_id=ambiguous, stage_goal_complete=False)
    train_path = 'data/prepared/v16-reviewed-families/train-ready-after-overlap-review.jsonl'
    supplement_path = 'data/prepared/qwen35-v3-context-supplement/train.jsonl'
    train = read_jsonl(train_path)
    supplement = read_jsonl(supplement_path)
    positive_new = copy.deepcopy(supplement)
    age_changes = []
    replacements = {
        'q35-v3-supplement-daily-03-zh-CN': '米拉为她的兄弟预订了阅读隔间，但预订仍登记在她本人名下。图书管理员应核查米拉的预订，而不是让她的兄弟再订一次。',
        'q35-v3-supplement-daily-03-en': "Mira reserved a reading booth for her younger brother, but the reservation remains under her own name. The librarian should check Mira's booking rather than ask her younger brother to make a second one."}
    for row in positive_new:
        if row['id'] in replacements:
            old = copy.deepcopy(row)
            row['output']['translation'] = replacements[row['id']]
            row['review'] = dict(status='approved', reviewer='Codex AI; not independent human review', at=at,
                method='Separate post-recall source-grounded sibling-age correction; inputs, permission and groups unchanged',
                content_hash=fingerprint(dict(input=row['input'], output=row['output'])),
                notes='English brother is age-neutral; Chinese 弟弟 requires younger brother.',
                previous_review=old['review'], original_record_hash=fingerprint(old))
            age_changes.append(dict(id=row['id'], input_hash=fingerprint(row['input']),
                                    old_output=old['output'], new_output=row['output'],
                                    old_hash=fingerprint(old), new_hash=fingerprint(row)))
        validate_record(row, True, True, purpose='training')
    assert len(age_changes) == 2
    replay = train + positive_new
    assert len(replay) == 456 and len({r['group_id'] for r in replay}) == 220
    for row in replay:
        validate_record(row, True, True, purpose='training')
    assert replay[:416] == train
    outputs = {
        'original_review': 'runs/qwen35-v3-train-discovery-lexical-corrected-manual.jsonl',
        'candidate_review': 'runs/qwen35-v3-reviewed34-recall-conservative-manual.jsonl',
        'clean_pairs': 'data/prepared/qwen35-v3-reviewed33-current-errors/pairs.jsonl',
        'gradient': 'runs/qwen35-v3-reviewed33-gradient-diagnosis.json',
        'supplement': 'data/prepared/qwen35-v3-age-corrected-positives/supplement.jsonl',
        'replay': 'data/prepared/qwen35-v3-age-corrected-positives/replay.jsonl',
        'pool_audit': 'runs/qwen35-v3-reviewed33-current-error-pool-audit.json'}
    assert all(not Path(p).exists() for p in outputs.values())
    plan_path = 'data/prepared/qwen35-v3-reviewed34-repair/plan.json'
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    report = dict(at=at, sources=[dict(title='Cambridge English–Chinese green bean',
        url='https://dictionary.cambridge.org/us/dictionary/english-chinese-simplified/green-bean',
        finding='The entry explicitly permits 青豆 and 四季豆; unqualified 青豆 cannot demand peas.'),
        dict(title='MOE 青豆',url='https://dict.revised.moe.edu.tw/dictView.jsp?ID=101958&la=0&powerMode=0',
             finding='The regional entry refers to soybean seeds, further illustrating lexical ambiguity.')],
        correction_basis='Individual AI source reread and dictionary verification; discovered after training, not pretraining review.',
        old_trial_eligible=False, optimization_round_requirement_satisfied=False,
        invalidated_plan_hashes=[fingerprint(plan)], invalidated_outputs=[plan['output']],
        quarantined_adapter_sha256='3e95cd30ad4fa13b0db35102d86f9f0b8bf01630b1da5ba2f0fec9346166df61',
        reason='Reasonable green beans translation was wrongly used as a major-error CPO negative.',
        original_files={p:fingerprint(read_jsonl(p)) for p in (original_path,candidate_path,pool_path,train_path,supplement_path)},
        changes=changes, positive_age_corrections=age_changes, outputs=outputs,
        new_hashes=dict(original_review=fingerprint(original_new),candidate_review=fingerprint(candidate_new),
                        clean_pairs=fingerprint(new_pool),gradient=fingerprint(gradient),
                        supplement=fingerprint(positive_new),replay=fingerprint(replay)),
        counts=dict(original=dict(Counter(r['verdict'] for r in original_new)),
                    candidate=dict(Counter(r['verdict'] for r in candidate_new)), pairs=33,
                    pair_groups=len({r['group_id'] for r in new_pool}), replay=456, replay_groups=220),
        original_frozen_screening_preserved=True, corrected_candidate_screen_still_failed=True,
        unchanged_failures=dict(preservation_pass=36,preservation_minimum=39,supplement_pass=22,
                                supplement_minimum=32,supplement_major=5,supplement_major_maximum=4),
        no_dev_or_confirmation_generated=True, original_416_positives_reused_unchanged=True,
        ambiguous_green_peas_reference_retained_as_permissible_positive=True,
        prior_quarantines_remain_in_force=True, stage_goal_complete=False, release_approved=False, default_promoted=False)
    audit = json.loads(Path('runs/qwen35-v3-reviewed-current-error-pool-audit.json').read_text(encoding='utf-8'))
    audit.update(at=at,major_critical_pairs=33,review_hash=fingerprint(original_new),
                 review_path=outputs['original_review'],pairs_hash=fingerprint(new_pool),pool_path=outputs['clean_pairs'],
                 correction_report=str(report_path),correction_report_hash=fingerprint(report),
                 counts=report['counts']['original'], selection='33 source-grounded original-start errors; ambiguous food wording excluded.',
                 supersedes_error_pool_audit_hash=fingerprint(json.loads(Path('runs/qwen35-v3-reviewed-current-error-pool-audit.json').read_text(encoding='utf-8'))))
    audit['excluded_nonmajor_ids'] += [ambiguous]
    audit['diagnostic_error_strata'] = dict(Counter(f"{r['category']}/{r['input']['target_lang']}" for r in new_pool))
    write_jsonl(outputs['original_review'],original_new)
    write_jsonl(outputs['candidate_review'],candidate_new)
    write_jsonl(outputs['clean_pairs'],new_pool)
    write_json(outputs['gradient'],gradient)
    write_jsonl(outputs['supplement'],positive_new)
    write_jsonl(outputs['replay'],replay)
    write_json(outputs['pool_audit'],audit)
    write_json(report_path,report)
    # Verify all historical inputs remained byte-semantically identical.
    assert all(fingerprint(read_jsonl(p)) == sha for p,sha in report['original_files'].items())
    print(json.dumps(dict(report=str(report_path),counts=report['counts'], eligible=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
