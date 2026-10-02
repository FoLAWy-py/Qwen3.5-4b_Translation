"""Conservative lexical follow-up; never rewrite historical training or review evidence."""
import copy
from collections import Counter
from pathlib import Path

from scripts.decode_qwen35_repair_recall import load
from scripts.report_qwen35_error_gradients import summarize
from witrans_tools.common import fingerprint, now, read_jsonl, write_json, write_jsonl


def main():
    report_path=Path('runs/qwen35-v3-sweetener-negative-correction.json')
    assert not report_path.exists()
    at=now();rid='v10-food-15-zh-CN'
    prior_view='runs/qwen35-v3-train-discovery-lexical-corrected-manual.jsonl'
    old=read_jsonl(prior_view);new=copy.deepcopy(old)
    row=next(r for r in new if r['id']==rid)
    prior=copy.deepcopy(row)
    assert prior['verdict']=='major'
    row.update(verdict='minor',at=at,
        note='Cambridge将unsweetened同时译为未加糖的和未加甜味料的；原输出不加糖不能单独认定major。用蜂蜜加糖措辞虽欠精确，但可理解为用蜂蜜增甜，后句明确糖与任何甜味剂的范围区别。没有充分证据将整条作为稳定major负例，保守降为minor并排除；不等同于参考逐字正确。',
        rereview=dict(original_review_path=prior_view,original_review_hash=fingerprint(prior),
            original_verdict='major',original_note=prior['note'],correction_report=str(report_path),
            reviewer='Codex AI; not independent human review'))
    counts=dict(Counter(r['verdict'] for r in new))
    assert counts=={'pass':328,'minor':56,'major':32}
    pool_path='data/prepared/qwen35-v3-reviewed33-current-errors/pairs.jsonl'
    old_pool=read_jsonl(pool_path);pairs=[r for r in old_pool if r['id']!=rid]
    assert len(pairs)==32 and {r['id'] for r in pairs}=={r['id'] for r in new if r['verdict']=='major'}
    raw_probe=read_jsonl('runs/qwen35-v3-gradient-fp64.jsonl')
    selected=[r for r in raw_probe if r['id'] in {p['id'] for p in pairs}]
    assert len(selected)==32
    gradient33=load('runs/qwen35-v3-reviewed33-gradient-diagnosis.json')
    gradient=dict(at=at,original_probe_report_hash=gradient33['original_probe_report_hash'],
        original_executed_accounting=gradient33['original_executed_accounting'],selected_real_error_rows=32,
        selected_rows_hash=fingerprint(selected),clean_pairs_hash=fingerprint(pairs),overall=summarize(selected),
        scope='CPU subset of original actual FP64 probe; no new GPU calls/updates. All original205 forwards164 backwards remain counted.',
        stage_goal_complete=False)
    plan=load('data/prepared/qwen35-v3-reviewed33-repair/plan.json')
    correction=dict(at=at,source=dict(title='Cambridge unsweetened',
        url='https://dictionary.cambridge.org/us/dictionary/english-chinese-simplified/unsweetened',
        finding='Both 未加糖的 and 未加甜味料的 are listed; sugar wording alone cannot justify a major negative.'),
        issue_id=rid,input=next(p['input'] for p in old_pool if p['id']==rid),
        original_actual_rejected=next(p['rejected'] for p in old_pool if p['id']==rid),
        old_review_hash=fingerprint(prior),new_review_hash=fingerprint(row),old_verdict='major',new_verdict='minor',
        original_review_path=prior_view,original_review_hash=fingerprint(old),
        original_pool_path=pool_path,original_pool_hash=fingerprint(old_pool),
        invalidated_plan_hashes=[fingerprint(plan)],invalidated_outputs=[plan['output']],
        quarantined_adapter_sha256='073e596f8ee4ef9b8b7f1e436aa11b30f50e6fd899859b7cda3280d02b2fcd8b',
        reason='Unsweetened/sugar lexical choice and honey phrasing were not sufficiently established as a true major negative.',
        old_trial_eligible=False,optimization_round_requirement_satisfied=False,
        original_training_completion_audit='runs/qwen35-v3-reviewed33-training-completion-audit.json',
        actual64_update_and_budget_evidence_preserved=True,
        already_running_recall_policy='Preserve bounded113 diagnostic generations and individually review; do not terminate/restart or decode DEV. Quarantine forbids candidate acceptance and inheritance.',
        new_view='runs/qwen35-v3-train-discovery-sweetener-corrected-manual.jsonl',new_view_hash=fingerprint(new),
        clean_pool='data/prepared/qwen35-v3-reviewed32-current-errors/pairs.jsonl',clean_pool_hash=fingerprint(pairs),
        clean_gradient='runs/qwen35-v3-reviewed32-gradient-diagnosis.json',clean_gradient_hash=fingerprint(gradient),
        counts=counts,pairs=32,groups=len({p['group_id'] for p in pairs}),
        all_prior_quarantines_preserved=True,
        before_next_optimization='Full focused32 negative source reread with conservative severity and dictionary verification of any uncertain lexical contrasts before freezing any further trial. No new trial frozen by this correction.',
        stage_goal_complete=False,release_approved=False,default_promoted=False)
    for path in (correction['new_view'],correction['clean_pool'],correction['clean_gradient']):
        assert not Path(path).exists()
    write_jsonl(correction['new_view'],new);write_jsonl(correction['clean_pool'],pairs)
    write_json(correction['clean_gradient'],gradient);write_json(report_path,correction)
    assert fingerprint(read_jsonl(prior_view))==fingerprint(old)
    assert fingerprint(read_jsonl(pool_path))==fingerprint(old_pool)
    print(dict(report=str(report_path),counts=counts,pairs=32,groups=correction['groups'],old_trial_eligible=False),flush=True)


if __name__=='__main__':
    main()
