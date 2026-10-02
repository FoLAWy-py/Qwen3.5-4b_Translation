"""Freeze one changed factor: chosen-answer critical spans weighted3, same v15 budget."""
import copy
import json
from pathlib import Path
from witrans_tools.common import fingerprint, now, read_jsonl, write_json
from witrans_tools.critical_spans import encode_critical_spans, validated_annotations

SPANS = {
    'v10-daily-19-zh-CN': (['水的损坏'], 'water damage为水造成的损害，不能缩为水渍。'),
    'v5-seed-005-zh-CN': (['地面缆车'], 'funicular为轨道上的地面缆车，不是人行道。'),
    'v10-travel-20-zh-CN': (['一英里', '无障碍通行条件'], '保留a mile数值和accessibility通行条件。'),
    'v2-train-058-zh-CN': (['这份高汤本身', '甲壳类或贝类'], '询问高汤而非成品，shellfish涵盖甲壳类/贝类。'),
    'v10-food-13-zh-CN': (['食客', '炒鸡肉'], 'diner为食客，chicken stir-fry为炒鸡肉而非炒饭。'),
    'v10-food-02-zh-CN': (['用餐者', '花生酱绝不能提供'], '食客角色和严重过敏情境下的完全禁止必须保留。'),
    'v10-academic-01-zh-CN': (['可能会导致'], 'could lead只表示可能性，不是必然发生。'),
    'v8-multi-031-zh-CN': (['这个指标', '另一个指标'], '两处measure均为研究指标，不是措施。'),
    'public-short-0080-zh-CN': (['我嗓子有点哑'], 'frog in throat表达嗓音沙哑而非实体动物。'),
    'v5-seed-020-zh-CN': (['演员阵容'], '明确戏剧actors语境对应cast演员阵容。'),
    'v8-multi-035-zh-CN': (['现在该由你来决定了'], '明确决策责任语境对应ball in your court习语。'),
}


def main():
    root = Path('data/prepared/v16-span-trial')
    assert not root.exists(), 'Preserve experiment'
    prior = json.loads(Path('data/prepared/v15-error-repair/plan.json').read_text(encoding='utf-8'))
    diagnosis = json.loads(Path('runs/v15-error-diagnosis.json').read_text(encoding='utf-8'))
    assert diagnosis['recall']['pass_repair_rate'] == 0 and not diagnosis['optimization_screen_passed']
    pairs = read_jsonl(prior['pairs_path'])
    assert fingerprint(pairs) == prior['pairs_hash'] and set(SPANS) == {r['id'] for r in pairs}
    annotations = []
    for row in pairs:
        spans, note = SPANS[row['id']]
        # Food13's 食客 occurs twice; use its unique role-bearing clause.
        if row['id'] == 'v10-food-13-zh-CN':
            spans = ['食客想要半份', '炒鸡肉']
        assert all(row['output']['translation'].count(span) == 1 for span in spans)
        content = {k:row[k] for k in ('input','output','rejected','preference_issue')}
        annotations.append(dict(id=row['id'], critical_spans=spans, multiplier=3.0, reviewer='Codex',
            at=now(), note=note, content_hash=fingerprint({**content,'critical_spans':spans,'multiplier':3.0})))
    report = dict(at=now(), train_hash=fingerprint(pairs), annotations=annotations, annotation_hash=fingerprint(annotations),
                  scope='Explicitly read TRAIN source, chosen, actual rejected and issue. Only chosen NLL weighted; summed preference margins unchanged.', release_approved=False)
    validated_annotations(pairs, report)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-4B',local_files_only=True)
    coverage = []
    for row, entry in zip(pairs, annotations):
        encoded = encode_critical_spans(tokenizer,row,entry['critical_spans'],1024,3.0)
        assert encoded['token_weights'][-1] == 1.0
        assert all(weight == 0 for weight,label in zip(encoded['token_weights'],encoded['labels']) if label == -100)
        coverage.append(dict(id=row['id'], full_tokens=len(encoded['input_ids']),
                             supervised_tokens=sum(v!=-100 for v in encoded['labels']),
                             weighted_tokens=sum(v==3 for v in encoded['token_weights'])))
    write_json(root/'annotations.json',report)
    plan = copy.deepcopy(prior)
    plan.update(at=now(), output='models/witrans-4b-v16-span-cpo', critical_annotations=str(root/'annotations.json'),
        critical_annotations_hash=fingerprint(report), method='actual_error_cpo_with_critical_span_nll_and_same_replay',
        hypothesis='At the same16 updates, data, visit order, lr and replay, concentrating chosen NLL on reviewed error-bearing semantics produces decoded TRAIN repairs where v15 produced zero.',
        control=dict(plan='data/prepared/v15-error-repair/plan.json',plan_hash=fingerprint(prior),
                     adapter_sha256='996721ba625d008b8316ffbb689157e5f140bade0e907605bdb70fc3a543bba3'),
        changed_factor='Only chosen NLL token weights: reviewed spans3.0, other supervised tokens/EOS1.0; preference likelihoods unchanged.',
        selection='Predeclared final16; TRAIN recall first, no DEV checkpoint selection.',
        scope='One same-data/same-budget critical-span trial. No new negatives; all data exactly v15 TRAIN/replay. No algorithm superiority claim from one seed.',
        screening_rule=dict(train_recall_pass_at_least=3,train_recall_critical=0,format_direction_eos_all=True,
                            if_failed='Preserve evidence, no DEV generation; diagnose update strength before another configuration.',
                            if_passed='Freeze weights then generate the same known200/public116 once for semantic regression screening. Independent confirmation only after development gate passes.'),
        diagnosis_hash=fingerprint(diagnosis), full_token_budget=61450)
    write_json(root/'plan.json',plan)
    write_json(root/'token-coverage.json',dict(plan_hash=fingerprint(plan),coverage=coverage))
    print({'plan_hash':fingerprint(plan),'coverage':coverage,'changed_factor':plan['changed_factor']})


if __name__ == '__main__':
    main()
