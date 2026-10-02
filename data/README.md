# 数据阶段与来源

v2 为较小的优化试验：238 条新验收训练数据与 150 条去除数字重复的旧训练数据，共 388 条、192 个来源组。新开发与测试各40条。
本轮保留 API 等义不同措辞，具体修订与拒绝有逐条记录。中英方向错误采用衍生版本更正，原文件保留；正式开发集为 `prepared/v2/dev_corrected.jsonl`。
实验约束与模型晋级标准见 [v2 政策](v2_policy.md)，冻结数据与实际训练快照检查见 `../runs/v2-data-verification.json`。

另有两个先冻结后生成的精度验收集合：`prepared/v2/precision_test.jsonl` 和 `prepared/v2/bf16_test.jsonl`，各 40 条。后者明确设置 `training_allowed=false`，本地评测可以使用，训练、标注导入与训练划分默认拒绝。全部新精度测试都禁止加入后续训练或用于选权重。

v1 已冻结 3,000 条训练、1,000 条开发、1,000 条测试，来源是 `v1_families.py` 中 125 个原创中英句式组。
每组 20 个数量变体、两个方向；75 个训练组、25 个开发组、25 个测试组，组间不交叉。
数量变体不是独立语义覆盖。本次由 Codex 根据用户授权验收，见 [验收政策](v1_acceptance_policy.md)。

`prepared/v1/train.jsonl` 保存最终已验收数据；`generated/v1_candidates.jsonl` 和 `generated/v1_decisions.jsonl` 保存 API 候选、修订与指纹。
API 候选中的等义不同措辞也会替换为独立参考，替换率不是错误率。
`prepared/v1/test_corrected.jsonl` 独立修正两组参考的截止边界及 arrival 含义，原始冻结文件仍保留，原文及训练数据均未改变。
`prepared/v1/acceptance_120.jsonl` 为 100 条修正参考的筛查加 20 条提前冻结的独立挑战。
这 120 条已全部生成并由 Codex 逐条验收；完整 1,000 条测试尚未全部生成。模型语义验收未通过，见 [模型报告](../runs/v1-model-acceptance.json)。

`pilot_inputs.jsonl` 是本次任务原创的 40 条输入，只用于验证流程。不是正式覆盖语料。
类别比例 25% daily、15% travel、15% food、35% academic、10% hard；目标方向中英各半。
逐条来源许可记录在 `source` 中，无用户文件或外部抓取内容。

`pilot_decisions.jsonl` 保存 Codex 对 Qwen3-32B 输出的独立 AI 逐条复核和修订。
这不是人工质量验收，不代表专业翻译准确率。

`generated/` 为 API 候选和导入审核后的数据，`prepared/` 为冻结分组划分，均不默认加入版本控制。
下一轮扩充应优先增加独立语义、自然对话和长文本，避免靠更多数字变体扩大行数；继续保留审核身份及新开发/测试来源。
相同课文、会话、模板、改写和反向翻译必须归在同一 group_id；分组工具不能自动判断全部语义近重复。
