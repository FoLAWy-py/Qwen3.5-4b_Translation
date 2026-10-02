# RAG 结果与下一步

已完成本地索引、完整开发质量对照和正式测速。**177 条术语库没有质量净收益，暂不切换默认配置。** 这不能证明所有 RAG 无效，也不能证明微调已达到上限。

## 当前实现

[知识库](../data/rag-reviewed-training-terms-20261002.jsonl)来自 138 个既有合成训练来源组，Codex AI 阅读双语材料后提取词义和适用语境。未使用开发参考、新 confirmation、release 或候选测试输出；它不是独立人工词典。

建库使用 OpenAI `text-embedding-3-large`，3072 维，向量保存在本地 `rag/index.json`。库小，采用精确扫描，无需数据库服务。[模型说明](https://developers.openai.com/api/docs/models/text-embedding-3-large)及[embedding 文档](https://developers.openai.com/api/docs/guides/embeddings)。

| 路径 | 检索方式 | 新句是否调用 API |
|---|---|---|
| `local_terms` | 完整术语匹配＋词义语境锚点 | 否 |
| `semantic_terms` | 同样的候选筛选，再以 OpenAI 查询向量核对相似度 | 有候选时调用；重复查询可缓存 |

库向量只在建库和新增知识时计算，但新句仍需要查询向量。真正离线语义检索需要本地 embedding 模型，并用它重新编码整个知识库，不能混用两个模型的向量空间。本轮尚未验证本地 embedding 模型。

冲突词义放弃注入，用户显式词表优先，最多添加三个术语；超预算报错，不截断。两路径及采用条件在输出前冻结，见 [protocol.json](../runs/takeover-20261002/rag/protocol.json) 和 [prepared-freeze.json](../runs/takeover-20261002/rag/prepared-freeze.json)。

## 质量与速度

最终命中已知 14/200、公开 0/116，两个路径的完整模型输入全部相同。实际生成完整 316 条；本地路径明确复用相同输入的当轮输出，不宣称生成了 632 条。Codex AI 阅读全部原文、语境、参考和译文后判定：

| 集合 | 无 RAG pass/minor/major | RAG |
|---|---:|---:|
| 已知 200 | 168 / 19 / 13 | 168 / 19 / 13 |
| 公开 116 | 105 / 8 / 3 | 105 / 8 / 3 |

两侧采用同一合理歧义重审口径；历史口径已知均为 166/17/17，也没有净提升。六条输出变化多为等义措辞；新增 major/critical 为零，已知原有方向错误仍在。完整判定、分项和来源组区间见 [质量摘要](../runs/takeover-20261002/rag/quality/semantic-summary.json)及同目录逐条记录。这是开发回归，非独立验收。

固定 24 句，两路径各整轮预热后三轮正式测量，单进程交错运行；远程查询关闭缓存：

| 路径，各 72 次 | 端到端平均 | P95 | 平均检索 |
|---|---:|---:|---:|
| 本地术语筛选 | **2.937 秒** | **3.487 秒** | 0.000645 秒 |
| OpenAI 查询＋本地索引 | **3.119 秒** | **4.107 秒** | 0.176669 秒 |

NF4、总预算 1024、输出≤128，两路径各输出 2217 tokens。JSON/EOS 全正常，无 OOM 或 CPU 参数，峰值保留 3.432 GiB。索引初始化 0.593 秒、模型加载 16.638 秒、合计整轮预热 170.448 秒，复用已有磁盘编译缓存。端到端吞吐分别 10.484 与 9.871 输出 tokens/秒，不能称纯 decode 速度。

只有 2/24 句触发远程查询，正式六次单句 API 耗时 1.11–6.12 秒，均值约 2.112 秒。低命中率摊薄了延迟，扩大覆盖后不能保证仍≤4秒。也不能把本轮本地 2.937 秒与另一次无 RAG 3.164 秒相减宣称加速。

逐输入耗时、tokens、API 事件及配置见 [性能摘要](../runs/takeover-20261002/rag/performance/summary.json)和同目录原始记录。测速译文由 Codex AI 逐条审核为 **20 pass / 1 minor / 3 major**，方向 22/24；三个 major 与原无 RAG 输出相同。详见 [测速语义审核](../runs/takeover-20261002/rag/performance/semantic-summary.json)。

## 继续推进

先扩大有来源和词义审核的知识覆盖，再验证本地 embedding 的检索质量及 CPU 延迟。新配置先跑完整旧 316 条和正式 24 句；只有质量净收益且性能合格，才冻结配置并新建独立 confirmation。600 条 release 继续封存。

现有结果更支持先补知识和语境。若扩大覆盖后角色、否定、条件错误仍明显，再比较更强基座；不继续循环局部纠错训练。

实验入口（`.env` 的 `OPENAI_API_KEY` 仅在语义路径使用）：

```powershell
uv run --locked --extra compile python -m scripts.low_cpu_run --module scripts.rag_translate "We booked adjoining rooms." zh-CN --profile local_terms --trace
```

改为 `--profile semantic_terms` 使用远程查询；`--query-cache` 可启用重复查询缓存。原正式 `witrans` 行为保持。正式证据目录拒绝覆盖，重跑需另建目录和冻结协议。

93 项 CPU 测试通过，实际 wheel 隔离导入通过，34 项身份与证据校验通过：[包检查](../runs/takeover-20261002/rag/package-receipt.json)、[最终校验](../runs/takeover-20261002/rag/final-receipt.json)。原回执中的未推送状态描述的是该实验完成时；后续用户单独授权上传，不改变原实验记录。
