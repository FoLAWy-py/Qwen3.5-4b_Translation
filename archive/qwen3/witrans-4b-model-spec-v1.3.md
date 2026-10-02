# witrans-4b

## 单一通用翻译模型：数据、训练与 Python 调用

**文档版本：** 1.3  
**修订日期：** 2026-09-30  
**目标：** 得到一个可通过 Python 本地调用、覆盖日常交流、旅行、食物和多学科课堂的中英双向翻译模型。  
**基础模型：** `Qwen/Qwen3-4B`  
**标注来源：** `Qwen/Qwen3-32B` API 的最终译文  
**训练方式：** 基于模型生成标注数据的 QLoRA / SFT  
**本地训练资源：** 单张 8GB NVIDIA 显卡，具体显卡和依赖兼容性需确认。

本版替代 v1.2。交付目标是模型权重及直接调用方法，不是翻译服务平台。本文和配套 `witrans.py` 不包含已经训练好的权重；代码已完成语法及不加载权重的检查，尚未进行实际 GPU 推理和翻译质量验证。

---

## 1. 一个模型，不要求选择场景模式

日常交流、旅行、食物、课堂全部使用同一个模型、同一个 adapter、同一个输入协议和同一份系统提示词。不设置场景开关，也不增加隐藏的分类器或领域路由。

**这些类别用于准备训练数据和分别检查效果，不是模型调用时必须填写的选项。**

模型的统一任务是：根据原文及实际提供的上下文，输出忠实、自然的目标语言译文。口语保留口语特点，专业表述保留专业含义，不需要用户先标记“这是口语”或“这是课堂”。实际效果需要测试，不能因为使用同一个模型就宣称覆盖全部长尾术语。

例如，以下三句话均用同一方法调用，不传场景类别：

| 输入 | 译文示例 |
|---|---|
| Could you put the sauce on the side? | 可以把酱汁单独放吗？ |
| Is the deposit refundable? | 押金可以退吗？ |
| Let x be a positive real number. | 设 x 为正实数。 |

示例为人工拟定的预期表达，不是模型实测结果，也不是唯一正确措辞。

**目标语言仍需指定。** `target_lang="en"` 或 `target_lang="zh-CN"` 表达翻译方向，不是场景模式。中英混合文本统一译成指定目标语言，必要专名和代码保留。

上下文与术语表都是可选补充。普通句子不需要它们；孤立的多义词没有足够信息时，任何统一接口都不能消除信息本身的缺失。

## 2. “32B 输出 → 4B 微调”究竟是什么

准确描述为：

> 使用 Qwen3-32B API 为输入文本生成候选译文，审核后形成监督数据，对 Qwen3-4B 做 QLoRA / SFT，得到 witrans-4b。

```text
原文 ──→ Qwen3-32B API ──→ 最终译文 ──→ 审核后的训练数据
                                              │
已有 Qwen3-4B 权重 ─────────────────────────────┤
                                              ↓
                                         QLoRA / SFT
                                              ↓
                                          witrans-4b
```

只需要 32B 的最终输出。**不读取、不下载、不复制 32B 的参数，不获取它的中间激活，也不拟合它的逐 token 概率分布。** 本地需要的是 4B 的权重和训练数据，不是在本地加载 32B。

这种利用较大模型生成完整答案，再用这些答案训练另一模型的方式，可以称为“响应蒸馏”或“序列级蒸馏”。序列级知识蒸馏已有通过生成译文、再进行交叉熵训练的研究依据；本方案采用这一类输出监督思路，不声称复现论文的全部训练和解码设置。[R1]

但“把 32B 压成 4B”容易误解，所以本文主要使用 **“32B 标注数据 + 4B 监督微调”**。4B 从已有 Qwen3-4B 初始化，不是从 32B 删去参数，也不是从零训练。[R2]

**没有先蒸馏一次、再 QLoRA 一次的必需两阶段。** 这里读取 32B 答案进行的那次 QLoRA/SFT，就是从输出中学习的过程。SFT 是训练目标，QLoRA 是冻结量化基础权重、训练低秩增量的实现方式。[R3][R4]

## 3. 统一输入、输出和翻译行为

Python 调用仅要求原文和目标语言：

```python
result = model.translate("Can I check in early?", target_lang="zh-CN")
```

模型输入统一序列化为：

```json
{
  "text": "Can I check in early?",
  "target_lang": "zh-CN",
  "context": "",
  "glossary": {}
}
```

模型只生成：

```json
{"translation":"我可以提前办理入住吗？"}
```

Python 函数解析后返回同样结构的 `dict`。需要纯译文时读取 `result["translation"]`，无需为此训练另一个模型或增加输出模式。

统一行为要求：忠实保留人称、语气、否定、条件、数量、单位和术语含义；只翻译当前原文，不解释、不回答原文问题、不把历史重复翻译。半句话只翻译可见片段，不凭空续写。食物或旅行内容不增加原文没说的成分、路线、价格或保证。

明确的代码、公式、标识符、URL 与占位符要求保留，并放进测试集。规范格式目前指固定 JSON 输出及这些文本内约束，不建立完整 HTML/Markdown 文件转换系统。格式检查不能证明语义正确。

可选上下文与术语示例：

```python
result = model.translate(
    "The gradient is close to zero.",
    target_lang="zh-CN",
    context="We are discussing the optimization of a loss function.",
    glossary={"gradient": "梯度", "loss function": "损失函数"},
)
```

不要求提前建立课程包、注册领域或维护数据库。术语表就是一个普通 Python 字典，只放当前相关词条，并按原文词义使用；无关词条不能硬塞入译文。不传字典时仍必须能完成普通及常见专业内容翻译。

## 4. 数据覆盖：混合训练，不分模型

先用 **3,000–5,000 条**合格数据跑通训练并检查收益；初轮扩大到 **20,000–30,000 条**。只有评测显示确有覆盖不足时再增加，不把前版 60,000 条作为必须完成的前置门槛。以下比例是起始设计，不是已证明的最优配方。

| 数据类别 | 建议占比 | 覆盖内容 |
|---|---:|---|
| 日常交流 | 25% | 寒暄、请求、拒绝、购物、办事、同学/同事交流、自然口语 |
| 旅行 | 15% | 机场、行李、转机、酒店、押金、交通、问路、票务、取消 |
| 食物与点餐 | 15% | 菜名、食材、烹饪、口味、分量、饮食限制、过敏表述、结账 |
| 学科与课堂 | 35% | 专业词汇、定义、课堂问答、条件、推导表达、代码/公式周围的自然语言 |
| 跨场景困难样本 | 10% | 中英混合、半句、自我修正、跨句指代、歧义、数字与占位符 |

所有类别都使用相同 JSON 目标，不单独设“格式模式”。两个翻译方向初始各半；训练时混合打乱，不先全部训练课堂、再全部训练餐饮。

课堂覆盖计算机/AI、数学/统计、物理/工程、化学、生物/基础医学、经济/管理、人文/社会科学及课堂组织语言。需要翻译的是相关内容，不是让模型回答专业问题。没有相应数据和测试的细分科目，不宣传已经可靠覆盖。

术语放在真实句子和段落里训练，而不是只背词典。需要包含同词不同含义、提问与否定、简称和完整名称。保留足够多不带上下文、不带术语表的正常样本，避免模型依赖这些可选字段才会翻译。

### 单条数据

JSONL 每行保存一个对象；以下为展开的示例：

```json
{
  "id": "example-001",
  "group_id": "lesson-example-a",
  "category": "academic",
  "input": {
    "text": "The p-value is below 0.05.",
    "target_lang": "zh-CN",
    "context": "",
    "glossary": {}
  },
  "output": {"translation":"p 值低于 0.05。"}
}
```

`id`、`group_id`、`category` 只用于数据管理、去重和评测，不传给模型。模型只学习 `input → output`。数据来源、调用模型与审核情况可存独立记录，不能混入答案。

## 5. 用 32B 生成标注

用 Python 调用已选供应商的 Qwen3-32B API，把每条输入转换为候选译文。这里使用现成远程服务；**不需要为 witrans-4b 自己搭建 HTTP 服务**。

标注端和 4B 使用同一份翻译要求。标注端不能看到上线时 4B 不可能得到的隐藏后文，再要求 4B 学会那个答案。

默认要求标注端直接给最终 JSON；关闭 reasoning 的具体参数按供应商文档确认，不把本地 tokenizer 参数直接照搬成所有供应商的 API 参数。即使标注端使用 reasoning，也只把最终译文作为训练答案，不把推理文本放进 target。原始 Qwen3-32B 与 Qwen3-4B 本身支持 thinking / non-thinking，但 API 怎样暴露开关取决于供应商。[R2][R5]

最小标注流程：

```text
读取输入 → 调用 32B → 读取最终输出 → 检查与抽样审校 → 写入 JSONL
```

检查 JSON 只有一个字符串字段 `translation`，没有代码围栏、解释、重复键或截断。检查必要数字、符号和占位符；翻译语义需要另外审核，不能因为格式合法就直接全部收下。不要全局删除译文字段里的 `<think>` 字面量来“清理答案”。

每批按类别和方向人工抽检；遇到否定、过敏表述、数量、专业词义等关键错误，扩大对应类型的审核。32B 审核自己的输出只能辅助，不作为独立质量证明。保存已完成样本，失败有限重试，避免重跑全部数据。

只使用获准训练和外发标注的文本，检查供应商对输出训练用途的约定。费用先通过小批量实际 token 用量估算，本文不预填当前单价或总训练费用。

## 6. 在 8GB 显存上训练

训练继续采用原始 `Qwen/Qwen3-4B`，不静默换成其他同尺寸检查点。固定实际下载 revision，使用一个覆盖全部数据的 LoRA adapter。

初始配置：

| 配置 | 起始值 |
|---|---|
| 方法 | 4-bit NF4 QLoRA + SFT |
| Double quantization | 开启 |
| 计算精度 | 硬件支持时 BF16，否则 FP16 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0 |
| 目标层 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj |
| 微批次 / 梯度累积 | 1 / 16 |
| 总序列长度 | 2048 token 起步，包含输入和答案 |
| Gradient checkpointing | 开启 |
| 学习率 / 优化器 | 1e-4 / AdamW 8-bit |
| 训练轮数 | 先 1 轮；开发集仍有改善再尝试第 2 轮 |
| Loss | 仅最终 JSON 答案及 EOS；其他位置为 -100 |

NF4、量化权重上的 LoRA 和相应准备步骤有 PEFT 支持；这些具体超参数是本项目起点，不是官方最优配方或 8GB 不会 OOM 的保证。[R3]

训练可沿用 Unsloth 的内存优化实现；实际依赖先在显卡上跑通再固定版本。无需同时维护多个训练框架。显存不够时先缩短总长度到 1536/1024 并重新整理样本，再考虑 rank=8；不截掉正确答案。8GB 只说明容量，不说明速度。

### 训练时最重要的编码规则

配套 `witrans.py` 的 `make_messages()` 是统一输入构造函数。使用官方模板并固定 `enable_thinking=False`。模板可能含空的思考前缀，那是输入的一部分，不是要学习的答案。[R2][R6]

以下函数用于已审核、完整的样本；其生成的 labels 不能被训练器重新覆盖。TRL 支持预编码 labels，具体版本仍需检查 loss mask。[R4]

```python
import json
from witrans import make_messages


def encode_example(tokenizer, record, max_length=2048):
    messages = make_messages(**record["input"])
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    )
    answer = json.dumps(
        record["output"], ensure_ascii=False, separators=(",", ":")
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Missing EOS token")
    target_ids = tokenizer.encode(answer, add_special_tokens=False)
    target_ids += [tokenizer.eos_token_id]
    input_ids = list(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        raise ValueError("样本过长，需要重新切分；不能截断答案")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }
```

开训前解码有效 labels，确认只剩答案和 EOS；padding 也设为 -100。先检查少量样本能正常学习，再用代表性最长样本跑反向传播、评估和保存，确认不 OOM。

所有训练来源混合进入同一个 adapter。继续训练时保留代表性旧数据，避免只补一类内容后损坏其他类别；正常保存 checkpoint，而不是每轮反复合并并重新量化。

## 7. 只保留必要的模型评测

另留约 1,000 条开发集及约 1,000 条人工核对测试集作为起点，不计入训练数量。按来源划分，同一课文、会话、模板、改写和反向翻译不跨集合。数量不支持“全学科 99.9%”一类承诺。

固定同样提示词、输入信息、量化和解码，比较原始 Qwen3-4B 与 witrans-4b。32B 的最终译文可以作为参考，但不能自动当作真值。

重点看四件事：

| 检查 | 要回答的问题 |
|---|---|
| 翻译质量 | 是否漏译、增译、错译、改变语气或把提问变成回答？ |
| 学科及关键表达 | 术语词义、否定、条件、数量、单位是否正确？ |
| 输出格式 | 原始生成是否只有合法 JSON？有没有解释或 reasoning？ |
| 本地可用性 | 实际显存、完整译文耗时、连续调用是否稳定？ |

日常、旅行、餐饮和不同学科分别看，不让整体平均掩盖弱项。默认不传场景标签，专门检查“完全不传上下文/术语”时的正常使用效果；另测可选术语是否有帮助。

课堂只测当前文本和已经提供的前文，不提前透露未来内容。用 Python 循环调用做连续短句测试，不建立会话服务。记录首次加载耗时和预热后完整译文耗时，不能用首 token 速度代替完整结果速度。没有实际显卡测试，不承诺固定毫秒数。

失败案例优先从开发集分析；补充新来源训练样本后重新测试。冻结测试集不拿去反复训练或调参。

## 8. Python 直接调用

配套文件是 `witrans.py`。它在当前 Python 进程中加载基础模型和 adapter，直接执行 `generate()`，没有端口、网络请求或后台服务。Transformers 支持直接加载 Qwen3，PEFT 支持为基础模型加载训练好的 adapter。[R2][R7]

调用示例（训练权重准备好后）：

```python
from witrans import WiTrans

model = WiTrans(
    base_dir="./models/Qwen3-4B",
    adapter_dir="./models/witrans-4b/adapter",
)

# 同一个实例翻译点餐、旅行和课堂，不选择模式。
a = model.translate("Could you put the sauce on the side?", target_lang="zh-CN")
b = model.translate("Is the deposit refundable?", target_lang="zh-CN")
c = model.translate("Let x be a positive real number.", target_lang="zh-CN")
d = model.translate("这道菜可以少放一点盐吗？", target_lang="en")

print(a["translation"])
```

依赖为兼容的 CUDA/PyTorch、Transformers、PEFT、bitsandbytes、accelerate。参考代码用 NF4 在加载时量化基础模型，adapter 单独加载；所有模型文件必须已经在本地，所以实际翻译不联网。[R3][R8]

`base_dir` 必须对应训练时相同的原始 Qwen3-4B revision；`adapter_dir` 必须包含真实训练产物。没有 adapter 时明确报错，不能偷偷用原始 4B 冒充 witrans-4b。此入口不读取 GGUF。

默认关闭 thinking，先用贪心解码建立可重复的基线；这只是开发起点，不声称它一定比所有采样设置质量高。不要把内置 non-thinking 设置重新变成用户要选择的场景模式。

返回前严格解析 JSON；格式错误或输出截断就抛 Python 异常，不默默补括号或伪造成功。该检查只维护返回结构，不保证语义正确。训练和质量评测仍然必要。

连续课堂文本同样逐条调用，按需要传最近几句原文作为 `context`。不传历史也能独立翻译；需要消除跨句指代时才传。模型加载一次后复用，不逐句重新加载。

## 9. 最终需要交付什么

最小可用成果：

```text
models/
├── Qwen3-4B/               # 固定 revision 的基础权重和 tokenizer
└── witrans-4b/
    └── adapter/
        ├── adapter_config.json
        └── adapter_model.safetensors
witrans.py                  # 本地调用，以及训练/推理共用提示词
README.md                   # 使用方式、基础 revision、依赖版本、训练及测试说明
```

基础权重加 adapter 共同实现一个逻辑上的 witrans-4b，不代表要调用两个独立模型。adapter 单独不是完整 4B 权重；如需单独分发完整权重，可合并到相同 revision 的基础模型并重新评测，但不把合并作为本地 Python 使用的前置条件。[R7][R9]

训练记录、数据来源及许可需要保留；分发时保留基础模型来源与适用许可证。Qwen3-4B 官方仓库标示 Apache-2.0，但这不替代数据和 API 的使用许可。[R2]

**当前范围到这里结束：数据 → 训练 → 评测 → 权重 → Python 调用。** 不实现 HTTP/WS 服务、账号鉴权、数据库、队列调度、课程管理平台、应用界面或相关扩展路线。

## 10. 技术参考与核查边界

以下一手资料于 2026-09-30 查阅。它们用于核对术语、模板和库接口，不证明本项目已经训练成功，也不构成速度、显存或准确率保证。

| 编号 | 资料 | 用途 |
|---|---|---|
| R1 | [Sequence-Level Knowledge Distillation][R1]，第 3.2 节 | 生成译文再做监督学习的序列级蒸馏依据 |
| R2 | [Qwen3-4B 官方模型卡][R2] | 基础模型、Python 调用、non-thinking、许可证 |
| R3 | [PEFT Quantization][R3] | 量化基础权重上的 LoRA、NF4 与准备步骤 |
| R4 | [TRL SFT Trainer][R4] | SFT 与预编码 labels |
| R5 | [Qwen3-32B 官方模型卡][R5] | 标注模型与原始模型思考开关 |
| R6 | [Qwen3-4B tokenizer 配置][R6] | 模板前缀与 EOS |
| R7 | [PEFT configurations and models][R7] | adapter 保存与本地加载 |
| R8 | [Transformers bitsandbytes][R8] | Python 中的 4-bit 装载 |
| R9 | [PEFT LoRA][R9] | adapter 合并 |

[R1]: https://aclanthology.org/D16-1139/
[R2]: https://huggingface.co/Qwen/Qwen3-4B
[R3]: https://huggingface.co/docs/peft/main/developer_guides/quantization
[R4]: https://huggingface.co/docs/trl/sft_trainer
[R5]: https://huggingface.co/Qwen/Qwen3-32B
[R6]: https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json
[R7]: https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config
[R8]: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
[R9]: https://huggingface.co/docs/peft/main/developer_guides/lora
