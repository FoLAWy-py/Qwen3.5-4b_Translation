# Qwen3.5-4B 中英翻译

本地中英翻译，支持语境和术语表，返回 `{"translation":"..."}`。唯一主线是 Qwen3.5-4B + 冻结 v2 LoRA，NF4 加载，总预算 1024 tokens。旧 Qwen3 实现在 `archive/qwen3`，不进入默认安装和测试。

需要 Git LFS、uv、Python 3.12 和支持 BF16 的 NVIDIA 显卡。本机验证环境为 Windows、RTX 4060 Laptop 8GB。

```powershell
git clone https://github.com/FoLAWy-py/Qwen3.5-4b_Translation.git
cd Qwen3.5-4b_Translation
git lfs install --local
git lfs pull --include="models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors"
uv sync --locked
uv run --locked witrans download
uv run --locked witrans translate "Please send me the receipt." --target-lang zh-CN
```

下载器获取固定官方版本并核对权重哈希；官方大权重、环境、缓存和密钥不进入 Git。

```python
from witrans_tools import Qwen35Translator

translator = Qwen35Translator()
print(translator.translate(
    "The cast has been removed.", "zh-CN",
    context="We are discussing a plaster cast on a healed wrist.",
))
```

默认使用 eager。需要已验证的解码编译配置时：

```powershell
uv sync --locked --extra compile
uv run --locked --extra compile witrans translate "请把收据寄给我。" --target-lang en --runtime decode_compiled
```

固定 24 条短句，整轮预热后测三轮：平均 **3.164 秒**、P95 **4.039 秒**、峰值保留显存 **3.387 GiB**；加载 16.867 秒，预热含编译 164.296 秒。无 OOM 或 CPU 卸载，速度门槛达标。

质量还未达到发布门槛。新原创合成 confirmation 400 条为 **310 pass / 48 minor / 42 major（含 1 critical）**，由 Codex AI 逐条审核，非人工验收；release 600 条按预定门槛未执行，继续封存。已知 200 和公开 116 属于开发材料，不能称实际翻译准确率。

RAG 已完成完整开发质量对照和正式测速：177 条训练术语没有带来语义等级净提升，保持默认配置。新句的 OpenAI 查询向量仍需 API；全本地术语筛选不需要 API。

- [安装与运行](docs/install-and-run.md)：模型身份、CLI、Python 与本机环境说明。
- [本轮报告](docs/takeover-20261002.md)：性能、完整回归、独立确认及证据。
- [RAG 结果与下一步](docs/rag-analysis-20261002.md)：质量、延迟和本地检索的取舍。
- [运行 spec](witrans-qwen35-spec.md) · [验收规则](data/release_acceptance_policy.md) · [历史实验](docs/experiment-report.md)
