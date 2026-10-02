"""Write concise mainline documentation without changing frozen runtime code."""
from pathlib import Path

README='''# Qwen3.5-4B 中英翻译

本地中英翻译，支持语境和术语表，返回 `{"translation":"..."}`。正式主线使用 `Qwen/Qwen3.5-4B` 和冻结的 v2 LoRA，NF4 加载，输入加输出总预算为1024 tokens。本轮只整理工程、优化推理并验证质量，不开展新训练。

发行包名为 `witrans-qwen35`，Python 包为 `witrans_tools`，入口是 `Qwen35Translator` 和 `witrans` CLI。旧Qwen3专用实现、脚本和spec在 `archive/qwen3`，不进入默认安装和测试。通用提示词、JSON解析、审核和统计工具共用一份实现。

## 安装与调用

需要Git LFS、uv、Python3.12和支持BF16的NVIDIA CUDA显卡。本机实测为Windows / RTX4060 Laptop 8GB。首次克隆：

```powershell
git clone https://github.com/FoLAWy-py/Qwen3.5-4b_Translation.git
cd Qwen3.5-4b_Translation
git lfs install --local
git lfs pull --include="models/witrans-qwen35-v2-critical-cpo/adapter_model.safetensors"
uv sync --locked
uv run --locked witrans download
uv run --locked witrans translate "Please send me the receipt." --target-lang zh-CN
```

下载器固定官方revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` 并核对官方权重哈希。v2适配器SHA256为 `fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27`。官方大权重、环境、缓存和秘密不进入Git。

```python
from witrans_tools import Qwen35Translator

translator = Qwen35Translator()
result = translator.translate(
    "The cast has been removed.", "zh-CN",
    context="We are discussing a plaster cast on a healed wrist.",
    max_new_tokens=256,
)
print(result)
```

默认eager、dynamic cache、BF16基座计算、FP32 LoRA，关闭thinking、贪心解码。身份、revision、adapter哈希、prompt、CUDA、预算和卸载检查使用显式异常，在 `python -O` 下仍有效。超预算不会静默截断。

## 推理优化

Windows安装可选编译依赖后，选择已完整验证的解码编译配置：

```powershell
uv sync --locked --extra compile
uv run --locked --extra compile witrans translate "请把收据寄给我。" --target-lang en --runtime decode_compiled
```

prefill保持eager，单token解码用Inductor和固定1024静态缓存；每个请求创建新缓存。FP32 LoRA、提示词和解码规则冻结。整模型编译成本过高，已撤弃该方案。

固定24条短句，整轮预热后正式测三轮，共72次：

| 配置 | 平均 | P95 | 峰值保留显存 |
|---|---:|---:|---:|
| 原环境eager复核 | 6.638秒 | 8.634秒 | 3.342 GiB |
| 冻结解码编译 | **3.164秒** | **4.039秒** | **3.387 GiB** |

编译配置加载16.867秒，整轮预热164.296秒（包含编译），吞吐9.731 tokens/秒。正文20–120 tokens、输出不超过128 tokens；全部JSON/EOS正常，无OOM或CPU参数。正式速度满足门槛，首次调用成本另计。逐输入证据见[接手报告](docs/takeover-20261002.md)。

## 质量与证据范围

完整重跑已知200和公开116，并由Codex AI逐条审核。沿用历史评分口径，编译配置分别为 **166/17/17** 和 **105/8/3**（pass/minor/major），critical均为0；已知集仍有一条语言方向错误。合理歧义与惯用表达的重审另列，并对旧输出采用同一口径，不把改评分算成模型改善。

这些集合已经用于开发，不能把通过比例称为实际翻译准确率。新confirmation / release协议在新输出前冻结；至少400条/200来源组与600条/300来源组，来源、参考、许可、语境、歧义和历史隔离先审后测。合成来源与AI审核会明确标注，不称独立人工验收。新集合执行状态见[接手报告](docs/takeover-20261002.md)，尚未宣布发布通过。

详细步骤与本机CPU隔离限制见[安装说明](docs/install-and-run.md)；运行约束见[Qwen3.5 spec](witrans-qwen35-spec.md)；既有训练及开发成绩见[历史实验报告](docs/experiment-report.md)。数据许可按逐条来源与 [data/README.md](data/README.md)处理。
'''

def main():
    Path('README.md').write_bytes(README.encode('utf-8'))
    path=Path('docs/install-and-run.md');text=path.read_text(encoding='utf-8')
    text=text.replace('编译与 LoRA BF16 仍是需要完整质量回归的实验参数，默认不会自动启用。','FP32 LoRA解码编译已完成完整开发语义回归和正式测速，作为明确选项提供。LoRA BF16未完成GPU质量验证。')
    text=text.replace('--runtime compiled','--runtime decode_compiled')
    text=text.replace('它也属于实验选项，是否符合质量及性能要求以完整实测报告为准。','正式24句三轮平均3.164秒、P95 4.039秒，预热164.296秒含编译。完整旧集合回归已完成；不能替代新来源验收。')
    text=text.replace('uv run --locked python -m pytest -q','uv run --locked python -m scripts.low_cpu_run --module pytest -q')
    text+='\n本机未限制CPU的测试出现过Windows非法指令退出（0xc000001d）。本轮验证采用上述进程内隔离工具，将亲和性限于可用核的最后4个、线程数设为2，不修改系统设置。隔离下88项测试及最小CLI翻译通过；其他CPU组合稳定性未证明。推理同样可用 `python -m scripts.low_cpu_run --module witrans_tools.cli translate ...` 启动。正式测速采用相同隔离。\n'
    path.write_bytes(text.encode('utf-8'))

if __name__=='__main__':main()
