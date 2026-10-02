# 安装与运行

首次安装见 [README](../README.md)。发行包名 `witrans-qwen35`，Python 包 `witrans_tools`，正式入口为 `witrans` CLI 和 `Qwen35Translator`。从仓库根目录运行，模型相对路径按当前目录解析。

固定模型身份：

- 基座：`Qwen/Qwen3.5-4B`。
- revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 适配器：`models/witrans-qwen35-v2-critical-cpo`。
- 适配器 SHA256：`fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27`。

`witrans download` 核对官方权重的 LFS 哈希并生成基座清单。适配器须经 Git LFS 下载；指针文本无法通过加载校验。

```powershell
uv run --locked witrans translate "The cast has been removed." --target-lang zh-CN --context "A plaster cast on a healed wrist."
uv run --locked python -m witrans_tools translate "请把收据寄给我。" --target-lang en
uv run --locked witrans translate "The encoder is ready." --target-lang zh-CN --glossary '{"encoder":"编码器"}'
```

默认 NF4 + double quant、BF16 基座、FP32 LoRA、SDPA、dynamic cache，关闭 thinking，贪心单束。输出预算默认 256；正式短句测速为 128。输入加输出不得超过 1024，超预算直接报错。模型身份、哈希、CUDA、预算和卸载检查在 `python -O` 下仍有效；`translate` 要求合法单一 JSON 和正常 EOS。

可选 `--runtime decode_compiled` 使用 eager prefill、单 token 解码编译和每请求新建的 1024 静态缓存，安装命令见 README。首次编译成本另计，缓存位于 `.cache/qwen35`。BF16 LoRA 未完成质量验证，不作为已验证配置。

本轮迁移在新环境 `.venv-qwen35-mainline` 验证，保留原 `.venv-qwen35`。首次克隆直接用默认 `.venv`；若需隔离环境：

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$PWD/.venv-qwen35-mainline"
uv sync --locked --extra compile
uv run --locked --extra compile python -m scripts.low_cpu_run --module pytest -q
```

本机原生进程曾出现非法指令和访问冲突。实测使用 `low_cpu_run` 将进程限制到最后四个可用核、线程数 2，不修改系统设置；93 项 CPU 测试、最小实际调用及包构建通过。推理也可用 `python -m scripts.low_cpu_run --module witrans_tools.cli translate ...` 启动。其他 CPU 组合和平台尚未实测。
