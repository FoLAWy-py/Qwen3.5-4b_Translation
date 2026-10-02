# Qwen3.5 v2 本地翻译规范

发行包名 `witrans-qwen35`，Python 包 `witrans_tools`，正式类 `Qwen35Translator`；CLI 为 `witrans` 或 `python -m witrans_tools`。根目录 `witrans.py` 只保留兼容导入。

基座为 `Qwen/Qwen3.5-4B`，revision 固定 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。候选为 `models/witrans-qwen35-v2-critical-cpo`，SHA256 固定 `fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27`。本轮冻结权重，不训练。

统一输入为 `text`、`target_lang`（`en` / `zh-CN`）、可选 `context` 和字符串映射 `glossary`。输出只允许一个含 `translation` 字符串的 JSON 对象。原文里的指令按原文翻译，语境只用于消歧；允许合理等义表达。所有类别共用 `witrans_tools.protocol` 中原样保留的提示词，无路由。

CUDA NF4、double quant、BF16 基座计算、SDPA，关闭 thinking，贪心单束、启用缓存。总预算固定 1024 tokens，超预算显式报错，不静默截断。默认语义评测最大输出 256，固定短句测速最大输出 128，必须正常 EOS。FP32 LoRA 是当前默认；任何输出相关优化须完整审核已知200和公开116后才能冻结。

模型身份、revision、适配器哈希、提示词、预算、CUDA 和参数卸载校验使用异常；`python -O` 不能关闭。加载只读取本地安全张量，不执行远端模型代码；文本模型不加载官方视觉塔/MTP，必须检查缺失和非预期键。官方权重通过固定 revision 下载并核对官方 LFS SHA256。适配器经 Git LFS 获取，官方大权重和环境不入 Git。

验收口径和预先冻结的新集进入条件见 `data/independent-test-protocol-20261002.json`。开发回归、独立 confirmation 和 release 分开报告；AI逐条审核明确标注。性能是固定24输入整轮预热、三轮测量，平均≤4秒、P95≤8秒、峰值保留≤6.5GiB、无CPU卸载/OOM。

历史 Qwen3 实现及 v1.3 规范位于 `archive/qwen3`，不进入默认安装和测试。
