"""Compose the review report from preserved evidence; never run or select models."""
import json
from pathlib import Path
from witrans_tools.common import fingerprint

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    base='runs/takeover-20261002'
    perf=load(base+'/decode-final/summary.json')
    dev=load(base+'/development-final/semantic-summary.json')
    freeze=load('data/independent-20261002-frozen/freeze.json')
    pre=load('data/independent-20261002-v2-drafts/root-source-review-receipt.json')
    result=Path(base+'/confirmation/semantic-summary.json')
    if result.exists():
        r=load(result);n=r['counts'];ci=r['intervals']
        status=f'''confirmation 全部400条已实际推理并由Codex AI逐条审核：**{n['pass_count']} pass、{n['minor']} minor、{n['major']} major（含{n['critical']} critical）**。JSON {n['json_valid']}/400，方向 {n['language_correct']}/400，EOS {n['ended']}/400。明确约束保留 {r['context']['retention']:.2%}。

按来源组重采样10000次，pass的95%区间为 {ci['pass_bootstrap_95']}，major为 {ci['major_bootstrap_95']}。这是该均衡合成测试范围内的区间，不能外推为用户流量上的实际准确率。

| 类别×方向 | pass | minor | major（含critical） | n |
|---|---:|---:|---:|---:|
'''
        for k,s in r['strata'].items():
            c=s['counts'];status+=f"| {k} | {c['pass_count']} | {c['minor']} | {c['major']} | {c['rows']} |\n"
        status+='\n'
        for label in ('context','multisentence'):
            c=r[label]['counts'];status+=f"{label} 子集：{c['rows']}条、{c['source_groups']}组，{c['pass_count']}/{c['minor']}/{c['major']}（pass/minor/major）。分项与子集的来源组区间均保存在语义摘要。\n\n"
        failed=[k for k,v in r['gates'].items() if not v]
        if failed:
            status+='预先冻结门槛未通过：'+', '.join(failed)+'。**release未执行**；600条发布数据及来源审核已冻结，但没有候选输出或发布判定。未据confirmation修改权重、提示词或运行配置，未开展纠错训练。\n'
        else:status+='confirmation达到预先冻结的release进入门槛；release执行结果另列，不能以confirmation代替发布验收。\n'
    else:status='confirmation已启动，逐输入输出正在保存，语义判定尚未完成；release尚未执行。运行进度见 `runs/takeover-20261002/confirmation/progress.json`。\n'
    report=f'''# Qwen3.5单主线、性能与独立验证

本轮未训练、未推送、未宣布发布通过。保留v2权重；旧v7不存在，没有用历史成绩替代新集合上的直接对照。

本地开始时没有`.git`。只读远端HEAD核对为`0759b195ecddfac1e59732bb84a505cbde5f494f`，恢复Git元数据并以mixed reset建立差异基准，保留工作文件。初始工作树与该提交一致。Git身份为`FoLAWy-py <ljp2219819716@gmail.com>`，改动留本地供审核。

## 模型与安装

适配器`models/witrans-qwen35-v2-critical-cpo`，SHA256 `fe983cd436a3d2672e33071bb8e466a4e1ed2f64b76961c836880d8d5f8dfb27`。官方`Qwen/Qwen3.5-4B`固定revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；两片官方权重按官方LFS哈希重新核验。提示词hash `16465a8a3a3ef925155322e6fbd13df95a1c9348ece436eee5428646cbf7ba28`。身份取证见[model-freeze](../runs/takeover-20261002/model-freeze.json)。

包名`witrans-qwen35`，Python包`witrans_tools`，正式入口`Qwen35Translator`和`witrans` CLI。旧Qwen3实现、spec、142个脚本及专用测试归档于`archive/qwen3`，退出默认安装/测试。通用提示词、JSON解析与生成逻辑共用一份。原生成方法AST对照仅有默认输出预算512→256；正式测速显式128，质量评测显式256。

迁移环境`.venv-qwen35-mainline`完成`uv sync --locked`、导入、CLI最小实际翻译与wheel/sdist检查；原`.venv-qwen35`保留。安装步骤见[安装说明](install-and-run.md)，调用与约束见[spec](../witrans-qwen35-spec.md)。官方权重、环境、缓存、秘密不进Git或安装包。

加载、revision、adapter实哈希/元数据、prompt、CUDA/BF16、总预算、加载键及CPU/disk卸载均使用显式异常。`python -O`失败路径检查有效。CPU隔离下88项测试通过。本机未限制CPU曾出现非法指令0xc000001d及异常运行，最终未隔离uv build也出现访问冲突0xC0000005；原生构建改用相同进程内CPU隔离复核，记录见`final-build-receipt.json`。测试和GPU驱动进程均用`low_cpu_run`限制最后4个可用核、线程2，不修改系统设置。其他CPU组合的稳定性未证明，不推定已查明硬件原因。

## 性能与方案选择

以已有固定24句选择方案，输入hash `f4057624ca5a024d3bf286b0d0acb37fcacfd6273250ae9979535d64c7c570f9`。整轮预热，再测三轮共72次。正文20–120 tokens，输出≤128，输入+输出预算1024，NF4双重量化。

| 配置 | 平均秒 | P95秒 | 输出tokens/秒 | 峰值保留GiB |
|---|---:|---:|---:|---:|
| 原uv环境eager复核 | 6.637856 | 8.633615 | 4.632520 | 3.341797 |
| 冻结单token解码编译 | {perf['mean_seconds']:.6f} | {perf['p95_seconds']:.6f} | {perf['output_tokens_per_second']:.6f} | {perf['peak_reserved_gib']:.6f} |

最终配置：BF16基座、FP32 LoRA、SDPA、贪心、prefill eager，仅单token解码Inductor，dynamic=False，固定1024静态缓存且每个请求新建缓存。保持eager精度转换/除法舍入，关闭HF自动编译。加载{perf['load_seconds']:.6f}秒，整轮预热{perf['warmup_seconds']:.6f}秒包含编译，峰值allocated {perf['peak_allocated_gib']:.6f} GiB；全部JSON/EOS正确，无OOM或CPU参数。正式速度、显存门槛满足；首次成本单独报告。

历史11.6839秒与本轮eager 6.637856秒的差异没有匹配负载/时钟控制，不算优化因果收益。完整forward编译首输入预热518.767秒，第二prefill持续二十多分钟。三次py-spy采样及图结构显示FakeTensor/SymPy符号简化与gated-delta三角逆展开；据实质编译成本和算术结构撤弃该配置，核对自启动PID/命令/创建时间后结束，证据在`compiled/abort-receipt.json`。未因观察工具超时重启或终止GPU任务。解码原型曾测3.029秒，但缺`@wraps`导致额外全prefill logits，修正后的完整测量才是最终数字。BF16 LoRA未经GPU质量验证，未选择。

逐输入耗时、input/output tokens、原始译文、加载/预热成本、配置、显存分别见[baseline](../runs/takeover-20261002/baseline-observed/summary.json)及[最终性能](../runs/takeover-20261002/decode-final/summary.json)；各目录有`per-input.csv`和逐调用JSONL。性能摘要生成时标记semantic pending；之后完成的语义证据独立保存在development-final，不回写历史测速记录。

## 完整开发质量回归

冻结解码配置实际重跑已知200与公开116；Codex AI读完全部原文、语境和预测，审核角色、否定、条件、术语及方向。按历史口径分别166/17/17和105/8/3（pass/minor/major）；critical均0，JSON/EOS全正确，已知语言方向199/200。相对历史164/19/17仅两条minor→pass，原始输出变化10/3条，没有新增major。

五条无消歧语境的合理解释或惯用表达单独重审，并对旧新输出使用同一评分：旧166/21/13，新168/19/13。重审造成的分数变化不算模型改善。公开不变。真实已知错误仍含机械press误作新闻、file打磨语境误译并方向错误、issue未遵循问题语境。实际旧输出、同口径配对统计与来源组区间见[完整语义回归](../runs/takeover-20261002/development-final/semantic-summary.json)。这些集合已用于开发，不称实际翻译准确率或泛化证明。

## 新来源、冻结与独立性范围

协议先于新输出冻结：[测试协议](../data/independent-test-protocol-20261002.json)。提前取消缺失v7对照与无关新训练/多种子修复门槛；保留用户的全部质量和性能阈值，均衡不加虚构流量权重。最终运行冻结hash `{fingerprint(load(base+'/runtime-freeze.json'))}`，在新来源编写前冻结，之后未调参。

先编写500个不同原创场景卡，5类别各100组；前40组/类为confirmation，后60组/类为release，每组双向。500组均为AI合成，非独立人工数据。经用户明确允许，仅场景卡及服务自身新生成双语材料送DeepInfra Qwen/Qwen3-32B，旧语料、候选输出、秘密未外发；服务不提供模型revision。许可按[DeepInfra条款第7节](https://deepinfra.com/terms)与逐条原创来源记录，许可审核不推定AI作品获得人类版权。

第一次泛泛编写产生70组，发现参考误译、虚构语境及近似重复后，在任何候选新输出前全部撤弃，保留`independent-20261002-drafts/rejected-before-output.json`。第二流程绑定明确场景，服务自己的审核仅作建议；根Codex AI亲自读完500组双语、场景及历史近邻，修正条件反转、角色、遗漏和专业术语，移除未审核背景，逐条适用术语由Codex AI添加并双向核对。数据编写/参考审核流程未读取候选测试输出；修订记录见[root source review](../data/independent-20261002-v2-drafts/root-source-review-receipt.json)。

历史隔离清单在来源准备前冻结，含347文件、11009个规范化原文/参考。逐组阅读历史最近邻，最终对全部旧文本及新组做规范化精确与RapidFuzz≥65检索；发现的同类食材洗涤近改写及旧机会成本定义在输出前重作不同场景，不靠换名凑来源。最终精确重合零、阈值候选零。字符串检索用于找候选，来源家族判断仍由Codex AI阅读完成；不能据此保证所有潜在语义关联均被发现。AI合成、同一AI参考审核及有限场景采样限制仍须报告。

| 集合 | 条数 | 来源组 | 每类别×方向 | 明确术语约束 | 实际多句 |
|---|---:|---:|---:|---:|---:|
| confirmation | 400 | 200 | 40 | {pre['contextual_terms']['confirmation']} | {pre['actual_multisentence']['confirmation']} |
| release | 600 | 300 | 60 | {pre['contextual_terms']['release']} | {pre['actual_multisentence']['release']} |

两套来源组不重叠，双向同组，均不入训练或性能选型。逐条来源许可/署名/参考/约束/歧义/历史审核与数据hash见[frozen data](../data/independent-20261002-frozen/freeze.json)。冻结时间 `{freeze['at']}`，在confirmation加载和推理前；数据冻结hash `{fingerprint(freeze)}`。

## 独立确认与发布进入判定

{status}

最终审核校准将两条葱类泛化、以及一条存在性表达欠明确由初判major改为minor；接受未明确矛盾的合理表达，门槛不变，记录在`confirmation/review-calibration.json`。不调整模型、提示词、运行配置或冻结数据。

critical为`confirmation-hard-019-zh-CN`：凸轮塞改成快挂，凸轮瓣接触岩面改成嵌入岩石。基于保护器材与安置条件混淆可能引导错误攀岩保护的具体情境，Codex AI计为critical并包含在major中。保护器材区分及其危险后果依据[American Alpine Club保护系统说明](https://publications.americanalpineclub.org/articles/13201213455/know-the-ropes-protection)和[Wild Country凸轮塞说明](https://www.wildcountry.com/en-us/climbing-friends)；危险性分级是AI审核判断，供用户复核。

逐条格式、方向、EOS及语义分别记录；合理等义表达可通过，不用字符串匹配自动判语义。审核人为Codex AI，非独立人工验收。来源组bootstrap区间只描述此测试；critical同时计入major。新输出不用于优化；若未来据它改权重、提示词或影响输出的运行配置，该集合须转回归并另建独立测试。
'''
    Path('docs/takeover-20261002.md').write_bytes(report.encode('utf-8'))
    if result.exists():
        path=Path('README.md');text=path.read_text(encoding='utf-8')
        old='新集合执行状态见[接手报告](docs/takeover-20261002.md)，尚未宣布发布通过。'
        n=load(result)['counts'];new=f"原创合成confirmation 400条实际审核为{n['pass_count']}/{n['minor']}/{n['major']}（pass/minor/major），由Codex AI逐条判定，非人工验收。执行与release进入判定见[接手报告](docs/takeover-20261002.md)，尚未宣布发布通过。"
        if old in text:path.write_bytes(text.replace(old,new).encode('utf-8'))
        else:
            import re
            revised=re.sub(r'原创合成confirmation 400条实际审核为\d+/\d+/\d+',
                f"原创合成confirmation 400条实际审核为{n['pass_count']}/{n['minor']}/{n['major']}",text)
            path.write_bytes(revised.encode('utf-8'))

if __name__=='__main__':main()
