# 论文索引

初次归档日期：2026-10-13；RPent 论文于 2026-10-16 从 arXiv 下载并归档。当前第 2 步调研的三篇主要论文如下，PDF 均保留原始内容。

| 项目 | 本地 PDF | 论文标题 | 论文来源 | 代码入口 |
| --- | --- | --- | --- | --- |
| GPT-Policy | [GPT-Policy.pdf](GPT-Policy.pdf) | In-Context Robot Learning with VLM Agents | [arXiv:2609.19138](https://arxiv.org/abs/2609.19138) | [cheng-haha/GPT-Policy](https://github.com/cheng-haha/GPT-Policy) |
| GPT-as-Policy | [GPT-as-Policy.pdf](GPT-as-Policy.pdf) | Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies | [arXiv:2609.38537](https://arxiv.org/abs/2609.38537) | [anonymous-report-421/GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy) |
| RPent / Harness VLA | [Harness-VLA.pdf](Harness-VLA.pdf) | Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents | [arXiv:2607.08448v5](https://arxiv.org/abs/2607.08448v5) | [官方 RLinf/RPent](https://github.com/RLinf/RPent)；[本地 fork](https://github.com/jiguangBrt/RPent) |

GPT-Policy 和 GPT-as-Policy 的本地 PDF 均为 arXiv v1；Harness VLA 为 v5（2026-09-24），共 42 页。RPent 是框架名称，Harness VLA 是相关论文名称。在线论文和代码可能继续更新，应按实验记录固定实际参考版本；论文与第三方代码的权利和使用条款归各自作者或项目所有。

技术路线、代码可复用范围及方案比较见 [第 2 步调研](../STEP2_SCHEMES.md)。前两个项目的公开代码版本见 [2026-10-13 参考项目快照](../rules/2026-10-13/reference_project_snapshot.json)；RPent 的本地代码提交、论文来源和 PDF SHA-256 见 [2026-10-16 RPent 快照](../rules/2026-10-16/rpent_snapshot.json)。

此前归档的 [PhyAgentOS.pdf](PhyAgentOS.pdf)（[arXiv:2607.16636v1](https://arxiv.org/abs/2607.16636v1)）作为历史参考保留；当前第 2 步已将其内容替换为 RPent / Harness VLA。历史核查记录保留在原日期快照中。

## GAWM 补充参考（2026-10-20）

- [GAWM.pdf](GAWM.pdf)：*Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning*，固定为 [arXiv:2609.03565v2](https://arxiv.org/abs/2609.03565v2)，2026-09-19 修订，共 5 页。PDF 保留 arXiv 原始内容。
- 论文公布的代码：[zsibot/gawm](https://github.com/zsibot/gawm)。
- [阅读与服务器实现对照笔记](GAWM_READING_NOTE.md)；[下载来源、版本及 SHA-256](../rules/2026-10-20/gawm_paper_snapshot.json)。

[专家示范包中文导读](DEMO_PACKAGE_GUIDE.md)：已下载作者发布的十份 RoboDojo 示范，结合“把物品装入盒子”展示图片、阶段说明、机器人状态和动作参考，并解释换布局后为什么不能直接重放坐标。
