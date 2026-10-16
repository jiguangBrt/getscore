# RoboTwin / RoboCasa 实验路线记录

创建日期：2026-10-13；状态更新：2026-10-16。榜单规则最近核查日期：2026-10-13。

**当前处于探索阶段，准备进行第 3 步：搭建仿真环境并复现基线。技术路线尚未定案。**

## 一、任务目标与当前意向

长期目标是在 RoboTwin / RoboCasa 上开展具身智能实验，争取可提交的完整榜单成绩；具体技术路径仍待探索。此前记录的榜单推进顺序为 RoboTwin → RoboCasa，计算资源为同机 8×A100 40GB；规则见 [第 1 步记录](STEP1_RULES.md)。

近期初步参考 **RPent**，尝试复现 **π0.5 + GPT-6**，了解服务如何启动、信息如何在模块间流动，以及任务如何执行。有了实际运行经验后，再考虑是否将 π0.5 替换为更快、更小的模型。

目前未确定：

- 是否采用固定技能接口，技能是否写死，以及接口需要多大灵活性。
- 高层模型与动作模型如何分工、交互和切换。
- 是否需要显式阶段规划、动作审核、结果检查或恢复，以及采用什么实现。
- 最终模型、训练方法和完整评测方案。

这些是探索问题，不是待实现的既定需求。RPent 和其他参考项目中的做法用于理解和比较，不自动成为本项目设计。

## 二、推进阶段

保留原来的七步作为进度框架，后续阶段的内容随探索结果调整：

| 阶段 | 当前记录 |
| --- | --- |
| 1. 明确榜单规则和目标 | 已整理目标、资源及主要规则，仍有执行前待核实项；见 [STEP1_RULES](STEP1_RULES.md) |
| 2. 调研参考项目并形成方案候选 | 已有论文和代码调研记录；候选方案未定案；见 [STEP2_SCHEMES](STEP2_SCHEMES.md) |
| 3. 搭建仿真环境并复现基线 | 准备进行；初步参考 RPent 的 π0.5 + GPT-6；见 [STEP3_BASELINE](STEP3_BASELINE.md) |
| 4. 尝试实验方案 | 待基线复现和观察后确定，不预设技能接口或固定架构 |
| 5. 按需训练或替换模型 | 后续可能方向，是否需要及如何实施尚未确定 |
| 6. 迭代与评测 | 随实验方案确定比较方法和评测配置 |
| 7. 打包提交 | 在方案、结果及榜单条件明确后推进 |

## 三、近期探索范围

- 尝试跑通 RPent 的 π0.5 + GPT-6，观察启动和执行过程。
- 结合代码、日志和视频，了解观测、工具请求、动作与结果反馈。
- 记录实际遇到的问题与想法，再决定哪些机制保留、修改或另做实验。

RPent 源码已在 `RPent/` 子模块中，尚未验证仿真或模型运行成功。本地 RPent 的 π0.5 集成可从 LIBERO / LIBERO-PRO 入手，这是搭建参考，不作为用户已经选定的仿真环境。具体模型版本、规划器、推理强度和任务清单也在实际搭建时确定。

## 四、两个榜单的关键区别

以下是 2026-10-12 核查到的公开规则摘要。正式实施与提交前，应重新确认规则及适用赛道。

| 项目 | RoboTwin 2.0 | RoboCasa365 当前多任务榜单 |
| --- | --- | --- |
| 评测范围 | 50 个双臂任务，分别评测干净场景和随机化场景 | 50 个目标任务，分为 Atomic-Seen、Composite-Seen、Composite-Unseen，并非全部 365 个任务 |
| 数据协议 | 榜单规定 50 个任务，每任务 50 条 `demo_clean` 示范，共 2,500 条 | 榜单说明使用 Human300 预训练数据，未见组合任务评测零样本泛化 |
| 成绩 | 分别报告 clean2clean、clean2random 及其平均值 | 报告三个任务分组的成功率和总体成绩 |
| 提交 | 公开代码、权重和技术报告，向维护者申请收录 | 向榜单仓库提交成绩 JSON 的 PR，提供复现材料，团队核验后收录 |
| 方案约束 | 混合系统及外部模型访问的具体收录要求需核实 | 要求明确的方法或数据创新及可测量的提升；短暂微调或小改动不自动构成新条目 |

**数据网站提供了某类示范，不等于该榜单协议允许使用它训练。** 特别是 RoboCasa 的目标任务数据与当前零样本评测设置，需要区分清楚。

## 五、参考入口

### 榜单与规则

- [RoboTwin 官方榜单与评测、收录要求](https://robotwin-platform.github.io/leaderboard)
- [RoboTwin 官方代码](https://github.com/RoboTwin-Platform/RoboTwin)
- [RoboCasa 官方榜单与评测协议](https://robocasa.ai/leaderboard.html)
- [RoboCasa 榜单提交仓库](https://github.com/robocasa-benchmark/leaderboard)
- [RoboCasa 数据集说明](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)

### 优先调研项目

- [RPent：当前初步参考框架](https://github.com/RLinf/RPent)，本地代码位于 `RPent/`，探索记录见 [第 3 步准备](STEP3_BASELINE.md)。
- [GPT-Policy](https://github.com/cheng-haha/GPT-Policy)
- [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy)
- [银河通用 GPT-as-Policy 相关论文](https://arxiv.org/abs/2609.38537)
- [Harness VLA：RPent 相关论文](paper/Harness-VLA.pdf)，[arXiv v5](https://arxiv.org/abs/2607.08448v5)。

### 执行层与架构参考

- [openpi：π0.5 代码、权重及微调示例](https://github.com/Physical-Intelligence/openpi)
- [XPolicyLab：策略适配与评测接口](https://github.com/XPolicyLab/XPolicyLab)
- [Hi Robot：层级视觉语言动作模型](https://arxiv.org/abs/2502.19417)
- [Code as Policies：通过程序组合机器人 API](https://code-as-policies.github.io/)
- [SayCan：语言规划与可执行技能结合](https://github.com/google-research/google-research/tree/master/saycan)
