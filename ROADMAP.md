# RoboTwin / RoboCasa 实验路线记录

创建日期：2026-10-13；状态更新：2026-11-02。榜单规则最近核查日期：2026-10-13。

**当前进入第 4 步：讨论抓取、放置与杯子扶正技能的强化学习实验方案。第 3 步已有模型接入、真实任务和批量评测记录；尚未开始本轮训练环境搭建或模型更新。**

## 一、任务目标与当前意向

长期目标是在 RoboTwin / RoboCasa 上开展具身智能实验，争取可提交的完整榜单成绩；具体技术路径仍待探索。此前记录的榜单推进顺序为 RoboTwin → RoboCasa，计算资源为同机 8×A100 40GB；规则见 [第 1 步记录](STEP1_RULES.md)。

近期已用 **语言模型 + RPent + LIBERO-PRO** 接入 **π0.5 / GAWM** 并开展批量评测。下一步意向是在两种动作模型基础上强化抓取、稳定放置及侧翻杯子扶正，允许抓杯壁或杯柄，不限定抓柄；实验实施方案仍在讨论。训练资源更新为 `49.233.66.228` 的 GPU 0–3，服务器代码与环境均独立放在 `/data/wangyihan`，不借用其他用户的代码或环境。

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
| 3. 搭建仿真环境并复现基线 | 已有 π0.5 / GAWM 真实链路及 120 项批量评测记录；结果不属于统一预算的严格配对复现，见 [STEP3_BASELINE](STEP3_BASELINE.md) 和 [README](README.md) |
| 4. 尝试实验方案 | 当前阶段：讨论批量 RL 闭环、技能任务与奖励、RLinf 适用性；尚未实施，见 [STEP4_RL_SKILLS](STEP4_RL_SKILLS.md) |
| 5. 按需训练或替换模型 | 意向为强化 π0.5 / GAWM 的抓取、稳定放置和扶正；待第 4 步确定任务、资源与训练路径后实施 |
| 6. 迭代与评测 | 随实验方案确定比较方法和评测配置 |
| 7. 打包提交 | 在方案、结果及榜单条件明确后推进 |

## 三、近期探索范围

- 明确抓取、稳定放置、侧翻杯子扶正的任务分布、成功条件和技能边界。
- 讨论环境采样、策略推理、模型更新及权重同步构成的批量 RL 闭环。
- 评估 RLinf 作为训练底座的适用性，以及 π0.5 与当前 GAWM 各自需要的训练适配。
- 保留语言模型 + RPent 的系统路线，讨论低层技能训练和完整系统验证如何衔接。

RPent 源码位于 `RPent/` 子模块，现有实验环境为 LIBERO-PRO，π0.5 / GAWM 均已有真实运行验证。当前推理部署与评测口径见 [README](README.md)，第 4 步的提议和待讨论问题见 [STEP4_RL_SKILLS](STEP4_RL_SKILLS.md)。用户本轮仅要求更新进度并讨论方案，没有要求开始部署或训练。

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
