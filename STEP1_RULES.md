# 第 1 步：榜单规则和目标

核查日期：2026-10-13（北京时间）

本文件保留 2026-10-13 的目标与规则核查记录。当前进度见 [ROADMAP](ROADMAP.md)：截至 2026-10-16 仍处于技术探索阶段，初步参考 RPent，准备搭建环境并复现基线。用户此前确认先 RoboTwin、后 RoboCasa；具体技术路线、训练方案和完整评测配置尚未确定。

## 已确定的目标

用户选择：**两个榜单都做，先 RoboTwin，再 RoboCasa；优先获得可提交的完整榜单成绩**。完整评测、实质方法贡献、复现材料和收录条件共同构成目标。

已知计算资源：**同一台服务器上 8 张 A100，每张 40GB 显存**。训练/API 预算和服务器内存、存储配置尚未记录。

当时的训练建议（未定案）：先使用公开预训练策略权重，在规定数据上适配，完成小脑基线后比较混合系统。RoboTwin 优先研究一个策略联合覆盖 50 个任务；RoboCasa 按 Human300 多任务设置推进，保持未见组合任务零样本条件。基础权重的最终选择及收录资格尚未确认。

资源判断：足以开始基线复现、微调和评测，不等于保证榜首。OpenPI 官方给出的单卡参考为推理 >8GB、LoRA 微调 >22.5GB、全参数微调 >70GB；40GB 单卡不能直接套用后者，多卡全参数微调需使用支持的分片方案并实测显存。多张卡不会自动合成一块显存，具体支持取决于 JAX/PyTorch 实现和所选版本。[OpenPI 资源说明](https://github.com/Physical-Intelligence/openpi#requirements)。

区分训练层次：公开 π0.5 基础权重已接受大规模机器人预训练，项目微调可以复用这些能力，无需自行重做预训练。RoboCasa 配置虽然名为 pretrain_human300，官方 π0 / π0.5 配置仍从相应 base checkpoint 初始化，因此数据集名称中的“预训练”不意味着所有权重随机初始化。[基础权重说明](https://github.com/Physical-Intelligence/openpi#model-checkpoints)、[RoboCasa 训练配置](https://github.com/robocasa-benchmark/openpi/blob/main/src/openpi/training/config.py)。

## 两个实验在做什么

共同流程：仿真环境给出任务指令和观测，策略系统输出机器人动作，环境判定整项任务成功或失败；重复运行后统计成功率。本项目关注这些环境中的任务执行与评测，具体策略组织和对照方式仍待探索。

**RoboTwin 2.0：双臂操作。** 例如左臂抓起红色方块，交给右臂，再放到蓝色垫子；或者按颜色顺序堆叠三块积木。重点包括操作精度、双臂协调，以及背景、光照和干扰物变化后的稳定性。官方例子：[双臂交接](https://robotwin-platform.github.io/doc/tasks/handover_block.html)、[三块积木堆叠](https://robotwin-platform.github.io/doc/tasks/stack_blocks_three.html)。

**RoboCasa365：厨房日常操作。** 基础技能包括抓放、开门和操作按钮；多阶段活动包括补充厨房用品、制作咖啡等。可以研究任务分解、技能顺序和阶段切换。这些官网演示用于理解平台，不代表每个演示都在当前 50 个榜单任务中。[官方演示](https://robocasa.ai/)。

## 已核实的主要规则

| 项目 | RoboTwin 2.0 | RoboCasa365 当前多任务榜单 |
| --- | --- | --- |
| 任务 | 50 个双臂任务 | 50 个目标任务：18 Atomic-Seen、16 Composite-Seen、16 Composite-Unseen |
| 参考训练协议 | Aloha-AgileX，每任务 50 条 demo_clean，共 2,500 条 | Human300 预训练数据；未见组合任务零样本评测 |
| 测试场景 | demo_clean 与 demo_randomized | 预训练厨房，区别于目标数据的 10 个留出厨房 |
| 回合数 | 每任务每种场景 100 次；完整一套共 10,000 回合 | 尚需从正式评测入口核对 |
| 成绩 | clean2clean、clean2random 及二者平均值 | 三组成绩与 50 个任务的总体平均成功率 |
| 提交入口 | 榜单指定邮箱；当前适配文档另要求策略接入 PR 和复现 checkpoint | submissions/ 中添加 JSON 并提交 PR |

来源：[RoboTwin 榜单](https://robotwin-platform.github.io/leaderboard)、[RoboCasa 榜单](https://robocasa.ai/leaderboard.html)、[多任务评测说明](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html)。RoboCasa 分组任务数不同，Overall 不应简单平均三个分组。

RoboTwin 要求公开代码、权重和技术报告。外部闭源 GPT API 如何满足这些要求，公开规则未明确；不能据此认定允许或禁止。[收录要求](https://robotwin-platform.github.io/leaderboard)。

RoboCasa 要求实质方法或数据创新、超出噪声的提升、基础模型归属说明与论文链接；小幅微调不自动获得新条目资格。当前默认环境版本是 1.0.1。闭源提交也需要给团队验证访问，API 混合系统的具体验证方式待核实。[提交仓库](https://github.com/robocasa-benchmark/leaderboard)。

RoboCasa 另有目标任务示范，尤其 Composite-Unseen 示范不能直接用于训练后继续声称当前零样本成绩。Human300 是此前记录的参考设置，尚未据此确定训练方案；额外数据、标注和预训练模型的适用边界需明确。[数据说明](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)。

## 官网观看建议

1. [RoboTwin 首页](https://robotwin-platform.github.io/)：查看 `50 RoboTwin 2.0 Bimanual Tasks` 和 `Domain Randomization`，留意两只手如何配合，以及场景变化。
2. [RoboCasa 首页](https://robocasa.ai/)：查看 `Training Foundational Robot Skills` 和 `Generating Composite Tasks with LLM Guidance`，对比基础操作与补充用品、制作咖啡等多阶段演示。
3. 观看时判断：更想解决抓取、双臂交接和稳定性，还是多步骤规划、阶段切换和失败恢复？两边都有连续动作和执行难题，这只是选择研究重点的参考。

## 版本候选与后续核实项

代码版本候选记录在 [repository_snapshot.json](rules/2026-10-13/repository_snapshot.json)。尚未安装运行，不代表已经验证兼容性或得到榜单认可。RoboTwin 固定的 XPolicyLab 子模块与最新版本不同，不能无记录替换。RoboCasa 早期 π0.5 提交详情标记 1.0.0，需在最终所选版本重新测基线，不能直接用历史成绩作为公平对照。[π0.5 提交详情](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/pi05_2026-04-02.md)。

- [x] 明确主要目标：可提交的完整榜单成绩。
- [x] 初查任务范围、参考数据、主要指标与提交入口。
- [x] 记录代码版本候选。
- [x] 确定先 RoboTwin、后 RoboCasa；记录 8×A100 40GB 同机配置。

以下项目随第 2—7 步推进，在相关训练、正式评测或提交前完成：

- [ ] 记录训练/API 预算、服务器内存与存储，验证训练实现的资源占用。
- [ ] 固定兼容版本、机器人配置、资产和完整任务清单。
- [ ] 核对正式回合数、逐任务动作上限、允许观测、随机种子及异常处理。
- [ ] 固定基础权重与数据清单，核实额外数据和预训练模型边界。
- [ ] 确认外部 GPT / 混合系统的收录与复核要求。
- [ ] 冻结基线与候选系统的共同预算、开发/最终评测划分及提交材料。

待向主办方核实：API 大脑和公开小脑权重是否可收录；复核访问方式及费用；大脑可用的观测与反馈；额外数据和交互边界；认可的代码版本、动作预算与贡献条件。目前仅记录问题，尚未联系主办方或获得答复。
