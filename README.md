# getscore

RoboTwin / RoboCasa 具身智能冲榜项目：由 GPT 完成场景适应、阶段规划、结果检查与重新规划，由 π0.5 等 VLA 提供夹取、打开、移动、放置等可调用动作能力，并通过执行层落实安全约束。

## 项目入口

- [路线总览](ROADMAP.md)：任务目标、七步推进、完成条件、榜单区别和首个里程碑。
- [第 1 步记录](STEP1_RULES.md)：已确定的目标、榜单规则、计算资源及后续核实项。
- [第 2 步方案](STEP2_SCHEMES.md)：三篇论文技术路线、公开代码范围、方案候选及第一轮实验建议。
- [论文索引](paper/README.md)：GPT-Policy、GPT-as-Policy 和 PhyAgentOS 的本地 PDF、论文来源及代码入口。

## 推进流程

明确规则 → 调研方案 → 复现基线 → 实现最小系统 → 按需训练 → 迭代与正式评测 → 打包提交。

两个榜单都做，先 RoboTwin、后 RoboCasa。先在 RoboTwin 跑通基线并验证“大脑 + 小脑”，随后迁移到 RoboCasa，分别完成完整评测和提交材料。计算资源为同机 8×A100 40GB。

第 1—2 步的目标、规则梳理与方案调研已完成，下一步搭建环境并复现基线；具体评测配置与混合系统收录条件仍需在后续执行前核实。候选方案尚未得到运行验证。

## 当前内容

```text
getscore/
├── README.md
├── ROADMAP.md
├── STEP1_RULES.md
├── STEP2_SCHEMES.md
├── rules/
│   └── 2026-10-13/
│       └── repository_snapshot.json
└── paper/
    ├── README.md
    ├── GPT-Policy.pdf
    ├── GPT-as-Policy.pdf
    └── PhyAgentOS.pdf
```

当前仓库保存路线规划和调研资料；仿真环境、训练与评测实现将随项目推进补充。
