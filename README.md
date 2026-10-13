# getscore

RoboTwin / RoboCasa 具身智能冲榜项目：探索 GPT 等高层模型与 π0.5 等动作策略结合的任务执行架构。

## 项目入口

- [路线总览](ROADMAP.md)：任务目标、七步推进、完成条件、榜单区别和首个里程碑。
- [第 1 步记录](STEP1_RULES.md)：已确定的目标、榜单规则、计算资源及后续核实项。
- [论文索引](paper/README.md)：GPT-Policy、GPT-as-Policy 和 PhyAgentOS 的本地 PDF、论文来源及代码入口。

## 推进流程

明确规则 → 调研方案 → 复现基线 → 实现最小系统 → 按需训练 → 迭代与正式评测 → 打包提交。

两个榜单都做，先 RoboTwin、后 RoboCasa。先在 RoboTwin 跑通基线并验证“大脑 + 小脑”，随后迁移到 RoboCasa，分别完成完整评测和提交材料。计算资源为同机 8×A100 40GB。

第 1 步的目标与规则梳理已完成，下一步进入方案调研；具体评测配置与混合系统收录条件仍需在后续执行前核实。

## 当前内容

```text
getscore/
├── README.md
├── ROADMAP.md
├── STEP1_RULES.md
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
