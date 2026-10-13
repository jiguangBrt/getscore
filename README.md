# getscore

RoboTwin / RoboCasa 具身智能冲榜项目：探索 GPT 等高层模型与 π0.5 等动作策略结合的任务执行架构。

## 项目入口

- [路线总览](ROADMAP.md)：任务目标、七步推进、完成条件、榜单区别和首个里程碑。
- [论文索引](paper/README.md)：GPT-Policy、GPT-as-Policy 和 PhyAgentOS 的本地 PDF、论文来源及代码入口。

## 推进流程

明确规则 → 调研方案 → 复现基线 → 实现最小系统 → 按需训练 → 迭代与正式评测 → 打包提交。

首先跑通两个仿真环境的基线，再在一个环境的少量代表任务上验证“大脑 + 小脑”能否带来可重复的提升。

## 当前内容

```text
getscore/
├── README.md
├── ROADMAP.md
└── paper/
    ├── README.md
    ├── GPT-Policy.pdf
    ├── GPT-as-Policy.pdf
    └── PhyAgentOS.pdf
```

当前仓库保存路线规划和调研资料；仿真环境、训练与评测实现将随项目推进补充。
