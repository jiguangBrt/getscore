# getscore

面向 RoboTwin / RoboCasa 的具身智能实验项目。目前处于探索阶段，初步参考 RPent 框架；具体技术路线和接口设计尚未确定。

## 项目入口

- [路线记录](ROADMAP.md)：长期目标、当前探索意向、推进阶段和榜单区别。
- [第 1 步记录](STEP1_RULES.md)：已确定的目标、榜单规则、计算资源及后续核实项。
- [第 2 步方案](STEP2_SCHEMES.md)：三篇论文技术路线、公开代码范围及未定案的方案候选。
- [第 3 步准备](STEP3_BASELINE.md)：近期复现意向、需要了解的问题和待确定事项。
- [论文索引](paper/README.md)：GPT-Policy、GPT-as-Policy 和 RPent / Harness VLA 的本地 PDF、论文来源及代码入口。

## 推进流程

明确规则 → 调研方案 → 复现基线 → 实现最小系统 → 按需训练 → 迭代与正式评测 → 打包提交。

此前记录的目标是 RoboTwin → RoboCasa，计算资源为同机 8×A100 40GB。近期先参考 RPent，尝试复现 π0.5 + GPT-6，了解启动信息流和执行过程；具体仿真环境及运行配置在搭建时确定。

**当前状态（2026-10-16）：探索阶段，准备进行第 3 步——搭建仿真环境并复现基线。** 初步参考 RPent，后续有尝试更快、更小动作模型的意向。是否采用固定技能接口、如何规划和控制动作均待探索。RPent 源码已作为子模块引入，尚未验证运行成功；参考框架的现有设计不代表本项目已经采纳。

## 当前内容

```text
getscore/
├── README.md
├── ROADMAP.md
├── STEP1_RULES.md
├── STEP2_SCHEMES.md
├── STEP3_BASELINE.md
├── RPent/                     # RPent 代码子模块
├── data/                      # 本地下载资料，不纳入 Git
│   └── gpt-policy-robodojo-contexts/
├── rules/
│   └── 2026-10-13/
│       └── repository_snapshot.json
└── paper/
    ├── README.md
    ├── GPT-Policy.pdf
    ├── GPT-as-Policy.pdf
    ├── Harness-VLA.pdf         # RPent 相关论文，arXiv v5
    └── PhyAgentOS.pdf          # 历史参考
```

`data/gpt-policy-robodojo-contexts/` 是此前调研 GPT-Policy 时下载的 RoboDojo 专家示范包，约 37 MB，包含 10 份任务示范、480 张图片及阶段、状态和动作参考。它用于阅读推理上下文，当前 RPent 代码没有引用该目录，也不能替代 RPent 所需的仿真资产、π0.5/SAM3 权重或 RPent memory。详见 [专家示范包中文导读](paper/DEMO_PACKAGE_GUIDE.md)。
