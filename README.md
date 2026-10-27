# getscore

面向 RoboTwin / RoboCasa 的具身智能实验项目。目前处于探索阶段，初步参考 RPent 框架；具体技术路线和接口设计尚未确定。

## 项目入口

- [路线记录](ROADMAP.md)：长期目标、当前探索意向、推进阶段和榜单区别。
- [第 1 步记录](STEP1_RULES.md)：已确定的目标、榜单规则、计算资源及后续核实项。
- [第 2 步方案](STEP2_SCHEMES.md)：三篇论文技术路线、公开代码范围及未定案的方案候选。
- [第 3 步准备](STEP3_BASELINE.md)：近期复现意向、需要了解的问题和待确定事项。
- [本地 LIBERO-PRO 运行说明](LOCAL_LIBEROPRO.md)：本机仿真、远程官方 π0.5/SAM3、fake 或真实 Codex 规划器、超时配置和启动命令。
- [GAWM + LIBERO-PRO 接入](LOCAL_GAWM_LIBERO.md)：50.73 上的 GAWM 服务、fake 验证、真实 Codex 完成沙拉酱入篮的记录与启动入口。
- [本地实验回放网站](run_viewer/README.md)：后续实验过程统一使用 `run_viewer` 复盘，不再维护手工整理的 `review/`。输入 outputs 路径，查看四项统计、每个 Turn 的公开事件、工具调用和前后画面；启动 `./scripts/run-viewer.sh`。
- [论文索引](paper/README.md)：GPT-Policy、GPT-as-Policy 和 RPent / Harness VLA 的本地 PDF、论文来源及代码入口。
- [Kimi 机器人任务调查](KIMI_ROBOTICS_SURVEY.md)：相关论文、模型与推理强度、对 GPT 的评测差距及比较限制。

## 推进流程

明确规则 → 调研方案 → 复现基线 → 实现最小系统 → 按需训练 → 迭代与正式评测 → 打包提交。

此前记录的长期目标是 RoboTwin → RoboCasa。当前先用本机运行 LIBERO-PRO 和 RPent，远程 `192.168.50.73` 提供官方 π0.5-LIBERO 与 SAM3。启动入口默认使用不消耗 token 的 fake 规划器；加 `--planner codex` 可使用现有 ChatGPT 登录调用 GPT-6 Astra，推理强度为 low。该服务器实查为 6×RTX 5090 32GB；此前的 8×A100 40GB 记录不适用于这次部署。

**当前状态（2026-10-21）：已接入 RPent 原生 Codex 规划器，GPT-6 Astra/low 分别搭配远程官方 π0.5 和 GAWM 完成同一个 LIBERO-PRO 任务。** `libero_object_swap / task 2 / seed 0` 的沙拉酱入篮均获得环境原生成功标志。GAWM 首轮因 episode 上限错位失败，修复后第二轮在连续 episode 内成功；视频和原始记录已保存。尚未统计完整 benchmark 成功率。

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
