# GAWM 论文阅读与服务器实现对照

阅读日期：2026-10-20。原文为 [GAWM.pdf](GAWM.pdf)，固定版本 [arXiv:2609.03565v2](https://arxiv.org/abs/2609.03565v2)，5 页；作者 Muyuan Liu、Yue Huang、Zheng Liang、Xiang Gao。论文公布代码为 [zsibot/gawm](https://github.com/zsibot/gawm)。

## 论文方法

- JEPA 在潜在空间预测动作执行后的观测表征，不重建图像。
- 逆动力学 IDM 从相邻表征预测已执行动作；状态对齐 SA 从连续表征预测物理状态。物理测量仅用于训练监督。
- 编码器是随机初始化的 ViT-Tiny/14，输出 192 维表征；部署使用目标图像和 CEM 动作序列规划。
- 轨迹更“直”不一定意味着规划更好，还需检查表征变化是否集中在低维子空间。

以上对应原文第 2–5 页。第 3 页 Table I 的 SA+IDM 成功率如下：

| TwoRoom | Reacher | PushT | OGBench-Cube |
| ---: | ---: | ---: | ---: |
| 100% | 85% | 98% | 87% |

结果平均于三个独立训练种子，每个种子评测 50 个固定起点—目标问题；不能直接与服务器上单个权重的历史日志混用。

## 与 49.233.66.228 上代码的关系

以下结合论文和 2026-10-19 的只读调查记录作出判断，未重新运行代码：

| 服务器实现 | 与本论文的关系 |
| --- | --- |
| `/data/liuyang/code/gawm`，产物位于 `/data/liuyang/gawm` | 仓库说明中的论文标题、代码入口、四个任务和 CEM 规划与原文对应；此次未逐项审计源码是否严格复现论文配置。 |
| `/data/liuyang/gawm-exp*` 及 gaoxiang 的 StarVLA 实验快照 | 已调查的主要配置采用 DINOv3、未来特征监督和 ACT 动作头，不能按同名 GAWM 视为本论文的直接复现。 |

此前记录的 LIBERO 93.25%–94.70% 是服务器上 StarVLA 实验的成绩，不是这篇论文的实验结果。

本地调查报告：`outputs/_archive/records/server-audit-2026-10-19/GAWM-StarVLA调查报告.md`。下载版本和文件校验记录见 [归档快照](../rules/2026-10-20/gawm_paper_snapshot.json)。
