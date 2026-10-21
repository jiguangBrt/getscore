# outputs 目录索引

2026-10-21 已整理。实验统一命名为 `仿真名-动作策略名-指挥策略名-YYYYMMDD-HHMMSS`，北京时间精确到秒。

`pi05` / `gawm` 表示真实动作策略，`mockvla` 表示模拟动作接口；`fake` 表示固定脚本指挥，`gpt6astra` 表示 GPT-6 Astra 指挥（由 Codex CLI 调用）。模型、推理强度、任务和种子仍以每轮配置为准。

```text
outputs/
├── liberopro-pi05-fake-20261019-164950/
├── liberopro-pi05-gpt6astra-20261020-104215/
├── liberopro-gawm-fake-20261021-114551/
├── liberopro-gawm-gpt6astra-20261021-162125/
├── liberopro-gawm-gpt6astra-20261021-162733/
└── _archive/
    ├── runs/       # 早期模拟 VLA 检查及前两轮 GAWM 调试
    └── records/    # 保留的部署、权重来源、校验及调查记录
```

## 顶层实验

| 目录 | 内容 |
| --- | --- |
| [liberopro-pi05-fake-20261019-164950](../outputs/liberopro-pi05-fake-20261019-164950/) | fake 指挥 + 真实 π0.5，3 次推理、15 个动作，链路验证，未完成任务。 |
| [liberopro-pi05-gpt6astra-20261020-104215](../outputs/liberopro-pi05-gpt6astra-20261020-104215/) | 真实 GPT-6 Astra/low + π0.5，沙拉酱入篮成功，环境 terminated=true。 |
| [liberopro-gawm-fake-20261021-114551](../outputs/liberopro-gawm-fake-20261021-114551/) | 最终 GAWM 链路验证，含 SAM3、3 次真实推理和 25 个动作；未完成任务。 |
| [liberopro-gawm-gpt6astra-20261021-162125](../outputs/liberopro-gawm-gpt6astra-20261021-162125/) | 首轮真实 GPT + GAWM 未完成任务，初始化稳定步数与底层 episode 上限错位导致 worker 断开；失败记录保留。 |
| [liberopro-gawm-gpt6astra-20261021-162733](../outputs/liberopro-gawm-gpt6astra-20261021-162733/) | 修复边界并扩大预算后，真实 GPT-6 Astra/low + GAWM 在连续 episode 内完成沙拉酱入篮；8 次策略请求、447 个动作，环境 terminated=true。 |

## 早期调试

移至 [outputs/_archive/runs](../outputs/_archive/runs/)：

- `liberopro-mockvla-fake-20261019-164642`：模拟 VLA 的早期检查，不是 π0.5 测试。
- `liberopro-gawm-fake-20261021-114320`：首轮真实 GAWM 检查。
- `liberopro-gawm-fake-20261021-114445`：检查 GAWM 提示适配入口的调试轮次。

六轮实验均完整保留，只移动目录，522 个原始文件的内容逐文件校验未变。历史日志、配置和原始模型事件中的旧路径保留为当时的真实记录；查询当前路径请使用 [清理与迁移清单](output-cleanup-20261021.json)。派生复盘材料已更新为新路径。

## 审核入口

- 首轮真实成功实验：[数据流与时间轴](liberopro-pi05-gpt6astra-20261020-104215/task-flow.html)、[可读记录](liberopro-pi05-gpt6astra-20261020-104215/transcript_object_swap_t2_s0.md)、[原始提示词导出](liberopro-pi05-gpt6astra-20261020-104215/initial-prompt.txt)。
- 最新 GAWM fake：[原始配置与结果](../outputs/liberopro-gawm-fake-20261021-114551/configuration.json)、[原始记录](../outputs/liberopro-gawm-fake-20261021-114551/transcript_object_swap_t2_s0.json)、[视频](../outputs/liberopro-gawm-fake-20261021-114551/episode.mp4)；此前派生审核文件已清理。
- 真实 GAWM 成功：[视频](../outputs/liberopro-gawm-gpt6astra-20261021-162733/episode.mp4)、[最终画面](../outputs/liberopro-gawm-gpt6astra-20261021-162733/agentview_high.png/12.png)、[原始配置与结果](../outputs/liberopro-gawm-gpt6astra-20261021-162733/configuration.json)；未生成新的 Markdown/HTML 复盘。
- 启动说明：[π0.5](../LOCAL_LIBEROPRO.md)、[GAWM](../LOCAL_GAWM_LIBERO.md)。

`outputs/<实验目录>/` 保存原始运行产物；`review/<实验目录>/` 保存可读记录、HTML 及审核数据。两者的实验目录名一致。

默认每轮只保存 `outputs/`。可读 Markdown 和 HTML 仅在用户明确要求时生成，不随实验自动创建。

## 实验内部文件

| 文件或目录 | 用途 |
| --- | --- |
| configuration.json | 本轮模型、服务、参数及结果；运行结束与任务成功分开记录。 |
| transcript_*.json | 规划器消息、工具调用及返回结果。 |
| states.json | 初始及动作后的观测、原生终止标志、动作参数与结果。 |
| episode.mp4 | 实际仿真动作视频。 |
| network-timing.jsonl | 逐次动作策略请求时延。 |
| agentview*.png/、wrist*.png/ | 双相机图像，内部的 00.png 等按观测记录编号。 |
| *_depth.npz/、*_world*.npz/、*_metadata.json/ | 深度、世界坐标和相机参数。 |
| segment_*.json/、segment_overlay_*.png/ | SAM3 分割结果和可视化（存在时）。 |
| codex_*.txt、.stream.jsonl、.last | 真实 Codex 的文字、原始事件及最后回复（存在时）。 |
| fake-memory/ | fake 实验的空记忆占位。 |

部分带 .png、.npz、.json 后缀的条目是目录；编号表示观测记录号，不等于原生动作数。判断任务是否成功需查看环境终止标志，不能仅看 completed 或模型 finish 声明。

## 技术记录与删除项

必要记录在 [outputs/_archive/records](../outputs/_archive/records/)，这些目录不是仿真实验，沿用原记录名称：

- gawm-setup-20261021：权重来源、配置、严格加载、迁移校验和远程最终状态；删除本地重复权重、源码及传输归档。
- liberopro-setup-2026-10-19：最终环境和远程模型检查、官方权重校验、资产清单及必要验证日志。
- liberopro-codex-setup-2026-10-20：真实规划成功核对、认证/模型检查和契约测试记录。
- 其他 gawm-*、robotwin-checkpoints-*、starvla-weights-*、server-audit-2026-10-19、server-status-*：历史权重选型、模型调查和来源证据，不代表服务器实时状态。

删除了过时的 server-setup-2026-10-19 和补充调查 server-audit-2026-10-20，以及失败安装残留、早期环境失败捕获和可重建的 Sphinx 网页。共释放约 392 MiB；删除明细见 [清理清单](output-cleanup-20261021.json)。
