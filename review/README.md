# 实验复盘

`outputs/<实验目录>/` 保存每轮运行原本产生的日志、状态、视频、图像和模型写出的结果；`review/<实验目录>/` 保存事后整理的 Markdown、HTML、提示词导出及分析数据。复盘工具只读取原始数据，不在 `outputs/` 中增加文件。

首轮真实 Codex 实验：[数据流与时间轴](liberopro-pi05-gpt6astra-20261020-104215/task-flow.html)、[可读任务记录](liberopro-pi05-gpt6astra-20261020-104215/transcript_object_swap_t2_s0.md)、[初始提示词](liberopro-pi05-gpt6astra-20261020-104215/initial-prompt.txt)。整理数据位于同目录的 `task-flow-data.json`。

已有原始记录的用途说明见 [outputs 目录索引](outputs-index.md)。

fake 和 Codex 启动器默认只保存 `outputs/` 原始记录，不自动创建 `review/` 可读版。仅在用户明确要求时生成 Markdown 或 HTML；已有复盘材料保留。按需手动生成可读记录：

```bash
.venv/bin/python -m scripts.render_libero_transcript outputs/<实验目录>
```

本次指定历史实验的 HTML 可重新生成：

```bash
.venv/bin/python -m scripts.build_libero_run_report outputs/liberopro-pi05-gpt6astra-20261020-104215
```

两种整理操作均不调用模型。
