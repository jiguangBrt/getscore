# 本地 LIBERO-PRO + 远程 π0.5

本地工作目录为 `/home/jh/getscore`。本机运行 LIBERO-PRO、RPent 及 fake 或 Codex 规划器；服务器 `user@192.168.50.73` 运行官方 π0.5-LIBERO 和 SAM3。当前采用阻塞式同步推理。

## 启动实验

```bash
cd /home/jh/getscore
./scripts/run-liberopro.sh
```

默认执行 `libero_object_swap / task 2 / seed 0`，任务是将 salad dressing 放入 basket。fake 规划器依次发出 3 次 `pi0_doubled(max_chunks=1)`，每次执行模型返回序列中的前 5 个动作，再发出 `finish(status=stuck)`。这是有限步数的链路验证，结束状态不代表任务成功。可增加动作调用次数：

```bash
./scripts/run-liberopro.sh --chunks 40
```

规划器由本机回环地址上的固定脚本提供，不调用付费 GPT API；运行时覆盖 API key 并将 token 计数设为 0。当前规划器只重复任务指令，不提供真正的策略判断。使用隔离的本地 memory，不下载其他模型生成的任务记忆。

配置位于 `configs/liberopro-local.toml`。每次结果写入独立的 `outputs/liberopro-pi05-fake-YYYYMMDD-HHMMSS/`，包含配置、对话记录、状态、视频、日志及 `network-timing.jsonl`。命名统一为“仿真名-动作策略名-指挥策略名-时间”，时间为北京时间，精确到秒。启动入口会清除继承的 ROS/Conda Python 路径，使用本项目 `.venv` 和隔离的 LIBERO 配置。

## GPT-6 Astra / Codex CLI 规划

```bash
cd /home/jh/getscore
./scripts/run-liberopro.sh --planner codex
```

该入口调用 RPent 原生 CLI，关键参数为 `--planner codex --model gpt-6-astra --reasoning-effort low`。RPent 使用现有 Codex Python SDK 启动本机 Codex CLI，通过 RPent 的本地 MCP 工具观察场景和执行动作。未修改 RPent 的规划算法。

RPent 不强制使用 API key：`api`、`codex`、`claude_code` 是不同规划后端。当前选择 `codex`，复用本机已存在的 ChatGPT 登录；启动器检查该登录，并在子进程中移除 API key 和自定义 API 地址变量。登录失效时，在本机运行 `codex login` 完成浏览器登录。Codex 支持 ChatGPT 订阅登录与 API key 登录，使用方式见 [OpenAI 官方认证说明](https://learn.chatgpt.com/docs/auth)。当前方式会消耗账号的 Codex 使用额度。

`configs/liberopro-local.toml` 中 `[codex]` 固定选择 `gpt-6-astra` 和 `reasoning_effort = "low"`。`low` 即启用推理，不需要再加一个 `reasoning=true` 参数；官方模型页列出了 Astra 支持的 [推理强度](https://developers.openai.com/api/docs/models/gpt-6-astra)。规划器保存服务返回的推理摘要和推理 token 统计，是否有可见摘要由模型返回内容决定，不等于保存完整内部思考过程。

当前预算为最多 24 次模型回复、600 秒规划时间、400 个原生仿真步骤。真实规划器自行选择工具和动作次数，不受 fake 模式固定 3 次动作调用的限制。可覆盖预算：

```bash
./scripts/run-liberopro.sh --planner codex \
  --max-turns 24 --planner-timeout-s 600 --max-episode-steps 400
```

使用 `data/liberopro-codex-memory/` 作为独立的本地持久记忆目录，首次创建空占位；没有下载其他模型的任务记忆。预算限制下的一轮实验不构成完整 benchmark 评测。

每轮结果在 `outputs/liberopro-pi05-gpt6astra-YYYYMMDD-HHMMSS/`。指挥策略字段取实际配置的模型名（去掉连字符），而非调用工具 Codex CLI 的名字。除视频、图像、状态和网络时延外，重点查看：

- `codex_*.txt`：模型文字、工具调用摘要及返回的推理摘要。
- `codex_*.txt.stream.jsonl`：SDK 原始事件，便于核对工具调用、错误、token 和推理 token。
- `codex_*.txt.last`：模型最后的文字回复（存在时）。
- `transcript_*.json`：RPent 汇总的规划结果和用量。
- 初始完整 RPent 输入位于 `codex_*.txt.stream.jsonl` 的第一条 `item/completed → payload.item.type=userMessage → content[0].text`；不是在 `transcript_*.json` 的 `messages` 中。可在 `run_viewer` 中查看启动输入、事件时间、工具参数和返回结果及前后画面。
- `configuration.json`：本轮模型、推理强度、完整启动参数及结果；`run_status=finished` 只表示运行结束，`task_success` 取自环境的原生终止标志，两者分别记录。`planner_finish` 是模型的结束声明，需要与环境结果对照。

过程复盘统一使用[本地实验回放网站](run_viewer/README.md)：运行 `./scripts/run-viewer.sh`，打开 http://127.0.0.1:8765，输入 `outputs/` 或某次实验目录。fake 和 Codex 启动器只保存原始产物；不再生成或维护手工 `review/` Markdown、HTML。旧目录已删除，历史材料可从 Git 历史查看。

## 服务与权重

| 组件 | 地址 | 运行位置 |
| --- | --- | --- |
| OpenPI WebSocket | `ws://192.168.50.73:8008` | 服务器 GPU 0 |
| SAM3 HTTP RPC | `http://192.168.50.73:8114` | 服务器 GPU 1 |
| RPent/OpenPI 适配层 | `http://127.0.0.1:8113` | 每次实验自动启动并清理 |

远程项目为 `/mnt/reacher-fast/openpi_ur_pp_202607`，代码在其 `repo/`。复用服务器已有的 `/mnt/reacher-fast/ckpt/pi05_libero`，16 个文件均与官方 `gs://openpi-assets/checkpoints/pi05_libero` 的大小和 CRC32C 一致，有 MD5 的文件另核对 MD5。项目 `checkpoints/openpi-assets/checkpoints/pi05_libero/` 引用这些文件；不要移动被引用的原权重。核对记录在 `outputs/_archive/records/liberopro-setup-2026-10-19/checkpoint-existing-verified.json`。

SAM3 复用 `/mnt/data800/myCodex/assets/models/sam3.pt`。它运行在容器 `getscore-sam3-libero`，环境和代码保存于远程项目的 `services/`，不修改已有 OpenPI 环境。

查看或重启远程服务：

```bash
ssh user@192.168.50.73 'systemctl --user status getscore-pi05-libero'
ssh user@192.168.50.73 'tail -n 30 /mnt/reacher-fast/openpi_ur_pp_202607/logs/pi05-libero-sync.log'
ssh user@192.168.50.73 'systemctl --user restart getscore-pi05-libero'
ssh user@192.168.50.73 'docker logs --tail 30 getscore-sam3-libero'
ssh user@192.168.50.73 'docker restart getscore-sam3-libero'
```

π0.5 使用本次建立的临时 systemd 用户服务，断开 SSH 后继续运行；重启服务器后需要重新启动。若该服务单元已不存在：

```bash
ssh user@192.168.50.73 'systemd-run --user --unit=getscore-pi05-libero --collect --property=WorkingDirectory=/mnt/reacher-fast/openpi_ur_pp_202607/repo --property=StandardOutput=append:/mnt/reacher-fast/openpi_ur_pp_202607/logs/pi05-libero-sync.log --property=StandardError=inherit /mnt/reacher-fast/openpi_ur_pp_202607/services/start-pi05-libero.sh'
```

本次没有重启服务器。`reacher-fast` 磁盘的 UUID 自动挂载已经写入 `/etc/fstab`；上述推理服务没有设为开机自动启动。

## timeout 与 RTC

`timeout` 表示等待上限，不注入人工网络延时。当前配置为：

- `openpi.connect_timeout_s = 20`：建立连接的超时。
- `openpi.request_timeout_s = 300`：一次模型请求的超时，包含首次 JIT 编译。
- `bridge.rpc_timeout_s = 330`：RPent 到适配层的动作预测 RPC 超时，预留模型超时错误返回时间。

网络记录包含完整往返时间、服务端推理时间和两者差值估计。差值包含传输、排队及序列化等开销，不是精确的单向网络延迟。小样本不能代表持续运行时的延迟分布。

服务器已有客户端 `packages/openpi-client/src/openpi_client/websocket_client_policy.py` 原生支持 `connect_timeout_s` 和 `request_timeout_s`。新增适配层复用该客户端，对普通观测和 `rtc_v1` 请求使用同一组超时。超时关闭连接，不自动重放动作请求；下一次明确请求再重连。

远程 `scripts/marvinpro_rtc_smoke.py` 增加 `--connect-timeout-s` 和 `--request-timeout-s`，沿用原 RTC 算法；备份为 `.bak-getscore-20261019`。当前 LIBERO-PRO 执行器只启用同步模式。原 MarvinPro RTC 的 16 维动作约束仍保留；本次没有将其移植为 LIBERO 的 7 维 RTC 执行器。`vla.infer_raw` 可转发已有兼容服务的 RTC 请求。

## 环境和数据

本地 Python 3.11，Torch 2.7.1、NumPy 1.26.4、MuJoCo 3.3.0、robosuite 1.5.2。LIBERO-PRO 的 864 个资产文件已与指定源码的 Git blob 校验值核对。源代码版本见 `configs/liberopro-sources.json`，已安装依赖快照见 `configs/liberopro-environment.freeze.txt`；快照包含本机 editable 路径，迁移机器时需调整路径。RLinf 源码由 `RPENT_RLINF_ROOT` 提供，本地不安装训练栈或 π0.5 权重。

本机 RTX 3060 Laptop 6GB 承担渲染，推理模型在远程 RTX 5090 上。当前虚拟环境约 7.5GiB，源码、资产和安装缓存另占约 4.5GiB；建议本地预留 30–50GiB，再按视频、深度图和点云的保存量增加空间。一次短实验的大小不能直接外推长时间收集的数据量。

完整 benchmark 成功率和 LIBERO 的 RTC 执行尚未评测；2026-10-19 的记录仅涉及 fake 规划器，2026-10-20 的真实 Codex 接入记录单独保存。

## 验证记录（2026-10-19）

真实链路结果保存在 `outputs/liberopro-pi05-fake-20261019-164950/`，自动核对摘要为 `outputs/_archive/records/liberopro-setup-2026-10-19/policy-chain-result.json`：

- fake 规划器发出 4 次响应，输入/输出 token 均为 0。
- 远程官方 π0.5 调用 3 次，执行 15 个原生仿真动作；`episode.mp4` 可解码，含 15 帧，整次输出约 33MiB。
- 保存 4 份状态及对应图像、深度、相机参数和点云。正常记录 `finish(status=stuck)`，原生任务成功为 false；没有进行成功率评测。
- 实验拥有的环境子进程已退出，适配层端口已释放；远程两个模型服务继续运行。

首次模型请求含 JIT 编译，往返约 40.91 秒；后两次约 1.069 秒和 0.316 秒，模型推理分别约 74ms、73ms。另 3 次真实观测组件请求返回有限的 `[10,7]` 动作，往返约 481ms、481ms、374ms，模型推理约 86ms、75ms、74ms。适配层转换为 float32 并执行前 5 个动作。逐次数据见实验的 `network-timing.jsonl` 和 `outputs/_archive/records/liberopro-setup-2026-10-19/remote-components/`。

SAM3 已验证点选及真实场景文本分割。`basket` 在默认 `min_score=0.2` 下得到分数 0.871 的掩码；`salad dressing` 的分数为 0.168，默认阈值会拒绝，单独以 `min_score=0.1` 测试时得到正确瓶身掩码。本次未修改服务默认阈值。远程镜像的 CUDA 13 NVRTC 库搜索路径已补入 `services/remote-sam3-start.sh`。实测模型显存约为 π0.5 8.6GiB、SAM3 8.1GiB，分别位于 GPU 0、1。

离线检查：

```bash
env -u PYTHONPATH -u VIRTUAL_ENV -u CONDA_PREFIX .venv/bin/python -m pytest -q \
  tests/test_remote_openpi.py \
  RPent/tests/unit_tests/robots/libero \
  RPent/tests/unit_tests/rpent/robots/components/test_pi05_vla_client_contracts.py
```

上述 154 项通过，涵盖同步/RTC 协议共用超时、超时不重放动作、图像方向、状态与动作契约、LIBERO 运行时。服务器已有 WebSocket/RTC 相关 10 项测试通过。修改文件的 Ruff 检查、依赖兼容性检查以及中英文 Sphinx 严格构建均通过；中文新增参数在生成 HTML 中的标记已核对。证据在 `outputs/_archive/records/liberopro-setup-2026-10-19/`。没有运行全部机器人测试、Python 多版本矩阵或真实 RTC 任务。

完成时本地根分区约余 569GiB，远程 `reacher-fast` 约余 31GiB。重复下载的临时权重和无用的安装缓存已清理，没有改动原权重或服务器的代理设置。

## Codex 接入检查（2026-10-20）

检查记录在 `outputs/_archive/records/liberopro-codex-setup-2026-10-20/`。Codex Python SDK 与其捆绑 CLI 均为 0.163.0，确认认证类型为 ChatGPT；模型目录包含 `gpt-6-astra`、图像输入与 `low` 推理强度。使用现有登录发出的最小真实请求返回 `RPENT_ASTRA_OK`，耗时约 8.37 秒。

RPent 现有 Codex 契约测试 51 项、本地 LIBERO/通信适配检查 154 项通过；新启动入口的帮助、脚本语法及 Ruff 检查通过。本轮没有修改 RPent 的 Codex 后端，也没有修改用户的全局 Codex 配置或复制登录凭据。

首轮真实规划结果保存在 `outputs/liberopro-pi05-gpt6astra-20261020-104215/`，核对摘要为 `outputs/_archive/records/liberopro-codex-setup-2026-10-20/policy-chain-result.json`：

- `libero_object_swap / task 2 / seed 0` 在一个连续 episode 内完成抓取并将 salad dressing 放入 basket，没有 reset；最终环境 `terminated=true`、`truncated=false`，模型记录 `finish(status=success)`。
- 19 次模型回复、36 次工具调用；远程 π0.5 请求 16 次均成功。第一次抓取没有成功，规划器观察画面后改用完整任务指令抓取，再用移动与释放工具完成放置。
- 执行 344 个原生动作，保存 12 份状态和可解码的 344 帧视频；按 20fps 播放为 17.2 秒，包含模型等待时间的整轮运行约 482 秒。
- SDK 累计记录输入 1,114,334 token，其中缓存输入 1,024,640；输出 3,282 token，输出中的 reasoning token 为 194。这是多次请求的累计量，不是单次上下文长度；使用 ChatGPT 登录额度，不能按 API 单价直接推算费用。
- 环境子进程已退出，本轮适配层和 MCP 监听已关闭；远程 π0.5、SAM3 服务继续运行。

这是一项任务、一个种子的成功样例，未做多任务或多种子成功率统计。`object_swap_t2_s0.json` 是模型保存的任务审计，`object_swap_t2_s0_recipe.jsonl` 是 RPent 保存的成功动作配方；判断成功仍以环境原生标志及状态/视频证据为准。

动作数按逐动作录制的帧数核对：当前 `move_to.steps_used` 统计轨迹采样次数，包含到位后只检查、不执行动作的最后一个样本；本轮 7 次移动共有 7 个这样的样本，直接累加工具计数会得到 351，扣除后为 344。具体核对写在上述摘要的 `tool_step_count_note` 中。
