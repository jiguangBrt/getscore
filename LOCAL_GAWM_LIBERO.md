# 本机 LIBERO-PRO + 远程 GAWM

2026-10-21：已完成 fake 链路验证，并在用户审核后使用 **真实 Codex CLI（GPT-6 Astra / low）+ GAWM** 完成沙拉酱入篮任务。第二轮在一个连续 episode 内成功，环境原生 `terminated=true`、`truncated=false`。

任务沿用此前成功样例：`libero_object_swap / task 2 / seed 0`，把沙拉酱放入篮子。这里使用 LIBERO-PRO 的 swap 场景，不是标准 LIBERO 的完整 benchmark。

## 权重与部署

来源：`wangyihan@49.233.66.228:/data/wangyihan/checkpoints/gawm_libero_80000_20261018_002850.pt`。

目的地：`user@192.168.50.73:/mnt/reacher-fast/gawm/checkpoints/gawm_libero_80000_20261018_002850.pt`。

权重为 129,355,451 字节；源文件、迁移清单、本地中转副本及远程副本的 SHA-256 一致：

```text
f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13
```

同时复制了对应的配置、归一化统计、VTT 任务向量、DINOv3 资产和推理源码。部署时共 248 个源码/资产文件与本地中转副本逐文件校验一致，记录在 [transfer-verification.json](outputs/_archive/records/gawm-setup-20261021/transfer-verification.json)。输出整理后，本地中转权重、依赖归档及重复源码已删除，来源清单、配置、校验和部署记录保留；50.73 上的部署不受影响。

66.228 上 liuyang 名下的文件仅通过 sudo **读取和复制**，没有修改其代码、配置、权重、Python 环境或实验进程。接入脚本位于本机 `scripts/`，部署文件位于 50.73 的新目录 `/mnt/reacher-fast/gawm/`。

| 组件 | 位置 |
| --- | --- |
| RPent、fake/Codex 规划器、LIBERO-PRO 仿真 | 本机 `/home/jh/getscore` |
| GAWM | 50.73，GPU 2，`ws://192.168.50.73:8115` |
| 本机 GAWM/RPent 适配层 | `http://127.0.0.1:8116`，每轮自动启动、清理 |
| SAM3 | 复用 `http://192.168.50.73:8114`，GPU 1 |

GAWM 服务容器为 `getscore-gawm-libero`，复用已有镜像 `resunet:cu128-fixed` 的 PyTorch 2.11.0+cu130 和 torchvision 0.26.0+cu130；在新目录中隔离 Python 3.10 依赖。已有 OpenPI 的 PyTorch 2.7.1+cu126 不支持 5090，因此未用它执行 GAWM，也没有修改它。GAWM 实测占用约 788 MiB 显存；原 π0.5 和 SAM3 服务继续运行。

复制的源码 `REVISION` 是 `libero-base 4badfc2`，权重训练记录的版本是 `libero-base 4b9cc4c`。这不是训练时完整源码快照的一致性证明；本次直接实例化 GAWM，以 `strict=True` 加载全部 413 项，缺失项和多余项均为零，并验证了真实推理。没有使用通用加载器的“过滤不兼容参数后继续运行”路径。

## 观测和动作契约

- 直接传递 RLinf 已转正的两路 RGB 图像，不再翻转；由原 GAWM `rgb_pixels` 完成 OpenCV linear 缩放到 256×256 和 ImageNet 归一化。
- 8 维状态顺序为 XYZ、轴角、两指关节位置；归一化和动作反归一化复用原 `PolicyNormProcessor` 及 `unified_libero_wm` 注册项。
- GAWM 输出 `[1,8,7]`，每次执行全部 8 步。连续动作不另行缩放；夹爪按照原 StarVLA LIBERO 评测方式，`open > 0.5` 转为 `-1`，否则为 `+1`。
- 使用同步推理。连接超时 20 秒、模型请求超时 120 秒、本地 RPC 超时 150 秒；请求超时不重放动作。

GAWM 使用固定训练任务向量，不支持 π0.5 那样任意改写子指令。环境原始文字是 `Pick the salad dressing and place it in the basket`；策略明确使用词表中的等价文字 `pick up the salad dressing and place it in the basket`。两者分别记录，未知任务文本会报错，不做隐式猜测。

为兼容 RPent，工具名仍是 `pi0_pick` / `pi0_doubled`，但接入后的后端为 GAWM。独立入口 `gawm_rpent_entrypoint.py` 在系统提示中说明后端及上述固定指令；修正移动、夹爪和释放仍使用现有脚本工具。fake 接入时没有修改 RPent 子模块；真实测试中补充修复了本机 `robots/libero/env_server.py` 的 episode 上限处理，见下文。真实规划从独立空记忆目录 `data/liberopro-gawm-codex-memory` 开始，不沿用 π0.5 的成功配方。

## fake 验证与审核材料

最终验证轮次：[liberopro-gawm-fake-20261021-114551](outputs/liberopro-gawm-fake-20261021-114551/)。调试过程的前两轮保留在 `outputs/_archive/runs/`，正式审核以最终轮次为准。

后续命名统一为“仿真名-动作策略名-指挥策略名-时间”：fake 使用 `liberopro-gawm-fake-YYYYMMDD-HHMMSS/`，当前真实指挥模型使用 `liberopro-gawm-gpt6astra-YYYYMMDD-HHMMSS/`。时间为北京时间，精确到秒，不含微秒；同秒重名时报错，不覆盖已有记录。

启动器默认只保存 `outputs/` 原始记录，不自动生成 `review/` Markdown 或 HTML；仅在用户明确要求时整理可读复盘。此前 fake 派生审核文件已清理，以下链接直接指向保留的原始记录。

| 检查 | 结果 |
| --- | --- |
| fake 规划器 | 7 次响应、7 次工具调用，输入/输出 token 均为 0 |
| SAM3 篮子 | 找到掩码，score 0.871，成功反投影世界坐标 |
| SAM3 沙拉酱 | 找到瓶身掩码，score 0.168，显式使用 `min_score=0.1` |
| 脚本工具 | `set_gripper(gripper=-1, steps=1)` 成功 |
| 真实 GAWM | 3 次请求均成功，输出和执行形状均为 `[1,8,7]` |
| 仿真与记录 | 25 个原生动作，5 份观测及两相机图像、深度、相机参数和世界坐标图 |
| 视频 | 256×256、25 帧、20fps、1.25 秒，逐帧解码通过 |
| 环境结果 | `terminated=false`、`truncated=false`；fake 按预设预算 `finish(stuck)` |
| 清理 | 本轮环境子进程、fake 服务和本机适配层已退出；远程模型服务保留 |

这轮验证证明组件连通和实际动作执行，**不代表完成抓取入篮，也不是成功率评测**。fake 脚本只进行了 3 次策略调用，未做任务判断或失败纠正。

最终轮次的往返耗时约为 592 / 80 / 82 ms，服务端推理约为 557 / 32 / 26 ms；首次包含启动后的预热。往返减去推理的差值仍包含序列化、排队和调度开销，不是精确单向网络时延。服务器与本机时钟存在偏差，延迟使用各自的单调计时，不按跨机器时间戳直接相减。

- [原始任务记录](outputs/liberopro-gawm-fake-20261021-114551/transcript_object_swap_t2_s0.json)
- [原始配置与结果](outputs/liberopro-gawm-fake-20261021-114551/configuration.json)
- [视频](outputs/liberopro-gawm-fake-20261021-114551/episode.mp4)
- [篮子分割](outputs/liberopro-gawm-fake-20261021-114551/segment_overlay_00.png/00.png)、[沙拉酱分割](outputs/liberopro-gawm-fake-20261021-114551/segment_overlay_01.png/00.png)
- [逐次时延](outputs/liberopro-gawm-fake-20261021-114551/network-timing.jsonl)

重跑 fake：

```bash
cd /home/jh/getscore
./scripts/run-gawm-libero.sh
```

已有记录可离线重新审核，不调用模型：

```bash
.venv/bin/python -m scripts.verify_gawm_fake outputs/liberopro-gawm-fake-20261021-114551
```

用户已审核并授权真实测试，入口为：

```bash
./scripts/run-gawm-libero.sh --planner codex
```

当前默认配置为 GPT-6 Astra / low，最多 40 次回复、900 秒规划时间、1000 个原生仿真动作；使用已有本机 ChatGPT 登录，配置在 [liberopro-gawm.toml](configs/liberopro-gawm.toml)。初始预算为 24/600/400，真实成功轮次使用了上述扩大后的预算，并已设为后续默认值。fake 的短链路预算不变。

## 真实 Codex + GAWM 成功记录

成功轮次：[liberopro-gawm-gpt6astra-20261021-162733](outputs/liberopro-gawm-gpt6astra-20261021-162733/)。Codex 自行读取初始画面、反投影确认目标和篮子位置、调用 GAWM 抓取，并用脚本工具收紧夹爪、提升、搬运及释放。第一次 GAWM 接近偏离目标后，Codex 在同一个 episode 内重新靠近瓶身，再次调用策略并完成抓取；没有 reset。

- 最终环境 `terminated=true`、`truncated=false`，规划器 `finish(status=success)`；运行器正常退出，`task_success=true`。
- 19 次模型回复、32 次工具调用；8 次真实 GAWM 请求全部成功，执行 64 个策略动作，其余为现有脚本工具动作。
- 共执行 447 个原生动作，保存 13 份观测；完整视频为 447 帧、256×256、20fps，播放时长 22.35 秒，逐帧解码通过。包括模型等待的本轮规划/执行耗时约 240.8 秒。
- SDK 累计输入 1,139,844 token，其中缓存输入 1,061,632；输出 3,519 token，含 reasoning token 210。使用现有 ChatGPT 登录额度；这是多次回复累计值。
- 最终观测的 `release` 返回原生成功标志，图像可见绿色瓶盖的沙拉酱位于篮内。本轮环境进程和本机适配层已退出，远程 GAWM/SAM3/π0.5 继续运行。

`pi0_pick.success` 是每次调用内部重新计算的局部抓取启发式值；本轮最后一次抓取调用因“本次调用内没有重新下降”而仍返回 false。Codex 根据瓶身确实随夹爪提升的图像证据继续搬运，最终是否完成任务仍由环境原生标志确认。

原始材料：[视频](outputs/liberopro-gawm-gpt6astra-20261021-162733/episode.mp4)、[最终画面](outputs/liberopro-gawm-gpt6astra-20261021-162733/agentview_high.png/12.png)、[配置与结果](outputs/liberopro-gawm-gpt6astra-20261021-162733/configuration.json)、[状态与动作](outputs/liberopro-gawm-gpt6astra-20261021-162733/states.json)、[Codex 记录](outputs/liberopro-gawm-gpt6astra-20261021-162733/codex_object_swap_t2_s0.txt)、[原始事件](outputs/liberopro-gawm-gpt6astra-20261021-162733/codex_object_swap_t2_s0.txt.stream.jsonl)。本次没有自动生成新的 Markdown/HTML 复盘文件。

此前一轮 [162125](outputs/liberopro-gawm-gpt6astra-20261021-162125/) 未完成任务：底层 robosuite 的 400 步上限包含初始化的 15 步稳定动作，外层 RLinf 从稳定后开始计数，导致第 385 个任务动作后底层先终止，继续执行时报错并断开 worker。该轮失败记录和视频完整保留，不计作成功。

修复仅在本机 RPent 的环境配置中设置 `init_params.ignore_done=true`，由 RLinf 统一负责 episode 截断；LIBERO 的 `_check_success()` 仍独立返回任务成功标志，未修改成功判据。真实仿真回归将上限设为 2 步，验证初始化稳定阶段正常完成、任务第 2 步返回 `truncated=true`，且 worker 可继续读取最终相机/状态。该回归通过，原有 161 项离线检查也通过。扩大预算后的成功轮次需要 447 个动作，超过原先的 400 步预算。

这是一项任务、一个种子的成功样例，包含前一轮失败尝试，不构成 benchmark 成功率评测。66.228 上 liuyang 名下的代码、环境、模型和进程始终没有改动。

查看或重启 GAWM 服务：

```bash
ssh user@192.168.50.73 'docker logs --tail 40 getscore-gawm-libero'
ssh user@192.168.50.73 'docker restart getscore-gawm-libero'
```

没有配置开机自动启动。远程新目录还保留了推理源码和依赖归档，便于复查及重新部署；完成时 `reacher-fast` 约余 30 GiB。

## 离线检查

```bash
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python -m pytest -q \
  tests/test_remote_gawm.py tests/test_remote_openpi.py \
  RPent/tests/unit_tests/robots/libero \
  RPent/tests/unit_tests/rpent/robots/components/test_pi05_vla_client_contracts.py
```

161 项通过，涵盖 GAWM 图像/状态传递、8 步动作块、夹爪转换、模型错误不执行/不重试及现有 LIBERO/OpenPI 契约。相关脚本的 Ruff 检查、格式检查、语法检查及两个启动入口的帮助检查通过。
