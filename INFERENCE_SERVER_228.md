# 66.228 推理服务迁移

迁移日期：2026-10-28。旧公司服务器已收回，旧服务器的半小时连通检查已停止。

本机 `/home/jh/getscore` 保留 LIBERO-PRO、RPent、Kimi 3.1pro 登录和实验产物。新服务器为 `wangyihan@49.233.66.228`，所有新增推理文件位于 `/data/wangyihan/getscore-inference-20261028/`。原实验目录 `outputs/long120_1027` 不变。

## 组件与来源

| 组件 | 源码、权重与依赖 |
|---|---|
| π0.5-LIBERO | `/home/jh/openpi_sync`，commit `3ab5477b975d5683b2368d7c4cb97e83cb86e36f`；官方 `gs://openpi-assets/checkpoints/pi05_libero`，16 个对象、12,439,085,481 字节，已全部下载并逐文件核对 CRC32C 以及存在时的 MD5。PaliGemma tokenizer 来自本机保留的原服务器备份。已在 GPU 1 启动，16 步真实仿真与 RPent 完整短流程均验证通过。 |
| GAWM | `/data/wangyihan/checkpoints/gawm_libero_80000_20261018_002850.pt`，SHA-256 `f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13`。DINOv3、VTT 向量、归一化统计和推理源码按原部署清单重建并逐文件校验。训练启动工具 `scripts/run_ablation_local.py` 已变化且不参与推理，因此未复制。 |
| SAM3 | 源码沿用本机 `data/sam3-service`，commit 见 `configs/liberopro-sources.json`；RPent RPC 实现来自本机。官方原版权重已下载、迁移并在本机和远程核对 SHA-256，3,450,062,241 字节，SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。2026-10-28 已在 GPU 0 启动，真实 swap 场景的篮子分割验证通过，score 0.8359375。最新状态见 `sam3-supervisor.json` 与 `sam3-verification.json`。 |

OpenPI 依赖从 `/data/liuyang/code/openpi/.venv/lib/python3.11/site-packages` **只读复制**到自己的 `openpi/.venv`，使用自己的 Python 3.11，并将 editable 路径指向本次复制的源码。关键版本：Torch 2.7.1+cu128、JAX 0.5.3。

GAWM/SAM3 依赖从 `/data/gaoxiang/Code/starVLA/.venv/lib/python3.10/site-packages` **只读复制**到自己的 `vision310.venv`。关键版本：Torch 2.8.0+cu128、Transformers 4.57.0、NumPy 1.26.4；在该副本内补装 SAM3 所需依赖。与旧部署 Torch 2.11.0+cu130 存在版本差异，当前验证以严格权重加载和真实推理为准。

不修改同事的源码、权重、环境、进程、GPU 全局设置。模型服务启动时采样三次 GPU 显存及利用率，要求平均利用率不超过 35%，按采样期间最大显存占用排序；为服务预算之外预留至少 8 GiB 显存。服务以较低 CPU 优先级启动，限制 CPU 线程和自身显存预算；负载不合适时等待，不终止其他进程。

## 服务入口

服务器只监听回环地址：π0.5 `18008`、GAWM `18115`、SAM3 `18114`。本机运行以下脚本建立转发，配置见 `configs/liberopro-228.toml`：

```bash
./scripts/connect-inference-228.sh
```

启动服务前，先确认端口未被本次服务占用；已有正常实例无需重复启动。新服务的 GPU 选择、PID、命令和日志分别保存到服务器的 `records/` 与 `logs/`。

```bash
ssh 49.233.66.228 'python3 /data/wangyihan/getscore-inference-20261028/services/start_migrated_inference.py gawm'
ssh 49.233.66.228 'python3 /data/wangyihan/getscore-inference-20261028/services/start_migrated_inference.py pi05'
ssh 49.233.66.228 'python3 /data/wangyihan/getscore-inference-20261028/services/start_migrated_inference.py sam3'
```

SAM3 启动前必须将获准下载的 `sam3.pt` 放到服务器本次目录的 `checkpoints/sam3.pt`。π0.5 启动前必须存在完成校验后生成的 `checkpoints/pi05_libero/download-manifest.json`。

## 验证与后续 13 项实验

证据目录：`output/investigations/server-migration-20261028/`。GAWM 的 `gawm-smoke/verification.json` 记录真实 swap 仿真中的 2 次推理、16 次动作及 16 帧可解码视频；该检查不使用 SAM3 或规划器，不代表任务成功率。π0.5 的同类检查写到 `pi05-smoke/`。`pi05-supervisor.json` 记录其下载、部署和验证进度；下载结束采用进程退出事件推进，不需模型持续轮询。

SAM3 的 `sam3-supervisor.json` 记录官方权重下载、本机和远程 SHA-256 校验、等待低负载 GPU、启动及分割验证进度。成功后保存 `sam3-verification.json` 与篮子掩码，完成事件只回调一次。下载权限生效后无需用户另外传递 Token。

SAM3 启动采用原独立服务文件入口，避免导入机器人注册表及远程推理环境不需要的规划器依赖；保留启动器按 UUID 选择的 GPU，不用 CLI 再覆盖设备编号。首次启动失败的记录保留在证据目录，修复后的完整权重与校验结果已复用，无需重新下载。

SAM3 的文本分割和点选分割均已通过真实场景测试，分别见 `sam3-verification.json`、`sam3-point-verification.json`。GAWM + SAM3 的 RPent 完整短链路也已通过：`gawm-rpent-smoke/verification.json` 记录 6 次脚本规划工具调用、两次真实 GAWM 推理、两次 SAM3 分割与世界坐标定位、17 次原生动作和 17 帧视频。该测试使用本地脚本规划器，付费模型调用为零；短测试未完成任务，不作为成功率评测。

该完整短测试可用 `configs/liberopro-gawm-228.toml` 运行：

```bash
./scripts/run-liberopro.sh --config configs/liberopro-gawm-228.toml --chunks 2
```

π0.5 的 RPent 完整短流程已通过：`pi05-rpent-smoke/verification.json` 记录 3 次脚本工具调用、两次真实推理、10 次原生动作和 10 帧视频，付费模型调用为零。三项推理服务和两种策略的完整短链路均已就绪。

2026-10-28，原定 13 项 Kimi 3.1pro / 2000 步重跑已全部完成并核验，新增 3 项成功。报告位于 `outputs/swap-kimi-2000step-20261027/report.md`，最终核验位于该目录 `final-audit/verification.json`。8 项初始画面与旧记录不完全一致；原批次对应任务的 6 种方法也各有不同初始画面，已记录该环境的固定家具布局随机性，不能将结果视为严格固定画面的配对评测。以下为历史启动命令，不应重新运行已有有效结果：

```bash
.venv/bin/python -u scripts/run_swap_kimi_2000.py \
  --batch outputs/swap-kimi-2000step-20261027 \
  --inference-config configs/liberopro-228.toml
```

选择仍为 Kimi+π0.5 任务 0、3、4、5、6、8、9，以及 Kimi+GAWM 任务 0、3、5、6、8、9。模型 `kimi-code/k3.1-pro`、high，seed 0、初始状态 0；从头重跑，每项上限 2000 动作，规划时间预算 7200 秒。正式任务开始时重新注册一次性完成回调，结束后再汇报；不恢复旧服务器的周期检查。

原部署细节仍保留于 `LOCAL_LIBEROPRO.md`、`LOCAL_GAWM_LIBERO.md` 和 `outputs/_archive/records/`。
