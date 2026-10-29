# RPent / LIBERO-PRO 实验与回放（getscore）

本项目用 RPent 连接高层语言规划器、LIBERO-PRO 仿真和远程动作模型，比较纯策略、GPT-6 Astra、Kimi 3.1pro 搭配 π0.5 / GAWM 的表现，并提供实验回放页面。**本机负责规划、仿真、动作执行与记录；`49.233.66.228` 提供 GPU 模型推理。** 当前成果是已有模型的接入、评测和回放，没有在本项目中重新训练这些模型，也不是 RoboTwin / RoboCasa 的正式榜单成绩。

本文状态截至 **2026-10-29（北京时间）**。新复现使用本文的 `*-228.toml` 配置；其他文档中的 `192.168.50.73` 是已经收回的旧服务器，不能继续作为启动地址。

## 已有成果与结果范围

- π0.5、GAWM、SAM3 已迁移至 `49.233.66.228`。真实仿真、策略推理、分割及视频记录链路均有验证记录。
- 早期单任务中，GPT-6 Astra/low 搭配两种动作模型均完成过 `libero_object_swap / task 2 / seed 0` 的沙拉酱入篮任务。
- 当前整理目录 `outputs/long120_1027/` 有 120 个真实实验目录，环境成功标志记录为 63 项成功。每组各 10 个任务：

| 规划方式 | 动作模型 | Long original 成功数 | Long swap 成功数 |
|---|---|---:|---:|
| 纯策略（none） | π0.5 | 10/10 | 1/10 |
| 纯策略（none） | GAWM | 10/10 | 1/10 |
| GPT-6 Astra / low | π0.5 | 5/10 | 4/10 |
| GPT-6 Astra / low | GAWM | 6/10 | 2/10 |
| Kimi 3.1pro / high | π0.5 | 7/10 | 5/10 |
| Kimi 3.1pro / high | GAWM | 7/10 | 5/10 |

**这张表是当前保留结果的汇总，不能解释为统一预算的严格配对评测。** 94 项沿用原预算，另外选定的 13 项 original 和 13 项 swap 使用 2000 动作总预算；original 部分包含续跑。swap 的 13 项是只对原失败项从头重跑，新增 3 项成功，三项均在 1000 动作以内完成，不能归因于增加步数。8 项 swap 重跑的初始 RGB 与旧记录不完全一致：任务、seed、初态文件及机器人初态匹配，但固定家具布局存在未被初态文件序列化的随机性。迁移前后推理依赖版本也有差异。

可随仓库查看的原始核验 JSON 和汇总快照见 [reports/reproduction-20261029.json](reports/reproduction-20261029.json)，其中记录来源路径及文件 SHA-256。完整视频、轨迹和对话记录在原开发机的 `outputs/`，没有上传到 Git。fake 短流程只证明链路可用，结束为 `stuck` 或 `native_success=false` 属于预期，不能算任务成功。

## 1. 获取代码与本地环境

推荐 Linux x86_64、Python 3.11、支持 EGL 的 NVIDIA 驱动。本项目在 RTX 3060 Laptop 6GB 上进行本地渲染，推理模型放在服务器；本地建议预留 30–50 GiB，长实验录像另计。需要 Git、`uv`、SSH，以及下载 GitHub / Hugging Face 公开源码和资产的网络条件。

```bash
git clone --branch main --recurse-submodules \
  http://192.168.50.20/wangyihan/rpent.git getscore
cd getscore
./scripts/setup-liberopro-reproduction.sh
```

准备脚本会按固定提交获取 RLinf、LIBERO、LIBERO-PRO、OpenPI client，建立 `.venv`，按 [依赖快照](configs/liberopro-reproduction.requirements.txt) 安装本地环境，下载固定版本的仿真资产并生成当前机器的路径配置。不会安装远程模型服务或下载模型权重。RPent 本身固定为子模块提交 `e2eac0c29f3da273a96e96c1cd14c54b34b66a58`；其代码同步在本仓库的 `rpent-runtime` 分支，子模块可从内网获取。其他源码仍需访问 GitHub，资产来自 `RLinf/LIBERO-PRO-assets` 的 `e4b10c9adc5735c9c8776917527178e693ef10b3`。

本地关键版本为 Torch 2.7.1、NumPy 1.26.4、MuJoCo 3.3.0、robosuite 1.5.2。不要用最新依赖替代快照，也不要把 MuJoCo 升级到 3.4+ 后仍假定实验条件相同。脚本会拒绝不同提交的已有源码及不匹配的已有资产链接/配置，需先人工检查；不负责覆盖旧环境。既有环境原始清单见 [configs/liberopro-environment.freeze.txt](configs/liberopro-environment.freeze.txt)，其中的旧绝对路径不能直接用于另一台机器。

若无法联网下载，可从原开发机复制以下目录，再在新机器重新安装 editable 包、重建资产链接及 `data/libero-pro/config/config.yaml`，不要直接搬运含旧绝对路径的 `.venv`：

```text
data/rlinf-runtime/       data/libero-runtime/
data/liberopro-runtime/   data/openpi-client/
data/libero-pro/assets/
```

## 2. 49.233.66.228 具体提供哪些辅助

服务器部署目录为 `/data/wangyihan/getscore-inference-20261028/`。以下三个服务只监听服务器的 `127.0.0.1`，必须经 SSH 转发访问，不能直接请求公网 IP 的这些端口。

| 功能 | 服务器/本机转发端口 | 输入与输出 | 何时需要 |
|---|---|---|---|
| 官方 π0.5-LIBERO 动作推理 | `18008`，WebSocket | 相机图像、机器人状态、文字指令 → 动作序列；本地执行前 5 步 | 使用 π0.5 策略时 |
| GAWM 动作推理 | `18115`，WebSocket | 主/腕相机图像、状态、文字指令 → 动作序列；本地执行前 8 步 | 使用 GAWM 策略时 |
| SAM3 文本/点选分割 | `18114`，HTTP RPC | 图像与文本/提示点 → 分割结果，供本地结合深度和相机标定定位 | 调用分割/定位工具时；GAWM fake 短流程也会调用 |

仿真渲染、任务成功判断、RPent 工具调度、语言模型 CLI、实验录像与回放都在本机进行。服务器没有替代语言规划器，也不提供实验用的 GPT/Kimi 账号。纯策略 rollout 及 `smoke_inference_228` 的策略短检查不调用 SAM3；回放已有记录不依赖该服务器。

π0.5 使用官方 `pi05_libero` checkpoint，16 个对象共 12,439,085,481 字节，迁移时逐文件核对 CRC32C 及可用的 MD5。GAWM 使用已有训练产物 `gawm_libero_80000_20261018_002850.pt`，SHA-256 为 `f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13`，还依赖配套推理源码、DINOv3 权重、VTT 向量及归一化统计。SAM3 权重 SHA-256 为 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。这些文件不在 Git 中。**只靠本仓库不能从零重建 GAWM 权重和完整远端环境**；优先复用已部署服务，独立部署需向维护者获取对应模型和配套文件，SAM3 自行下载还需要获准访问其官方权重。

远端当前环境：OpenPI 为 Python 3.11 / Torch 2.7.1+cu128 / JAX 0.5.3；GAWM 与 SAM3 为 Python 3.10 / Torch 2.8.0+cu128 / Transformers 4.57.0 / NumPy 1.26.4。环境原由已有环境只读复制后隔离建立，不能把它描述成一个已经验证过的公开一键安装方案。完整来源和迁移记录见 [INFERENCE_SERVER_228.md](INFERENCE_SERVER_228.md)。

先向维护者取得有权访问部署目录和转发端口的 SSH 账号，并配置自己的 SSH 密钥。转发脚本默认使用 `wangyihan@49.233.66.228`，使用其他获授权账号时指定 `INFERENCE_SSH_TARGET`：

```bash
INFERENCE_SSH_TARGET='你的账号@49.233.66.228' ./scripts/connect-inference-228.sh
```

保持此终端开启，在另一个终端运行实验。脚本转发本机 `18008`、`18114`、`18115`；若端口已占用，先确认是否已有正确隧道。若模型服务未运行，请维护者按 [迁移说明](INFERENCE_SERVER_228.md#服务入口) 启动对应服务，不要重复启动或终止其他人的进程。

## 3. 先运行无需语言模型账号的短流程

在仓库根目录、SSH 隧道已建立的条件下执行：

```bash
# 脚本规划器 + 真实 π0.5 + 本机仿真，2 次推理约执行 10 个动作
./scripts/run-liberopro.sh --config configs/liberopro-228.toml --chunks 2

# 脚本规划器 + 真实 GAWM + SAM3 分割 + 本机仿真，约 17 个动作
./scripts/run-liberopro.sh --config configs/liberopro-gawm-228.toml --chunks 2
```

这里的 fake 仅指规划器按固定顺序调用工具，不是伪造模型动作。不会调用付费语言模型；π0.5 / GAWM / SAM3 均使用真实远程推理。首个请求可能有加载或编译等待。每次结果写入新的 `outputs/liberopro-.../`；检查 `configuration.json`、`console.log`、`network-timing.jsonl`、`states.json` 和 `episode.mp4`。成功与否以环境原生标志为准，不以规划器自述为准。

若只排查动作推理和渲染，不使用 RPent 规划及 SAM3：

```bash
env -u PYTHONPATH .venv/bin/python -m scripts.smoke_inference_228 \
  --policy pi05 --max-steps 16 --output outputs/smoke-pi05-228
env -u PYTHONPATH .venv/bin/python -m scripts.smoke_inference_228 \
  --policy gawm --max-steps 16 --output outputs/smoke-gawm-228
```

`--output` 须为尚不存在的目录；重复运行请更换名称。检查生成的 `verification.json` 是否为 `passed`，动作数与视频帧数是否一致。这两项检查使用 Long swap task 0，而上面的 RPent 短流程使用 object swap task 2。

## 4. 使用真实规划器与批量实验

真实 GPT 规划入口要求本机具有可用的 Codex / ChatGPT 登录和相应模型权限，登录信息不随仓库提供。两份 228 配置已经包含规划参数：

```bash
./scripts/run-liberopro.sh --planner codex --config configs/liberopro-228.toml
./scripts/run-liberopro.sh --planner codex --config configs/liberopro-gawm-228.toml
```

默认模型为 `gpt-6-astra` / low；Kimi 批量实验使用本机 Kimi CLI 登录及 `kimi-code/k3.1-pro` / high。账号、额度和模型访问条件需要各自准备，不能用 fake 流程替代真实规划结果。

先复现一个纯策略 Long 案例（`--pilot` 是 original task 0 的小规模运行，不是完整 120 项评测）：

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_long_benchmark.py \
  --output outputs/reproduce-long-pilot --pilot \
  --only-policy pi05 --only-planner none --max-steps 1000 \
  --inference-config configs/liberopro-228.toml
```

完整批次、选择性续跑及 13 项 swap 重跑入口分别见 `scripts/run_long_benchmark.py`、`scripts/run_selected_long_2000.py`、`scripts/run_swap_kimi_2000.py`。选择性重跑依赖原批次的结果、选择清单和轨迹，不能只克隆仓库就执行。历史输出目录中已有有效结果，不应为了尝试命令而覆盖或重复运行。基准脚本不传 `--inference-config` 时仍会使用旧服务器默认地址，务必明确传入 228 配置。

## 5. 回放与离线检查

```bash
./scripts/run-viewer.sh --default-path outputs
```

浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，输入新生成的实验目录即可查看工具调用、前后画面、公开对话、耗时、token 和视频。需要查看历史 120 项时，另行从原开发机复制 `outputs/long120_1027/`；其当前大小约数十 GiB，不包含在仓库或 228 推理服务里。回放器本身仅依赖 Python 标准库，也可以不安装仿真环境而运行 `python3 -m run_viewer.server --default-path /你的实验目录`。详见 [回放说明](run_viewer/README.md)。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest tests -q
```

本次提交在现有环境完成离线测试：79 passed、1 skipped；两种策略按本文命令重新跑通真实短流程，视频分别为 10 帧和 17 帧，付费语言模型调用为零。依赖快照的安装预检查为 223 个包无需变更，兼容性检查通过。环境准备脚本已做语法检查，尚未在全新机器完成下载重装。离线测试不重新验证任务成功率；新机器应以上述真实短流程作为验收。

## 文件与补充资料

| 路径 | 内容 |
|---|---|
| `RPent/` | 固定版本的框架子模块 |
| `scripts/`、`configs/` | 远程模型桥接、实验入口、结果整理与环境快照 |
| `run_viewer/` | 只读本地实验回放页面 |
| `reports/` | 随 Git 保存的轻量验证快照 |
| `data/`、`.venv/` | 本地源码、仿真资产、Python 环境；不纳入 Git |
| `outputs/`、`output/` | 原始实验、迁移核验、视频与轨迹；不纳入 Git |

- [推理服务迁移与部署细节](INFERENCE_SERVER_228.md)
- [π0.5 接入历史记录](LOCAL_LIBEROPRO.md)、[GAWM 接入历史记录](LOCAL_GAWM_LIBERO.md)：接口与早期结果可参考，旧地址不适用于新部署。
- [路线记录](ROADMAP.md)、[规则](STEP1_RULES.md)、[方案调研](STEP2_SCHEMES.md)、[基线准备](STEP3_BASELINE.md)
- [论文与代码索引](paper/README.md)、[Kimi 机器人任务调查](KIMI_ROBOTICS_SURVEY.md)

`data/gpt-policy-robodojo-contexts/` 是历史阅读用专家示范，当前程序未引用，不能替代仿真资产、模型权重或 RPent memory。
