# π0.5 v3 六小时训练启动记录

正式任务于北京时间 2026-11-04 11:59:38 启动，预计约 17:59:40 结束训练，随后完成最终评测、曲线和结果导出。已安排安静的完成回访，正常运行期间不作例行通知。

服务器：`wangyihan@49.233.66.228`。项目：`/data/wangyihan/getscore-rl-skills`。

输出目录：`/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938`。

监督 PID：`3643294`，已成为独立 session、父进程为 1。训练 PID：`3643297`；actor PID：`3645695`、`3645696`。

实际执行入口：

```bash
cd /data/wangyihan/getscore-rl-skills
./launch-v3-6h-20261104.sh
```

启动器以 `nohup setsid` 执行的有效命令为：

```bash
RL_SKILLS_CONFIG=/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938/effective-config.json \
/data/wangyihan/getscore-rl-skills/run-long.sh v3_supervise.py \
  --out /data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938 \
  --hours 6 --envs 16 --chunks 32
```

以上记录已执行的命令。配置、预检证据、源码快照和 hash 已固定在正式目录中；运行不再依赖之后可变的 current 链接。

## 配置与实现

仅使用物理 GPU 0、1、3，通过原有 CUDA UUID 白名单与 EGL 枚举绑定；0 为中央 learner，1/3 各 8 环境并参与全局梯度合并。每次更新 512 个执行块，每块最多 8 个原生动作。

从指定 pi05-libero base 和全新 rank16/alpha16 LoRA、critic 开始，没有加载上一轮 final/latest。基础权重 SHA256 已核验。所有依赖、源码、缓存、检查点与视频均使用自己的 `/data/wangyihan` 路径。W&B 为 offline。

200 原生步含保持/稳定时间；块内终止、timeout 最终观测 bootstrap、terminal 零 bootstrap、跨采样批次与参数更新保留活动 episode、独立评测环境和评测 RNG 恢复均已实现。奖励使用单次事件与单调进展额度，水平路程不退款，pick 的下降抖动额度整条 episode 共用。原 batch8 顺序保留，三卡全局加权梯度只执行一次中央 optimizer step。

每 2 小时 checkpoint 与固定种子评测，每技能 24 次尝试；latest 每 5 次更新，训练视频每 10 次更新，保留评测双视角视频及原生 trace。最多两次受限恢复，严重数值/配置错误停止；恢复会明确丢弃未完成模拟轨迹，不声称逐位续跑。

## 验收证据

- 两类杯子和摩卡壶各两个原始场景：18/18 有效物理初态。脚本抓取 3/6、放置 6/6、扶正 0/6；脚本轨迹没有进入 PPO 或 BC。六个放置初态真实夹持，碰撞几何间隙为 10.023–10.128cm。
- 4 次小规模真实更新，原 batch8 logprob 重算最大误差为 0；108 个 actor 梯度张量非零。三卡梯度与串行参考最大差为 4.77e-7。actor/critic 均发生权重变化，224 个 optimizer 状态与 RNG 保存恢复通过。
- 检查了 48 次跨批次延续、插入评测前后的模拟器状态 hash/计时/奖励账本一致，以及 16 条恰在第 200 原生步超时的轨迹、块内提前结束 mask 和 2/4/6 小时里程碑触发。
- 真实控制对照：受控局部放置回报 1.34559，空中松手 1.15714，搬远 -1.04645，倾倒 -0.80467。补充的真实“接触后再释放”对照回报 1.62732，成功触发接触、释放和撤离事件。
- 相同 RGB、机器人状态和模拟状态，只改变指令：base 第一块动作最大差 0.02151。只能证明该对照中存在指令敏感性，不能证明正确语言绑定。
- Inventory 保留 40 个物体/场景变体、18 类物体、10 个场景；白杯和黄白杯各跨 3 个场景，摩卡壶跨 2 个场景。训练种子低于 80000000，固定评测使用独立的 90000000 段种子。

## 正式 baseline 与限制

正式未更新 base：抓取 0/24；放置 0/21，另 3 次初始化失败；扶正 0/24。预检脚本成功次数不是策略成功率。

扶正预检尚无成功恢复样例，没有使用失败脚本作示范。Place 的终态成功判定仍可能包含空中松手后恰好落稳；这种动作的总回报更低，且没有受控接触/释放/撤离事件奖励，终态成功率不能代替动作质量。进程重启时只能按 checkpoint 活动轨迹快照记录丢弃，无法精确恢复崩溃中尚未写出的步。

首次物理检查发现原场景桌面比 workspace_offset 高约 2.68cm，已改为真实碰撞支撑面并重跑。塑形校准新增的推荐值为：受控下降速度上限 0.15m/s、下降进展姿态容差 15°、扶正受控抬升事件 2cm；最终稳定成功始终保持 ≤5°。这些是实现预检标定值，不冒充用户逐项指定。

阶段提交：`d783909`（运行时）、`82eea78`（物理/数值验收）、`c440fb6`（固定配置与后台启动器）。

详细证据见本目录 `v3-preflight-acceptance-20261104.json`、`PREFLIGHT_V3_20261104.md` 及 `preflight/` 视频和 trace。服务器最终输出将包括 `RESULTS.md`、`RESULTS.json`、`learning-curves.png`、checkpoint 清单、逐例评测和视频。

## 首个正式更新核验

已完成 update 1：512 个执行块、4084 个原生步，三卡分别处理 176/168/168 个 chunk，logprob 重算误差 0，非零梯度。前 3 次为配置要求的 critic warmup，actor LR=0；actor 非零更新已在单独的第 4 次预检中核验。正式 current-policy 的 critic 权重已相对 fresh-start 变化。

fresh-start 的 update/native_steps 均为 0，所有 LoRA B 为 0、optimizer 为空；确认没有加载上一轮权重。UUID 白名单、独立 session/PID、W&B offline 数据文件、4 个初始 checkpoint、2 个训练视频和 6 个 baseline 视频均已实测存在。详情见同目录 launch-verification JSON。

完成回访已创建（ID：0-5-v3），18:10 前保持安静，之后完成时汇报结果并停用回访；正常运行不例行通知。
