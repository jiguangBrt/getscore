# 2026-11-04 夜间续训至 11-05 09:00

用户授权：继续当前 checkpoint 和奖励函数强化训练至北京时间 2026-11-05 09:00；每3小时保存、及时Git、W&B监控。水平惩罚允许加大，本阶段选择保持不变，保留可比性。未改模型、物理、PPO、batch8数值分组或GPU分配。

- 服务器：`wangyihan@49.233.66.228`
- 项目：`/data/wangyihan/getscore-rl-skills`
- 专用代码：`experiments/overnight-v3-20261104/`
- 本run：`/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-overnight-20261104-to-20261105-0900`
- 父checkpoint：`runs/pi05-skills-v3-6h-20261104-115938-four-gpu/checkpoints/final.pt`
- 导入更新257、原生步1,372,917、224份optimizer状态；权重/optimizer/主进程及三个actor RNG逐项相同。父阶段结束的活动仿真不跨进程恢复；本阶段重新初始化环境，保留episode ID和课程进度。
- GPU：物理0中央learner，1/3/2各8环境，合计24；四卡各192个chunk计算梯度，全局768个chunk执行一次中央optimizer更新。GPU2与liuyang评测共享，不能停止或修改同事任务。
- 19:23建立本阶段时钟，19:25启动服务器监督器；正式时间见run-meta.json。
- 每3小时checkpoint和固定评测，预计22:23、01:23、04:23、07:23附近（在完整更新边界执行）；latest每5轮。每10轮训练视频。
- 08:50结束训练循环，预留约10分钟做最终评测、checkpoint和汇总；09:00服务器硬截止，只处理本run独立进程组。若最终评测超时，明确报告最后完整checkpoint与未完成评测，不能冒称完成。
- 固定评测每技能24次尝试；恢复后baseline为抓取0/24、放置0/21（另3次初始化失败）、扶正0/24。

服务器监督器初始PID `1809464`，训练PID `1809466`，三个actor PID `1811871/1811872/1811873`。PID可能随最多2次受限恢复而变化，以supervisor-pid.json、supervisor-status.json和events.jsonl为准。配置或数值错误不自动绕过。

## W&B 和结果

服务器直连api.wandb.ai返回403，因此服务器持续写完整W&B offline日志。本机使用用户既有W&B身份，将只读SSH取得的训练/评测指标实时中继到云端；凭据不复制到服务器、不写代码或Git。云端链接保存在本目录 `overnight/wandb-cloud.json` 和服务器run的同名文件。服务器训练、保存、评测和汇总均独立于本机、SSH及W&B连接。

本机中继脚本：`/home/jh/getscore/scripts/monitor_overnight_v3.py`。隔离SDK环境：`/home/jh/.cache/getscore-wandb-monitor/.venv`（wandb0.31.0）。中继持有只在该任务运行期间有效的sleep/idle inhibitor，不改永久电源配置；本机断电/关机仍会暂停云端中继，但服务器保留完整日志。

服务器每30秒更新RESULTS.md/RESULTS.json/GPU快照，每5分钟生成learning-curves.png；每3小时checkpoint阶段及最终阶段生成报告并对该报告执行独立Git提交。最终文件在本run的checkpoints/final.pt、evaluation-final.json、RESULTS.md、learning-curves.png和videos/。

本机副本在 `/home/jh/getscore/reports/rl_skills_v3_20261104/overnight/`。已创建本聊天每30分钟检查，以及11月5日09:00专门结果汇报。常规检查无异常时保持安静；完成后暂停两项自动检查。

## 故障处理与继续检查

1. 先只读检查本run的supervisor-status.json、status.json、events.jsonl和trainer-attempt*.log，不使用ray stop、pkill或任何跨任务操作。
2. 监督器最多启动3次，attempt-count.json累计计数，不因手动重启而重置。若训练仍活着，不重复启动；若数值/配置错误，保留现场并报告。
3. 只有监督器及本run训练均已退出、仍早于08:50、累计启动次数未到3时，才可执行以下专用入口。入口不依赖W&B凭据：

```bash
ssh -o BatchMode=yes 49.233.66.228 \
  '/data/wangyihan/getscore-rl-skills/.venv/bin/python /data/wangyihan/getscore-rl-skills/experiments/overnight-v3-20261104/launch_secure.py'
```

4. 核对checkpoint的config、run_meta、更新号、optimizer与RNG来源；不得将fresh-start当作恢复，不改训练截止或奖励。
5. 09:00汇报新增与累计更新/原生步，固定评测成功数及有效/尝试分母、初始化失败、checkpoint、曲线、代表视频、W&B链接和Git。零成功或无提升必须直接写明。

本阶段沿用的环境/奖励/模型/PPO源码与已验收父阶段逐文件SHA256一致。续训入口只修改来源验证、时钟、监控和报告；导入校验见服务器 `launch-validation.json`，本机镜像见 `overnight/launch-validation.json`。首轮真实更新校验会另存verification文件。

## 启动验收

首轮真实更新258通过：新增6,097原生步、累计1,379,014；四卡各192个chunk；actor LR=1e-6；没有重新warmup；logprob重算误差0；梯度范数1.39465；更新后KL约0.000129。同步权重相对导入checkpoint确实变化。完整证据见 `overnight/continuation-verification.json`。

硬截止和恢复次数通过独立CPU子进程测试：只终止临时测试进程组，截止后不重试，监督器重启不能重置最多3次启动的累计预算。测试没有触及训练或其他用户进程，证据见 `overnight/supervisor-validation.json`。

W&B云端：[夜间训练监控](https://forge.coreweave.com/wandb/jiguangbrt-university-of-dundee/rpent-pi05-skills/runs/overnight20261104u257)。已通过云端API读取确认run处于running并收到基线和训练指标。

服务器代码/配置初次提交：`1363a62`。后续验收提交和每3小时/最终报告提交可通过服务器 `git log` 检查。本聊天自动检查ID：`rl`；09:00汇报ID：`automation`。
