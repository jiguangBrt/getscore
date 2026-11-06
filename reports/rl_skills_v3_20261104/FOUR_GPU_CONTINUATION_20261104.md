# 同模型四卡续训记录

用户明确选择加入物理 GPU2，将同一个模型扩为24环境，接受一次受控重启；没有启动独立训练分支。

北京时间 2026-11-04 14:03:29，原任务在 **update95、385875原生步** 的完整保存边界切换。LoRA、critic、全部 optimizer 状态、旧 learner/actor RNG 与累计计数均保留。原始开始时间和 **17:59:40** 截止时间保持不变，不从重启重新计六小时。

- 监督 PID：`191182`；训练 PID：`191184`。
- 三个 actor PID：`193607`、`193608`、`193609`，分别使用物理 GPU1、3、2。
- Learner 保持物理 GPU0；CUDA UUID 顺序为物理 **0、1、3、2**，保留旧三卡 RNG 映射，末尾追加 GPU2。
- 每个 actor 仍为 batch8、32执行块；环境16→24，全局批次512→768。四卡全局加权汇总，仍然只有一次中央 optimizer step。
- 奖励、200原生步、LR、LoRA结构、跨批次延续和独立评测规则不变；保持 W&B offline、每2小时保存/评测和每10次更新记录训练视频。

新输出目录：

```text
/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938-four-gpu
```

实际后台命令：

```bash
RL_SKILLS_CONFIG=/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938-four-gpu/effective-config.json \
/data/wangyihan/getscore-rl-skills/experiments/four-gpu-v3-20261104/run.sh v3_supervise.py \
  --out /data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938-four-gpu \
  --hours 6 --envs 24 --chunks 32
```

命令中的 `--hours 6` 保留原任务参数；已存在的 run-meta 和入口检查强制沿用原始截止时间，不能延长预算。`configs/current.json` 现指向 `configs/skills-v3-four-gpu.json`，四卡必须使用上面的专用入口；原三卡代码和配置快照保留。

## 迁移与丢弃

父 checkpoint SHA256：`4599c3f2b3a7d74835ece08d88d329929768b1fde5976ea6edeb9feac67f9c4c`。

权重、optimizer、已有 actor RNG 和旧 CUDA RNG 前缀逐项比对一致。已有 optimizer 状态均为 step95，GPU2 新增确定性 seed42 CUDA RNG 和 actor-id2 初态采样流。正式配置与源码 hash 均固定。

未序列化模拟器中途状态：checkpoint 中的 **15个未完成episode** 明确丢弃，另有 **2810个未提交原生采样步** 被排除，未计入累计训练。所有已报告的完整更新均保留。父阶段目录保留，状态为 `resource_migrated`，不代表训练失败。

新目录保留父阶段 metrics、episodes、events、固定评测历史和初始/里程碑 checkpoint。迁移原始 checkpoint 与 `migration-start.pt` 另行保存。父阶段视频和 W&B 文件仍在原目录；四卡新视频与 offline W&B 写入新目录。

## 核验

四卡预检完成4次真实更新：原batch8 logprob重算误差0，四卡加权梯度与串行参考最大差4.77e-7；71次跨批次延续、3个评测前后状态比较、22条第200步超时轨迹均通过。物理环境、奖励、模型结构代码与父阶段逐字节一致。

首个真实续训更新 **96** 已完成：累计 **391984原生步**，本次新增 **6109步**；768个chunk由四卡各处理192个，logprob重算误差0。三个采样GPU实际为1、3、2，actor LR保持1e-6，未重新进入critic warmup。权重已继续变化。

阶段提交：`3d8dd31`（四卡运行时与数值验收）。详细证据见同目录 `MIGRATION.json`、`v3-four-gpu-preflight-20261104.json` 和 `v3-four-gpu-continuation-verification-20261104.json`。

完成回访 `0-5-v3` 已改为跟随四卡目录；18:10前保持安静，完成后统一交付两个资源阶段的累计结果及切换增量。自定义技能评测、扶正尚无成功样例和place终态成功不能代替完整动作质量等限制继续保留。
