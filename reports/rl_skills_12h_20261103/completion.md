# 12 小时试训完成记录

2026-11-04 09:42（北京时间）检查。任务于 06:40:30 正常结束，实际 12.054 小时；supervisor attempt0、exit0，无异常重启。862 次更新、3,520,609 个原生动作，物理 GPU 0、1、3 均已空闲，无训练进程残留。

## 固定初态评测

| 时点 | 更新 | 抓取 | 放置 | 扶正 |
|---|---:|---:|---:|---:|
| baseline | 0 | 14/24 | 1/16 | 0/23 |
| 3 小时 | 225 | 14/24 | 0/16 | 0/23 |
| 6 小时 | 435 | 15/24 | 0/16 | 0/23 |
| 9 小时 | 649 | 14/24 | 0/16 | 0/23 |
| 12 小时 / final | 862 | 16/24 | 0/16 | 0/23 |

每次每技能尝试 24 个固定案例。放置 8 个案例的真实夹持初始化失败，扶正 1 个案例的倾斜初始化无效，均另报且未计作策略成功。有效分母跨评测保持一致。最终抓取新增成功 case pick-01、pick-11，无原成功案例丢失；放置丢失 baseline 成功 case place-06。

结论：训练流程按预算完整运行，但整体技能提升目标未达到。抓取多完成 2 个固定案例，放置未改善，扶正没有学会。单种子小样本结果不足以证明抓取泛化提升，也不能解释为原 LIBERO 基准或 RPent 全流程成绩。

## 训练核查与限制

- 全部 862 轮旧概率重算最大误差为 0；记录的 post-update approximate KL 最大 0.002394，未触发 0.05 阈值。
- 共记录 2,775 次初始化重试错误， broad-object 的真实夹持初态覆盖仍需改进。
- 训练日志中的 completed 只包含采样期间真正结束的轨迹；每轮 32×8=256 步后仍未结束的环境会 bootstrap 并在下一轮重新初始化，短于单环境 400 步上限。因此不能把 completed 成功比例当成全量训练成功率，尤其不能用它替代固定案例评测。
- 32 次脚本扶正示范尝试均未满足完整成功条件，没有进行 BC 热身。扶正缺少成功探索样本，仍是后续设计要解决的问题。
- 不启动新训练。下一步应先检查奖励、成功保持条件、物理初始化和采样截断对技能完成的影响，建立有效的扶正成功数据，再决定是否继续训练。

## 产物

服务器运行目录：`/data/wangyihan/getscore-rl-skills/runs/pi05-skills-12h-20261103/`。

`checkpoints/` 已保存 initial、latest、best、final，以及 3/6/9/12 小时里程碑。final 为约 111.1 MiB 的 LoRA + critic + 优化器/RNG checkpoint，基础权重另存。

W&B offline 原始文件为 `wandb/offline-run-20261103_183741-2epwkm4a/run-2epwkm4a.wandb`，未上传云端。共保存 204 段训练/评测视频。

本机已同步[最终汇总](/home/jh/getscore/outputs/rl-skills-long-20261103/live/RESULTS.md)、[曲线](/home/jh/getscore/outputs/rl-skills-long-20261103/live/learning-curves.png)、全部评测 JSON 和视频。示例：[最终抓取](/home/jh/getscore/outputs/rl-skills-long-20261103/live/videos/eval-12.0h-actor0-pick.mp4)、[最终放置](/home/jh/getscore/outputs/rl-skills-long-20261103/live/videos/eval-12.0h-actor0-place.mp4)、[最终扶正](/home/jh/getscore/outputs/rl-skills-long-20261103/live/videos/eval-12.0h-actor0-upright.mp4)。
