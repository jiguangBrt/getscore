# π0.5 三技能训练结果

更新时间：2026-11-05 00:50:54 CST
状态：training
完成更新：473；原生动作步数：2688224

受控续训：物理 GPU 0、1、2、3，共享同一 action-expert LoRA + critic；24环境；夜间续训硬截止为北京时间2026-11-05 09:00，08:50起预留最终评测。
W&B 模式：offline；云端：https://forge.coreweave.com/wandb/jiguangbrt-university-of-dundee/rpent-pi05-skills/runs/overnight20261104u257；本地日志保留在 wandb/。

## 评测

| 时点 | 更新 | 抓取 | 放置 | 扶正 | 有效/尝试 |
|---|---:|---:|---:|---:|---:|
| baseline | 257 | 0.0% | 0.0% | 0.0% | 69/72 |
| 3.0h | 382 | 0.0% | 0.0% | 0.0% | 69/72 |

成功率分母为有效物理初态；初始化失败另列，不计作策略成功。评测是自定义技能环境，不能当作 LIBERO 原基准或 RPent 全流程成绩。

## Checkpoints

- [best.pt](checkpoints/best.pt) (111.1 MiB)
- [continuation-start.pt](checkpoints/continuation-start.pt) (111.2 MiB)
- [initial.pt](checkpoints/initial.pt) (111.1 MiB)
- [latest.pt](checkpoints/latest.pt) (111.2 MiB)
- [milestone-3.0h.pt](checkpoints/milestone-3.0h.pt) (111.2 MiB)

Checkpoint 包含 LoRA、value head、优化器、RNG、配置和基础模型哈希。推理时需与指定原始基础权重组合；不可直接当作完整基础模型。活动环境跨采样批次继续；进程重启时丢弃未完成环境轨迹并单独记录，不能逐位续跑。

## 视频

- [videos/train-v420-actor0.mp4](videos/train-v420-actor0.mp4)
- [videos/train-v420-actor1.mp4](videos/train-v420-actor1.mp4)
- [videos/train-v420-actor2.mp4](videos/train-v420-actor2.mp4)
- [videos/train-v430-actor0.mp4](videos/train-v430-actor0.mp4)
- [videos/train-v430-actor1.mp4](videos/train-v430-actor1.mp4)
- [videos/train-v430-actor2.mp4](videos/train-v430-actor2.mp4)
- [videos/train-v440-actor0.mp4](videos/train-v440-actor0.mp4)
- [videos/train-v440-actor1.mp4](videos/train-v440-actor1.mp4)
- [videos/train-v440-actor2.mp4](videos/train-v440-actor2.mp4)
- [videos/train-v450-actor0.mp4](videos/train-v450-actor0.mp4)
- [videos/train-v450-actor1.mp4](videos/train-v450-actor1.mp4)
- [videos/train-v450-actor2.mp4](videos/train-v450-actor2.mp4)
- [videos/train-v460-actor0.mp4](videos/train-v460-actor0.mp4)
- [videos/train-v460-actor1.mp4](videos/train-v460-actor1.mp4)
- [videos/train-v460-actor2.mp4](videos/train-v460-actor2.mp4)
- [videos/train-v470-actor0.mp4](videos/train-v470-actor0.mp4)
- [videos/train-v470-actor1.mp4](videos/train-v470-actor1.mp4)
- [videos/train-v470-actor2.mp4](videos/train-v470-actor2.mp4)

平均每轮耗时：88.84 秒。详细训练曲线见 metrics.jsonl 和 W&B；所有评测逐例结果均保留。

## 夜间续训来源

父checkpoint：/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-6h-20261104-115938-four-gpu/checkpoints/final.pt；导入更新257，原生步1372917。新增更新：216；新增原生步：1315307。权重、优化器及RNG逐项相同：True。上一阶段预算结束的活动环境不跨进程恢复。

## 完成轨迹与限制

自然完成轨迹: 8800；重启截断快照: 0；预算结束未完成: None；训练初始化失败: 698。
{'pick': {'success': 0, 'failure': 2902, 'timeout': 914}, 'place': {'success': 0, 'failure': 1, 'timeout': 3099}, 'upright': {'success': 0, 'failure': 531, 'timeout': 1353}}

本阶段从上一轮final checkpoint恢复LoRA/critic、优化器和全部RNG，奖励权重不变。扶正零成功时不宣称已学会；不使用未经验证的 BC 示范。固定评测报告有效初态和初始化失败，语言对照单独见预检。Place 的最终状态成功判定仍允许空中松手后恰好落稳；物理预检中这种动作总回报更低、没有受控接触/释放/撤离事件奖励，不能把终态成功率当作完整动作质量。重启截断记录来自 checkpoint 活动环境快照，崩溃中的未写出步不保证精确恢复。
