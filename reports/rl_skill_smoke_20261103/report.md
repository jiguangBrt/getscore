# π0.5 三技能 RL 最小流程记录

日期：2026-11-03（北京时间）。最终服务器目录：`/data/wangyihan/getscore-rl-skills/outputs/smoke_20261103_v2`。

## 完成范围

使用真实官方 `pi05_libero` checkpoint，完成自定义 LIBERO-PRO 技能环境 → Flow-SDE 训练采样 → 奖励/终止与 GAE → 六个微批次反向传播 → 一次 AdamW 更新 → 检查点权重加载 → GPU 0 到 GPU 1 权重同步。

没有 benchmark 或独立评测，没有成功率实验，没有大规模训练。视频全部属于更新前 v0 的训练采样，不反映本次更新带来的能力变化。

使用 RLinf 的真实 `Pi0RL` 模型、Flow-SDE 采样器、value head 及 PPO 损失，外层为本项目最小同步驱动。本轮没有验证完整 Ray/FSDP 分布式 runner，也没有接回语言规划器进行端到端测试。

## 三类训练任务与视频

全部使用 `pick up the white mug`，由不同初态和奖励区分任务，不强制抓杯柄。放置初态通过脚本真实接触、闭合夹爪并提升建立，未焊接或瞬移夹持；这些初始化动作不进入 on-policy 轨迹。

| 训练任务 | 本条轨迹动作数 | 策略 chunk 数 | 训练采样中观察到的结果 |
| --- | ---: | ---: | --- |
| 抓取 | 72 | 15 | 杯子提起约 12 cm，夹持条件连续保持 10 步 |
| 放置 | 142 | 29 | 杯子受目标盘支撑，松手撤离后稳定 20 步；最终倾角约 1.47° |
| 扶正 | 200 | 40 | 超时截断，最终仍约 86° 侧翻，未达到扶正条件 |

合计 414 个原生动作、84 个 chunk。上述是三条训练轨迹的结束情况，不是成功率统计，也不能证明模型因 RL 更新学会了新技能。

- [训练视频总览](/home/jh/getscore/outputs/rl-skills-smoke-20261103/training-overview.mp4)
- [抓取](/home/jh/getscore/outputs/rl-skills-smoke-20261103/pick/training.mp4)、[放置](/home/jh/getscore/outputs/rl-skills-smoke-20261103/place/training.mp4)、[扶正](/home/jh/getscore/outputs/rl-skills-smoke-20261103/upright/training.mp4)

视频包含主/腕相机、技能、步数、奖励、倾角、夹持和成功标志。三段分别为 73、143、201 帧，各含一帧初态；总览 417 帧、20 fps，约 20.85 秒，已逐帧解码并查看关键帧。上述本机 `outputs/` 视频不随 Git 传递，服务器保留相同文件。

## 受限更新和核验

- 基础模型共 3,356,188,433 参数；本轮仅动作输出层 `action_out_proj` 和价值头可训练，共 2,787,361 参数。视觉/语言骨干及 action expert 主体冻结。
- 三技能各选 2 个真实采样 chunk；microbatch=1，累积 6 个，**optimizer.step 仅执行 1 次**。这只是诊断批量，不是收敛训练配置。
- 六个微批次更新前的策略概率重算最大误差均为 **0**；梯度有限，裁剪前范数约 12.922，裁剪阈值 1.0。
- 10 个可训练参数张量实际变化。动作输出层最大变化约 `1e-6`，价值头最大变化约 `1e-5`。
- 保存的检查点约 32 MiB，含可训练权重、优化器、配置和 RNG，基础权重独立保留。检查点权重重载精确一致。
- 更新后的 10 个张量从物理 GPU 0 同步到 GPU 1 后逐元素一致，版本从 v0 到 v1。未执行 v1 评测或追加训练。

证据：[采样记录](rollout-summary.json)、[更新记录](update.json)、[权重同步](weight-sync.json)、[最终核验](verification.json)、[模型加载](model-load.json)、[权重转换](checkpoint-conversion.json)。

检查点权重加载不等于完整断点续训已验证。当前物理状态检查覆盖序列化后位姿回读；控制器历史、技能阶段状态以及所有采样器/Python/NumPy RNG 的中途恢复仍需后续验证。

## GPU、代码与依赖

最终使用物理 GPU 0 做 learner、GPU 1 做 rollout、GPU 3 做 EGL 渲染。CUDA 白名单为这三张卡的精确 UUID；驱动查询显示物理 GPU 3 对应 EGL 编号 4，实际进程监控核实图形进程在物理 GPU 3。见[渲染绑定核验](render-contract.json)。

首轮曾误将 EGL 编号 3 当成物理 GPU 3，实际短暂渲染在物理 GPU 4。该轮已作废、保留 `SUPERSEDED.md`，未用于优化器更新；最终三条训练轨迹重新绑定后重跑为 v2。

所有新代码、独立环境、数据、缓存和权重均在服务器 `/data/wangyihan/getscore-rl-skills`，不引用其他用户目录或虚拟环境。服务器直连 GitHub 不通，公开源码从用户本机已有快照独立传入，见[来源版本](source-provenance.json)。EGL/OpenGL 加载库通过公开 Ubuntu 包解压到自己的目录，没有修改系统库。原始官方权重没有覆盖。

关键版本为 Torch 2.8.0+cu128、Transformers 4.53.2、JAX 0.5.3、MuJoCo 3.3.0、robosuite 1.5.2、NumPy 1.26.4；[完整环境](requirements.freeze.txt)。安装后的依赖一致性检查通过，共 199 个包。旧 π0.5、GAWM、SAM3 服务已停止，未改动其他用户进程。

服务器代码入口：`code/skill_env.py`、`code/runtime.py`、`code/smoke_driver.py`；实际设置：`configs/smoke.json`。最小驱动硬限制一次更新，不配置自动继续。主要文件的 SHA-256 在最终核验中保留。

## 正式训练前的决策

全部临时设置与推荐值见[完整决策清单](DECISIONS.md)，可核对[本次实际配置](effective-config.json)。优先讨论：

1. 是否用不同指令/技能条件区分抓取、放置、扶正，还是坚持同一句并使用独立适配器。
2. 物体范围、初态、倾斜课程，以及放置/扶正的目标区域。
3. 奖励塑形、抓取保持、松手、直立和稳定阈值。
4. 是否需要示范热身；正式更新整个 action expert、LoRA 还是仅输出层。
5. 探索噪声、去噪/执行长度、学习率、PPO/GAE/critic、batch、任务比例和训练预算。
6. 保存、录像、停止与恢复策略；本轮没有授权或执行评测，之后何时允许评测另定。

本次建议不自动成为正式训练设置，也不会因为该单步流程通过而启动长训练。
