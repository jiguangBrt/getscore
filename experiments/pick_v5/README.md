# 随机单物体抓取 v5

基座为 pi05_libero，继承 v4 已验证的白杯 LoRA。每次只生成一个目标，均匀采样 12 种资产，随机 XY/yaw。只训练抓起并保持 2 秒；仍保留 200 个原生控制步（10 秒）任务时限。

资产：白杯 `porcelain_mug`、罐头 `alphabet_soup`、番茄酱瓶 `ketchup` / 罐 `tomato_sauce`、黄油盒 `butter`、cheese 盒 `cream_cheese`、黑碗 `akita_black_bowl`、白黄杯 `white_yellow_mug`、书 `black_book`、chocolate `chocolate_pudding`、摩卡壶 `moka_pot` / `yellow_moka_pot`。使用原始资产和原始物理属性。书使用该资产原有的立放姿态，本阶段没有覆盖所有摆放姿态或整张桌面的任意位置。

## 奖励与判定

沿用 v4 的有界接近/接触/抬升/稳定/保持进度与终局奖励；新增 `collision_monitor.py` 在全部 MuJoCo 物理子步积分非预期夹爪接触成本。目标 finger/pad 接触豁免，手掌撞目标和夹爪撞环境计成本。每个子步 `-.2 * dt * clip((F-.5)/(10-.5),0,1)`，不因碰撞直接终止。

稳定速度用相邻 20Hz 控制步实际物体位姿差计算，另存瞬时求解器速度；这是为避免薄盒接触约束的瞬时速度抖动造成永久判失败。成功仍需真实双侧接触、离桌≥5cm、连续2秒、速度≤1cm/s和0.1rad/s、高度跨度≤5mm。越界采用3cm内免费、3–10cm渐增软成本，20cm严重偏移或掉桌才硬终止。

## 训练流程

`multi_preflight.py`：CPU 正负/碰撞控制。`grasp_search.py`：辅助物理可行性检查，其成功不计入模型指标。

`collect_multi_demos.py`：每类少量真实执行的成功示范。`demo_warmup.py`：均衡抽取物体与提示词，监督 flow matching 初始化；明确与 PPO 分开统计。

`minibatch_ppo.py`：8个独立 MuJoCo 进程，批量策略推理，64 chunks/minibatch、2 epochs、精确重算与全批 KL guard；环境跨 rollout 边界保留，评估使用独立环境池。重置并行，重置脚本中暂时关闭相机、最终恢复；`reset_render_probe.py` 验证物理状态、图像与奖励完全一致。

`--initial` 为显式权重迁移、优化器重置；`--resume` 必须保持奖励/任务/场景契约一致，并记录丢弃的未完成环境。

正式 PPO 入口依赖新版本物理和优化器 smoke 验证。每个检查点固定验证每类4条，另行保留独立种子做最终验收。`best.pt` 以总体成功率选优；`best-coverage.pt` 优先覆盖有成功的物体类别。不能仅凭均值 reward 为正宣称每类都已学会。

当前进度与最终结果见 `reports/rl_pick_v5_20261106/`。


## 本轮结论与入口

805个主试训完整episode有374次真实成功、578次正总回报；固定验证从初始化28/48到末轮26/48，没有稳定改善。保留H update9（27/48、12类均有成功），不把监督初始化贡献算作RL净收益。最终独立验收、阶段漏斗与限制见报告。

`VALIDATED.json` 固定权重来源与SHA256；`launch-pilot.sh` 默认恢复该检查点，使用 `configs/validated.json`，在服务器安装为 `/data/wangyihan/getscore-rl-skills/run-pick-v5.sh`。例如继续**有界试验**：

```bash
bash /data/wangyihan/getscore-rl-skills/run-pick-v5.sh \
  --out /data/wangyihan/getscore-rl-skills/runs/pick-v5-next-pilot \
  --episodes 100 --max-updates 15 --hours 1
```

此入口已核对奖励/任务/场景恢复契约和权重哈希，没有自动启动。运行前检查GPU6仍空闲。技术验证通过不表示已证明长训会提升，当前不建议直接无人值守长训。

实验配置 `hold-potential.json`、`object-balanced.json` 的对照没有明确获益，默认保持原v5.1奖励和全批优势归一化。冻结前缀缓存已做配对数值等价验证并在61轮连续更新中使用；不冻结可训练动作专家/critic。

BDDL文件的language文本继承了白杯模板，仅用于场景元数据。实际策略指令由 `PickEnv.prompt` 按目标类型覆盖；观测和轨迹保存了正确的逐物体指令。恢复训练核对场景哈希，因此不为修改这段元数据改变既有场景文件。
