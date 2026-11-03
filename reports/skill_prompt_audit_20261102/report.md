# 上一阶段动作工具任务描述审计

日期：2026-11-02。只读分析，未调用模型或启动新实验。

## 范围与核验

- 采用 2026-10-29 首次审计的 80 次语言模型实验，与此前 268 + 41 的统计严格对齐。包含续跑前缀，排除纯策略基线与后续替换轮次。
- 共 309 次外层调用、54 条不同的原始 prompt；不做大小写、同义词或标点归并。重复调用分别计数，不等于 309 个独立任务或示范。
- 309/309 次均与保留的原始 stream 中相同调用 ID、工具名及完整 arguments 核对；续跑归一化改写了部分 ID，核对使用保留的改写前记录。逐次来源及行号见 [calls.csv](calls.csv) 和 [calls.json](calls.json)。
- 当前后续统计是 78 项、269 次 pick + 41 次 doubled = 310 次；它与本报告的旧快照属于不同口径。
- 工具调用 status 表示接口是否报错，不能作为技能成功标签；run_success 是完整任务结果，也不能直接当成每一次抓取的成功标签。

| 后端 | pi0_pick | pi0_doubled | 合计 |
|---|---:|---:|---:|
| π0.5 | 138 | 21 | 159 |
| GAWM | 130 | 20 | 150 |
| 合计 | 268 | 41 | 309 |

## π0.5 / pi0_pick：全部 37 条原文

共 138 次。`pick up …` 86 次、`grasp …` 38 次，共 124 次；另外 14 次为开炉灶或完整任务句，不能全部标成抓取训练样本。

| 原始 prompt | 次数 |
|---|---:|
| `grasp the white mug` | 13 |
| `pick up the moka pot` | 12 |
| `grasp the yellow and white mug` | 12 |
| `pick up the chocolate pudding` | 12 |
| `pick up the alphabet soup can` | 9 |
| `pick up the cream cheese box` | 8 |
| `turn on the stove` | 8 |
| `pick up the book` | 7 |
| `pick up the tomato sauce can` | 6 |
| `pick up the black bowl` | 6 |
| `pick up the butter` | 4 |
| `pick up the yellow and white mug` | 4 |
| `pick up the right moka pot` | 4 |
| `grasp the yellow mug` | 3 |
| `pick up the moka pot on the table` | 3 |
| `put the yellow and white mug in the microwave and close it` | 3 |
| `grasp the yellow and white mug on the plate` | 2 |
| `grasp the silver moka pot by its body` | 2 |
| `turn on the stove and put the moka pot on it` | 2 |
| `grasp the black book in the caddy` | 1 |
| `pick up the book and place it in the back compartment of the caddy` | 1 |
| `grasp the white mug by its handle` | 1 |
| `pick up the alphabet soup` | 1 |
| `pick up the moka pot in front of the gripper` | 1 |
| `grasp the body of the right moka pot` | 1 |
| `pick up the right moka pot by its body` | 1 |
| `pick up the yellow and white mug from the table` | 1 |
| `grasp the yellow mug handle` | 1 |
| `pick up the silver moka pot by its body` | 1 |
| `pick up the black bowl from the bottom drawer` | 1 |
| `pick up the black book` | 1 |
| `pick up the white mug` | 1 |
| `grasp the moka pot by its body and lift it` | 1 |
| `put both moka pots on the stove` | 1 |
| `grasp the right moka pot by the body` | 1 |
| `pick up the moka pot by its body` | 1 |
| `pick up the moka pot by its handle` | 1 |

## π0.5 / pi0_doubled：全部 12 条原文

共 21 次，主要是抽屉、炉灶、微波炉和书本接触操作，不是通用放置接口。

| 原始 prompt | 次数 |
|---|---:|
| `close the bottom drawer of the cabinet` | 7 |
| `push the bottom drawer closed` | 3 |
| `push the book down into the back compartment of the caddy` | 2 |
| `push the bottom drawer fully closed` | 1 |
| `push the yellow and white mug into the microwave` | 1 |
| `turn on the stove` | 1 |
| `grasp the black stove knob and turn it clockwise to turn on the stove` | 1 |
| `open the microwave door` | 1 |
| `open the microwave` | 1 |
| `pull open the microwave door` | 1 |
| `grasp the microwave door handle and pull the door open` | 1 |
| `put the yellow and white mug in the microwave` | 1 |

## GAWM：全部 10 条原文

150/150 次 prompt 与该实验的固定完整训练任务句一致。两种工具共用这个词表；句子长是当前接入约束，不能由频次推断它更适合技能训练。

| 原始 prompt | pick | doubled | 合计 |
|---|---:|---:|---:|
| `put both moka pots on the stove` | 26 | 0 | 26 |
| `turn on the stove and put the moka pot on it` | 17 | 6 | 23 |
| `put the white mug on the plate and put the chocolate pudding to the right of the plate` | 19 | 0 | 19 |
| `put the black bowl in the bottom drawer of the cabinet and close it` | 9 | 9 | 18 |
| `put the yellow and white mug in the microwave and close it` | 11 | 5 | 16 |
| `put both the alphabet soup and the tomato sauce in the basket` | 12 | 0 | 12 |
| `put the white mug on the left plate and put the yellow and white mug on the right plate` | 12 | 0 | 12 |
| `put both the cream cheese box and the butter in the basket` | 9 | 0 | 9 |
| `put both the alphabet soup and the cream cheese box in the basket` | 8 | 0 | 8 |
| `pick up the book and place it in the back compartment of the caddy` | 7 | 0 | 7 |

## 任务格式对训练的含义

- 历史常见格式是英文祈使句：动作 + 定冠词 + 目标物；必要时加空间限定或抓取部位。`pick up the white mug` / `grasp the white mug` 均为真实记录。
- 真实的部位限定包括 `grasp the white mug by its handle`、`grasp the yellow mug handle`、`grasp the silver moka pot by its body`。这仅表示尝试过这些条件，不证明抓法有效。
- 杯柄不是训练目标或额外奖励。默认指令不指定抓取部位，抓杯壁或杯柄都按最终任务成功和稳定性评价；确有部位要求时再使用部位限定句。
- 将 `pick up the white mug` 用于直立、倾斜、侧翻等不同初态，避免把姿态变化全部编码成新的任务名；若目标是扶正并放好，应使用含有放置和直立目标的新描述。
- 建议新增 `place the white mug upright on the plate`、`pick up the white mug and place it upright on the plate`。这两句是本轮拟定的训练格式，未出现在这 309 次调用中。
- 历史几乎没有独立稳定放置短指令；原系统由 move_to / release 完成搬运和释放。训练放置后必须设计对应的策略调用与终止语义，不能只替换权重后期待原脚本流程自动采用新技能。
- GAWM 当前任务向量只支持旧词表。若采用短技能句，需新增并训练相应条件，同时保留旧任务条件作为对照；不通过将所有新任务强行映射到旧向量来声称已对齐。
- 历史 max_chunks、抓取阈值属于执行参数，不拼接进任务描述。短句与长句、不同后端、技能类别应分层采样；不要按这 309 次调用的重试频次直接分配训练任务。

## 来源

- [固定审计快照](../long120_audit_20261029/audit.json)
- [旧函数统计](../function_call_tables.md)
- [RPent 工具实现](../../RPent/robots/libero/tools.py)
- [GAWM 批量实验指令约束](../../scripts/long_benchmark_entrypoint.py)
- 快照 SHA-256：`5055671fc366cd06fe9e9d63c7720153fbd7a1fd08d6fc28b992a1d8811e31e3`。
