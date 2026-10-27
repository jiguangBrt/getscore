# Kimi 辅助机器人任务：模型、推理配置与 GPT 差距调查

调查日期：2026-10-23（北京时间）。范围为本次讨论的 GPT-Policy，以及具有直接模型比较的 EmbodiedSkillBench、CodeActionBench、RobotWorld；另将 Robot-eval-bench 作为成功检测参考。论文事实、当前源码配置与本项目建议分别记录。本调查未调用付费模型、运行机器人或重新评测，也不改变项目现有技术路线。

## 1. 主要结论

- **没有一个可以跨任务通用的“Kimi 落后 GPT 多少”。** 必须说明具体型号、控制分工、输入、Agent 框架、预算和评分指标。
- **最接近“高层模型 + VLA 技能”的证据来自 EmbodiedSkillBench。** 在同一批选定技能下，Kimi K3 与 GPT-5.6 Sol 的报告成功率为 62% 与 63%；这一个百分点不足以证明稳定优劣。
- **直接控制或编写机器人代码时，某些评测差距明显更大。** CodeActionBench 中 Kimi K3/High 为 12%，GPT-6 Astra/High 为 73.33%；但 GPT-5.6/High 只有 16%，且 Astra 与 Kimi 的 Agent 框架不同。
- **成功检测应单独比较。** Robot-eval-bench 中，Kimi 与 GPT-5.6 Sol 的判断准确率相差 4 或 8 个百分点；这不是机器人执行成功率。
- **推理档位存在证据缺口。** 当前默认配置不能代替历史实验记录；不同厂商相同的 High/Medium 标签也不意味着相同的实际推理预算。

## 2. 已归档材料与阅读入口

三份新增 PDF 均从 arXiv 下载，固定为 v1，保留原始内容。已有 GPT-Policy.pdf 原样保留。原始 PDF 下载后经过文件格式与可读性检查；版本、页数与 SHA-256 见文末。

| 材料 | 本地文件 | 固定版本 | 建议先读 |
| --- | --- | --- | --- |
| In-Context Robot Learning with VLM Agents | [GPT-Policy](paper/GPT-Policy.pdf) | 2609.19138v1，已有文件 | 第 5 节、表 2：模型比较及其局限 |
| Embodied Skill Bench: Benchmarking How Well Agents with Embodied Skills Perform on Physical Tasks | [EmbodiedSkillBench](paper/EmbodiedSkillBench.pdf) | 2610.16238v1，新增，24 页 | 第 5.1—5.4 节、表 1/3；附录 A.2、表 5 |
| CodeActionBench: Evaluating Agentic Code-as-Policy for Embodied Manipulation | [CodeActionBench](paper/CodeActionBench.pdf) | 2609.33807v1，新增，28 页 | 主结果、附录 C 表 8、附录 D 表 11 |
| RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments | [RobotWorld](paper/RobotWorld.pdf) | 2610.10409v1，新增，62 页 | 第 5.1—5.3 节；附录 F 的比较范围 |
| Robot-eval-bench | [公开代码与数据](https://github.com/kite-ml/robot-eval-bench) | 评测项目，不作为论文归档 | README 的准确率、负样本构造与数据范围；src/judges.py 的模型配置 |

这不是穷尽所有 Kimi 具身研究的文献综述，而是对本次选出的、有直接机器人相关对照证据的材料进行核查。成功检测项目与机器人执行论文分开解释。

## 3. 模型和推理强度核查

“已知”表示能在指定来源中直接读到；“未明确”表示当前证据不能确定，不等于模型没有思考或使用最低档位。

| 工作 | Kimi 模型与调用方式 | GPT 对照 | 推理强度的证据状态 |
| --- | --- | --- | --- |
| GPT-Policy | 论文标为 Kimi K3；当前配置为 `kimi-k3[1m]`，经 Claude Code 兼容路径调用 | GPT-6 Astra，Codex | 当前两个配置均为 `medium`；论文表 2 的具体历史档位未核实 |
| EmbodiedSkillBench | Kimi-K3，Kimi Code CLI 0.40.0 | GPT-5.6 Sol，Codex CLI 0.148.0；RQ3 另测 GPT-6 Astra | RQ1 主比较未明确披露实际档位。附录的 Astra/Jev 配置及关系碗案例明确出现 `medium`，不据此推断全部实验 |
| CodeActionBench | `kimi-k3`，Reference harness | `gpt-6-astra`，Codex CLI 0.154.0；`gpt-5.6`，Reference harness | 表 8 明确三者均请求 `High`；Kimi 每次请求输出上限 131,072，GPT-5.6 为 128,000，Astra 由框架管理 |
| RobotWorld | Kimi K3 | GPT-6 Astra | 在核查到的实验说明中未找到可确认的明确 effort 档位 |
| Robot-eval-bench | `kimi-k3`，Moonshot 兼容 API | `gpt-5.6-sol` | 模型注册表未明确记录实际 effort；保留服务端默认采样不等于已确认默认推理档位 |

配置依据：[GPT-Policy Kimi](https://github.com/cheng-haha/GPT-Policy/blob/2ca201638e1139fe35973c8a2d209a894b389db1/configs/agents/kimi.json)、[GPT-Policy Codex](https://github.com/cheng-haha/GPT-Policy/blob/2ca201638e1139fe35973c8a2d209a894b389db1/configs/agents/codex.json)、[EmbodiedSkillBench 第 5.1 节与附录 C](https://arxiv.org/html/2610.16238v1)、[CodeActionBench 附录 C](https://arxiv.org/html/2609.33807v1)、[RobotWorld 第 5 节](https://arxiv.org/html/2610.10409v1)、[Robot-eval-bench 模型注册](https://github.com/kite-ml/robot-eval-bench/blob/e23bf79e1f944ec56281169176dad1621cc144b4/src/judges.py)。

### 3.1 为什么不能把配置标签当成实际推理算力

1. GPT-Policy 当前源码会把 `effort` 传给 Claude Code，但 Kimi 还经过服务兼容层；仅看本地字段，不能确认后端如何映射，也不能还原历史运行。
2. EmbodiedSkillBench 的公开 [agents.lock.json](https://github.com/HanZhi306/EmbodiedSkill-Bench/blob/3a6c645aac388eeff0a41dc2bb42174f55e3472b/configs/agents.lock.json) 中 `agents` 为空。运行器中出现默认 `medium` 字段，不足以恢复论文的全部 Agent 配置，更不能假设每个 provider 分支都按同一种方式传参。
3. CodeActionBench 明确指出，各厂商定义自己的 effort；同为 High 并不保证同样的 reasoning-token 预算。
4. 输出 token 上限不是实际使用量，也不等于独立的思考 token 上限。温度、上下文长度和推理档位同样是不同参数。

因此，现有材料无法回答“把 Kimi 从 Medium 提到 High，可以追回多少百分点”。需要固定环境和框架单独做档位实验。

## 4. 结果对照与解释

下表的“差距”按 **GPT 数值减去 Kimi 数值** 计算；单位为百分点。不同指标不能跨行求平均。

| 工作与条件 | Kimi | GPT 对照 | 差距 | 指标/样本 |
| --- | ---: | ---: | ---: | --- |
| GPT-Policy，人类视频辅助抓红毛巾 | 20% | Astra 100% | +80 | 个别运行的任务进度，不是成功率 |
| EmbodiedSkillBench，提供相同选定技能 | 62% | GPT-5.6 Sol 63% | +1 | Clean100，seed 26，100 个案例 |
| EmbodiedSkillBench，无技能接口条件 | 43% | GPT-5.6 Sol 11% | -32 | 同一任务集合的报告结果；框架和预算仍需核对 |
| CodeActionBench，Kimi/Reference 对 Astra/Codex | 12% | Astra 73.33% | +61.33 | 25 个任务×3 次，每配置 75 回合 |
| CodeActionBench，双方使用 Reference | 12% | GPT-5.6 16% | +4 | 每配置 75 回合，均请求 High |
| RobotWorld | 2/84，约 2.38% | Astra 16/84，约 19.05% | +16.67 | 84 个任务，每模型每任务一回合 |
| Robot-eval-bench，4 张关键帧 | 86% | GPT-5.6 Sol 90% | +4 | 任务成功判断准确率 |
| Robot-eval-bench，约 16 帧表示视频 | 84% | GPT-5.6 Sol 92% | +8 | 任务成功判断准确率 |

### 4.1 GPT-Policy：可以说明一个失败例子，不能说明普遍差距

本地 v1 的表 2 中，两者都提供 Human Video：Astra 运行 15.85 分钟、任务进度 100%；Kimi K3 运行 11.73 分钟、进度 20%。表中估计总 token 分别为 4.956M 与 1.735M。这些是个别运行，作者明确没有据此建立可靠排名。较短运行也可能来自未完成任务，不能据此判断 Kimi 更高效。

对本项目的意义：低层位姿生成的个别表现不足以预测高层技能调用效果；不能把 Kimi 的 20% 进度写成 20% 成功率。

### 4.2 EmbodiedSkillBench：与高层模型调用 VLA 最相关

表 1 的 with-skill 比较为各模型提供相同的任务对应技能，技能由 GPT-5.6 Sol 先选择，包含动作、感知和经验资源。Kimi K3 为 62%，GPT-5.6 Sol 为 63%；平均每任务动作调用为 4.37 与 2.38，耗时为 15.36 与 4.40 分钟，均包含失败回合。耗时比约 3.49，不是纯推理延迟比。

限制：只有 seed 26；Agent CLI 不同；技能选择过程由一个对照模型完成。附录 A.2 的无技能案例还分别在 GPT 的 40 次与 Kimi 的 25 次动作预算边界结束，说明不能默认所有跨模型预算完全相同。62% 与 63% 只能支持“这批配置下结果接近”。

第 5.4 节另外测 Astra：无技能 92%、技能执行 61%、保留原生控制并自主复用技能的 ADR 为 85%。该结果说明技能是否有益取决于模型和执行方式，不能把“增加 VLA 必定提高成功率”作为前提。它是另一组设置，不能将所有模式混合成单一模型排名。

### 4.3 CodeActionBench：差距包含模型、框架与执行策略

这项评测让模型从视觉出发构造几何估计、编写代码并执行机器人动作，不提供 VLA 或预制任务技能。表 8 的请求档位均为 High，但 Astra 使用 Codex，Kimi 与 GPT-5.6 使用 Reference harness。

因此，Astra 与 Kimi 的 +61.33 点反映完整 Agent 配置差异；共享 Reference 的 GPT-5.6 与 Kimi 则相差 +4 点。每任务三个重复、固定场景，不等于广泛布局泛化。不能把这项低层代码控制结果直接迁移为 RPent 高层规划器的预期差距。

### 4.4 RobotWorld：广覆盖但低重复的探索性证据

84 个任务覆盖操作、移动操作、行走、驾驶和飞行，各模型使用匹配的任务初始化协议、控制步数上限与评分规则。Astra 成功 16 项，Kimi 成功 2 项；每模型每任务仅一回合，不反映重复实验波动。

论文附录 F 也将这组数据界定为匹配任务上的观察性比较，而非历史、代码控制或推理预算的受控消融。它适合分析恢复、状态跟踪等失败机制，不支持推出一个通用的“机器人模型能力差距”。

### 4.5 Robot-eval-bench：检测结果的项目，不能当作执行论文

任务是根据轨迹图片判断成功/失败，覆盖 16 个 LeRobot 数据集。关键帧条件使用 4 张图，非原生视频模型的视频条件使用约 16 张图。Kimi 相对 GPT-5.6 Sol 的准确率差距是 4/8 点。

部分负样本由成功示范截断、任务错配、打乱或倒放构造，实际机器人失败还可能有滑落、遮挡、部分完成等不同分布。README 数值可用于选择候选检测器，但上线前应测本项目真实失败样本，尤其是把失败误报为成功的比例。该项目没有作为本次论文下载，不人为生成一份“论文 PDF”。

来源：[Robot-eval-bench 结果与数据说明](https://github.com/kite-ml/robot-eval-bench/blob/e23bf79e1f944ec56281169176dad1621cc144b4/README.md)。

## 5. 对本项目的建议：先测清规划与检测，再决定替换范围

本节是实验建议，不是已经取得的结果。当前项目 README 记录了 RPent、LIBERO-PRO 与远程动作服务的探索进展；本次调查没有重跑或确认这些服务状态，也没有修改路线图、模型配置或启动入口。

### 5.1 高层规划器对照

以当前已跑通的系统为基线，优先只替换高层模型，固定动作模型/checkpoint、感知工具、技能接口、任务、种子、观测和动作预算。先比较少量任务的目标选择、阶段切换、异常识别与恢复；有重复收益后再扩大任务集。

| 对照维度 | 需要控制或记录 |
| --- | --- |
| 模型 | 精确 API ID、服务商、版本；不能只写 GPT/Kimi |
| 推理设置 | 实际发出的 effort/thinking 参数、后端是否接受、输出限制与实际 token |
| 框架 | 尽量共享提示、工具和上下文管理；无法共享时将 CLI/适配器列为系统差异 |
| 执行能力 | 同一 VLA 或世界模型及相同技能能力，避免同时换模型和小脑 |
| 输入信息 | 相同观测、历史、示范与人工标注；固定开发期制作流程 |
| 预算与失败 | 控制步数、技能调用、墙钟上限、重试与基础设施失败统一记账 |
| 结果 | 官方成功率、目标/阶段错误、恢复成功率、实际耗时、调用与费用 |

可以在各模型确实支持的档位内分别比较，但 Medium/High 只作为各自配置标签，不宣称为等算力实验。所有失败回合保留，不挑选成功重跑替换正式成绩。

### 5.2 成功检测器对照

从本项目实际执行记录中建立包含成功与真实失败的固定样本集，对同样的图像序列独立比较检测模型。除准确率外，至少报告失败误报成功、成功误报失败、拒答/不确定、延迟与成本。内部检测器的判断与官方环境成功信号分别保存。

### 5.3 当前能够支持的决定

Kimi K3 值得作为技能编排和结果检查的实验候选，但尚没有证据保证它在本项目与某一 GPT 配置接近。也没有找到能量化 Kimi 推理档位收益的机器人受控消融。是否替换应依据上述本地对照，而不是对文献差距做平均或使用单个任务示例定论。

## 6. 来源、版本与归档校验

### 6.1 PDF 文件

| 文件 | 来源 | 页数 | SHA-256 |
| --- | --- | ---: | --- |
| GPT-Policy.pdf（已有，未覆盖） | [2609.19138v1](https://arxiv.org/abs/2609.19138v1) | 原归档版本 | `cbef940f06a274de0357a0711e019929289bab610ef6f371674c70cafcd1a176` |
| EmbodiedSkillBench.pdf | [原始 PDF：2610.16238v1](https://arxiv.org/pdf/2610.16238v1) | 24 | `0f5caed0de70e51f7d60af0658d6c7778e0e4a44a6ce151a98c74960d0521a97` |
| CodeActionBench.pdf | [原始 PDF：2609.33807v1](https://arxiv.org/pdf/2609.33807v1) | 28 | `18beecffa8e22b922f35a6068a1ac50917e8519ecdcc3637766ed30cece32a88` |
| RobotWorld.pdf | [原始 PDF：2610.10409v1](https://arxiv.org/pdf/2610.10409v1) | 62 | `7a6a53e4d7f62bcfaa4b57342fd6b3919134e648bd8e5a7dd1b563edb8bb42c6` |

页数依据实际下载文件的 PDF 元数据；例如 EmbodiedSkillBench 的 arXiv 摘要页备注为 25 页，实际本次下载 PDF 为 24 页，归档表按文件实测记录。

### 6.2 公开代码的核查版本

以下是本次读取时 main 对应的提交，便于区分当前代码与论文历史实验；提交日期不等于实验日期。

| 仓库 | 提交 | 提交时间（UTC） | 本次用途 |
| --- | --- | --- | --- |
| [cheng-haha/GPT-Policy](https://github.com/cheng-haha/GPT-Policy/tree/2ca201638e1139fe35973c8a2d209a894b389db1) | `2ca201638e1139fe35973c8a2d209a894b389db1` | 2026-10-12 11:56:34 | 默认模型、effort、provider 路径 |
| [HanZhi306/EmbodiedSkill-Bench](https://github.com/HanZhi306/EmbodiedSkill-Bench/tree/3a6c645aac388eeff0a41dc2bb42174f55e3472b) | `3a6c645aac388eeff0a41dc2bb42174f55e3472b` | 2026-10-10 10:49:50 | 配置锁与运行器；未从空配置恢复实验档位 |
| [kite-ml/robot-eval-bench](https://github.com/kite-ml/robot-eval-bench/tree/e23bf79e1f944ec56281169176dad1621cc144b4) | `e23bf79e1f944ec56281169176dad1621cc144b4` | 2026-09-22 20:30:17 | 模型 ID、判断输入、报告结果与负样本定义 |

### 6.3 关键证据定位

- GPT-Policy：本地 PDF 第 5 节、表 2；默认配置另看上面的两个 JSON，不能混为同一时间点的证据。
- EmbodiedSkillBench：本地 PDF 第 6 页为实验设置，第 7 页表 1 为跨模型主比较，第 9 页表 3 为 Astra 各模式，第 14 页表 5 与案例文字显示无技能案例预算，第 21 页附录给出 Jev 配置。
- CodeActionBench：本地 PDF 第 19 页表 8 为模型与请求 effort，第 22 页表 11 为资源统计。网页项目名有时显示 Kimi K3 Intl，本报告以论文表 8 的具体 ID `kimi-k3` 为准。
- RobotWorld：[第 5 节与附录 F](https://arxiv.org/html/2610.10409v1) 为实验协议、主结果与比较边界。论文的控制工具模式不应直接当成统一 VLA 技能调用模式。
- Robot-eval-bench：上面固定提交的 README、`src/judges.py`。较新模型条目可能使用不同发布范围，本报告只比较已经报告的 Kimi K3 与 GPT-5.6 Sol 原有结果，不将新模型条目直接补入旧比较。
