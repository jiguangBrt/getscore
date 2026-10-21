# 第 3 步准备：RPent 复现与探索

更新日期：2026-10-16。

**状态：探索阶段，准备搭建仿真环境并复现基线。技术路线尚未确定。**

## 当前意向

- 初步参考 RPent 框架，尝试复现 π0.5 + GPT-6。
- 先了解服务启动的信息流、模块之间的交互和任务执行过程。
- 后续考虑将 π0.5 换成更快、更小的动作模型，具体候选和实验方式待定。

本记录只描述近期探索意向。是否采用固定技能接口、是否把技能写死，以及高层规划和动作控制如何组织，都未定案。RPent 现有工具或技能设计是观察对象，不代表本项目要原样采用，也不代表已经决定改成另一种设计。

## 复现时关注的问题

- 启动哪些服务，各自接收和返回什么信息？
- GPT 和 π0.5 分别在什么时候被调用？
- 观测如何进入模型，模型输出如何变成环境动作？
- 一次执行之后，哪些结果会返回给规划器？
- RPent 的现有接口带来什么便利或限制，哪些问题值得进一步实验？

先通过实际运行、代码、日志和视频了解这些问题，再决定自己的设计。

## 已有资料与待确定配置

RPent 源码已作为 `RPent/` 子模块引入；基于 `efc392adc98a30045d8fb83c4097917f28e20a94` 增加本地部署适配，当前子模块提交为 `e2eac0c29f3da273a96e96c1cd14c54b34b66a58`。origin 为 `https://github.com/jiguangBrt/RPent.git`，upstream 为 `https://github.com/RLinf/RPent.git`。适配已提交并推送至个人 fork 的 `codex/liberopro-remote-openpi` 分支。

2026-10-19 已按用户选择配置本机 LIBERO-PRO、远程官方 π0.5-LIBERO/SAM3 和本地 fake 规划器，先使用阻塞式同步推理。真实链路验证了 3 次模型调用、15 个仿真动作及视频记录；GPT token 用量为 0。该检查不代表完整任务成功或基线成功率复现。配置、启动、超时与 RTC 范围、版本及验证记录见 [本地运行说明](LOCAL_LIBEROPRO.md)。

2026-10-20 已用 RPent 原生 `codex` 后端接入 GPT-6 Astra，推理强度为 `low`，复用本机 ChatGPT 登录，无需提供 API key。首轮 `libero_object_swap / task 2 / seed 0` 在一个连续 episode 内完成抓取和放置，原生环境成功；记录到 19 次模型回复、16 次 π0.5 请求和 344 个仿真动作。该单例不代表完整 benchmark 成功率，启动与证据见上述运行说明。

项目此前记录的长期榜单目标与资源见 [ROADMAP](ROADMAP.md) 和 [STEP1_RULES](STEP1_RULES.md)。当前复现用于探索框架，后续如何衔接目标榜单仍随实验推进。

2026-10-21 已接入远程 GAWM：先通过 fake 链路检查，再以真实 GPT-6 Astra/low 在同一任务、seed 0 的连续 episode 内完成沙拉酱入篮。成功轮执行 447 个原生动作；此前一轮因初始化稳定步数与 episode 上限错位失败，已修复并保留记录。详见 [GAWM 运行说明](LOCAL_GAWM_LIBERO.md)。

## data 目录

`data/gpt-policy-robodojo-contexts/` 是此前调研 GPT-Policy 下载的 RoboDojo 专家示范包，约 37 MB，包含 10 份示范、480 张图片及阶段、状态和动作参考。当前 RPent 代码没有引用该目录；它是调研资料，不是本轮复现所需资源或已经使用的训练集。详见 [中文导读](paper/DEMO_PACKAGE_GUIDE.md)。

## 本地参考入口

- [RPent 中文说明](RPent/README.zh-CN.md)
- [快速开始](RPent/docs/source-zh/rst_source/get_started/quickstart.rst)
- [LIBERO / LIBERO-PRO](RPent/docs/source-zh/rst_source/simulators/libero.rst)
- [规划器配置](RPent/docs/source-zh/rst_source/guides/configure_planner.rst)
- [记忆与探索](RPent/docs/source-zh/rst_source/guides/memory.rst)
