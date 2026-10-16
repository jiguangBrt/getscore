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

RPent 源码已作为 `RPent/` 子模块引入；本次核查的提交为 `efc392adc98a30045d8fb83c4097917f28e20a94`。origin 为 `https://github.com/jiguangBrt/RPent.git`，upstream 为 `https://github.com/RLinf/RPent.git`。尚未验证环境、模型服务或基线运行成功。

本地 RPent 已有 LIBERO / LIBERO-PRO 的 π0.5 集成，可作为搭建入口参考。具体仿真环境、GPT-6 版本、规划器、推理强度、权重、memory 和任务清单在搭建时核实确定；不把示例配置当作用户已选定方案。实际运行后记录版本、命令和结果。

项目此前记录的长期榜单目标与资源见 [ROADMAP](ROADMAP.md) 和 [STEP1_RULES](STEP1_RULES.md)。当前复现用于探索框架，后续如何衔接目标榜单仍随实验推进。

## data 目录

`data/gpt-policy-robodojo-contexts/` 是此前调研 GPT-Policy 下载的 RoboDojo 专家示范包，约 37 MB，包含 10 份示范、480 张图片及阶段、状态和动作参考。当前 RPent 代码没有引用该目录；它是调研资料，不是本轮复现所需资源或已经使用的训练集。详见 [中文导读](paper/DEMO_PACKAGE_GUIDE.md)。

## 本地参考入口

- [RPent 中文说明](RPent/README.zh-CN.md)
- [快速开始](RPent/docs/source-zh/rst_source/get_started/quickstart.rst)
- [LIBERO / LIBERO-PRO](RPent/docs/source-zh/rst_source/simulators/libero.rst)
- [规划器配置](RPent/docs/source-zh/rst_source/guides/configure_planner.rst)
- [记忆与探索](RPent/docs/source-zh/rst_source/guides/memory.rst)
