"""Format audited local data as readable Chinese reports."""
from collections import Counter
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
audit = json.loads((OUT / 'audit.json').read_text())
stats = json.loads((OUT / 'tool_counts.json').read_text())
assert not audit['failures'] and audit['audited_runs'] == 120
runs = audit['runs']
llm = [r for r in runs if r['planner'] != 'none']
task_names = ['字母汤和番茄酱放入篮子', '奶油奶酪盒和黄油放入篮子', '打开炉灶并放上摩卡壶',
              '黑碗放入柜子底层抽屉并关上', '白杯放左盘，黄白杯放右盘', '书放入收纳架后侧隔间',
              '白杯放盘中，巧克力布丁放盘右侧', '字母汤和奶油奶酪盒放入篮子',
              '两个摩卡壶放上炉灶', '黄白杯放入微波炉并关门']
families = [
    ('感知与观察', ['segment', 'back_project', 'view_env_state']),
    ('机器人动作', ['move_to', 'move_pose', 'pi0_pick', 'pi0_doubled', 'set_gripper', 'release', 'rotate_wrist', 'rotate_pitch']),
    ('文件与结束操作', ['read_text_file', 'write_text_file', 'list_dir', 'finish']),
]
assert set(stats['tools']) == {t for _, ts in families for t in ts}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |' for row in rows])


def task_tables(groups):
    parts = []
    for label, tools in families:
        parts += [f'### {label}', '', table(['任务 ID', *tools],
                  [[g['task'], *[g['calls'][t] for t in tools]] for g in groups] +
                  [['合计', *[sum(g['calls'][t] for g in groups) for t in tools]]]), '']
    return parts


segments = [s for r in runs for s in r['segments']]
resumed = [r for r in runs if r['execution_mode'] == 'resume']
anomalies = [(r, s, a) for r in runs for s in r['segments'] for a in s['anomalies']]
failed = Counter()
for r in llm:
    failed.update(r['failed_mcp_counts'])
failed_runs = sum(bool(r['failed_mcp_counts']) for r in llm)
lines = ['# long120_1027 记录完整性与函数调用统计', '',
         f"检查时间：{audit['checked_at']}。数据目录：`{audit['source']}`。", '',
         '## 结论', '',
         '- 120 个预期组合全部存在，120/120 的运行数据通过本次独立核验；共 63 次任务成功、57 次失败。失败表示任务未完成，不等于记录缺失。',
         '- 117 次的当前目录记录完整一轮 episode；另 3 次是续跑，当前目录的状态、轨迹和工具日志仅包含后半段。前 1000 步的对应记录保存在归档中，也通过核验。读取当前目录而不追溯归档会漏计。',
         '- 1 次当前模型会话以超时结束（见下文），因此不能说全部模型会话都正常收尾；该次 2000 步任务失败的轨迹、动作、视频及已完成工具调用记录齐全。',
         '- 使用语言模型的是 80 次：GPT 40 次、Kimi 40 次；其余 40 次是 `planner=none` 的纯策略基线。', '',
         '## 本次核验范围', '',
         f"实际检查了 {len(segments)} 段记录（120 个当前目录 + 3 个续跑前缀归档），核对 {sum(s['sha256_files_checked'] for s in segments)} 项已记录 SHA-256、{sum(s['observation_files_checked'] for s in segments)} 项状态引用文件。", '',
         '核验内容包括预期实验组合、必需文件、完整状态步号、命令与状态对应、原始轨迹的数组形状/类型/有限值、原生成功与截断标记、动作数、策略查询数、动作帧归属、文件哈希、全部视频逐帧解码、查看器解析、MCP 事件闭合和计数。Kimi 的归一化事件另与原始 `kimi_tool_calls.jsonl` 逐条核对函数、参数和错误状态。', '',
         '此处“完整”指已执行实验及已完成工具调用的记录可恢复；不表示所有实验成功，也不表示各实验使用相同预算或严格固定家具布局。', '',
         '### 续跑记录来源', '',
         table(['运行', '归档前缀步数', '续跑步数', '整轮步数', '前缀目录'],
               [[r['name'], r['segments'][0]['actions'], r['segments'][1]['actions'],
                 sum(s['actions'] for s in r['segments']), r['segments'][0]['directory']] for r in resumed]), '',
         '### 会话异常与汇总字段不一致', '']
for r, s, a in anomalies:
    lines += [f"- `{r['name']}`（{s['role']}）：`{a['transcript_error']}`；退出码 `{a['exit_code']}`；原生成功 `{a['native_success']}`、截断 `{a['native_truncated']}`。`transcript.error` 有超时信息，但 `benchmark_result.error` 为 `{a['result_error']}`；`planner_completion_error` 中保留了该信息。", '']
lines += ['## 统计口径', '',
          '- 每个任务汇总 8 次实验：original/swap × GPT/Kimi × GAWM/π0.5。下面分别提供合计和两种场景各 4 次的表。',
          '- 只统计模型调用的 RPent MCP 函数，每个独立完成事件计 1 次。返回错误的调用也计入，因为调用确实发生；失败次数另表列出。',
          '- `move_to`、`back_project`、`pi0_pick` 是日志中的实际名称，分别对应提问中的 moveto、backproject、pi0pick。',
          '- `pi0_pick` / `pi0_doubled` 按外层函数调用计数，不按内部动作 chunk、底层策略推理请求或仿真步数计数。后端是 GAWM 的实验也可能调用名称为 `pi0_pick` 的统一接口。',
          '- 3 次续跑合并归档前缀与续跑后缀；恢复时自动执行的 `resume_pending_policy_actions` 不视为新模型函数调用。重跑只统计当前保留的那一轮，已替换的旧轮不累加。',
          f"- GPT 的本地命令执行另计 {sum(r['shell_command_count'] for r in llm)} 次，不混入 RPent 函数表；感知、文件和 finish 函数均保留。", '',
          f"80 次语言模型实验共调用 RPent 函数 **{stats['total']} 次**，其中 **{sum(failed.values())} 次返回错误**，涉及 {failed_runs} 次实验。", '',
          '## 任务编号', '', table(['任务 ID（日志原始 0–9）', '任务'], list(enumerate(task_names))), '']
for variant, label in [('all', '全部 80 次：每任务 8 次'), ('original', 'original 40 次：每任务 4 次'), ('swap', 'swap 40 次：每任务 4 次')]:
    lines += [f'## {label}', ''] + task_tables([g for g in stats['by_task'] if g['variant'] == variant])
lines += ['## 返回错误的函数调用', '',
          table(['函数', '调用总数', '返回错误数'], [[t, stats['total_calls'][t], failed[t]] for t in stats['tools']]), '',
          '## 按模型、场景、策略后端汇总', '']
group_rows = []
for variant in ['original', 'swap']:
    for planner in ['gpt6', 'kimi']:
        for policy in ['gawm', 'pi05']:
            selected = [r for r in llm if (r['variant'], r['planner'], r['policy']) == (variant, planner, policy)]
            counts = Counter()
            for r in selected:
                counts.update(r['full_episode_mcp_counts'])
            group_rows.append([variant, planner, policy, len(selected), sum(counts.values()),
                               counts['move_to'], counts['segment'], counts['back_project'], counts['pi0_pick']])
lines += [table(['场景', '模型', '后端', '实验数', '全部函数', 'move_to', 'segment', 'back_project', 'pi0_pick'], group_rows), '',
          '## 可追溯明细', '',
          '- [逐实验函数次数](per_run.md)：80 次实验各自的所有函数调用次数。',
          '- [统计原始结果](tool_counts.json)：每任务、每场景、每实验计数及错误次数。',
          '- [完整核验结果](audit.json)：120 次实验、3 个前缀归档及逐调用来源文件/行号。',
          '- [核验程序](audit.py)：只读取源记录，报告单独写入本目录。', '']
(OUT / 'report.md').write_text('\n'.join(lines))
detail = ['# 80 次语言模型实验逐次函数调用次数', '', '口径及完整性说明见 [主报告](report.md)。3 次续跑均已合并归档前缀。返回错误的调用包含在总数中。', '']
for label, tools in families:
    detail += [f'## {label}', '', table(['运行', *tools],
               [[r['name'], *[r['full_episode_mcp_counts'].get(t, 0) for t in tools]] for r in llm]), '']
(OUT / 'per_run.md').write_text('\n'.join(detail))
print('Report:', OUT / 'report.md')
print('Total calls:', stats['total'], 'failed:', sum(failed.values()))
print(table(['task', 'move_to', 'segment', 'back_project', 'pi0_pick', 'all'],
            [[g['task'], *[g['calls'][t] for t in ['move_to', 'segment', 'back_project', 'pi0_pick']], sum(g['calls'].values())]
             for g in stats['by_task'] if g['variant'] == 'all']))
