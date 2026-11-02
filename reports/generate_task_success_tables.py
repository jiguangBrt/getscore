"""Render the latest original/swap 10-by-6 results, with explicit archived fallbacks."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
DATA = ROOT / 'outputs/long120_1027'
BATCH = ROOT / 'outputs/_archive/records/rerun-left-right-2000step-20261029'
read = lambda p: json.loads(p.read_text())
selection = read(BATCH / 'selection.json')
old = {i['name']: Path(i['archived_source']) for i in selection['cases']}
labels = {(r['suite'], r['task_id']): r['language_zh'] for r in read(
    ROOT / 'data/libero-original-pro-task-swap-comparison-20261023/tasks_zh.json')}
columns = [('none', 'pi05', '纯 π0.5'), ('none', 'gawm', '纯 GAWM'),
           ('gpt6', 'pi05', 'GPT6 + π0.5'), ('gpt6', 'gawm', 'GPT6 + GAWM'),
           ('kimi', 'pi05', 'Kimi + π0.5'), ('kimi', 'gawm', 'Kimi + GAWM')]


def existing_annotations(path):
    """Keep manually added model comments and the optional remarks column."""
    annotations, remarks, variant = {}, set(), None
    if not path.exists():
        return annotations, remarks
    for line in path.read_text(encoding='utf-8').splitlines():
        if line == '## 原版场景':
            variant = 'original'
        elif line == '## Swap 场景':
            variant = 'swap'
        elif variant and line.startswith('|'):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if cells[0] == '中文任务名' and '备注' in cells:
                remarks.add(variant)
            if not cells[0].split(' · ')[0].isdigit():
                continue
            task = int(cells[0].split(' · ')[0])
            assert len(cells) in (7, 8), f'Unexpected table columns: {line}'
            notes = []
            for cell in cells[1:7]:
                assert cell.startswith(('✓', '×')), f'Missing result symbol: {cell}'
                comment = re.sub(r'^\s*\d+:\d{2}(?=<br>|$)', '', cell[1:]).strip()
                notes.append(comment.removeprefix('<br>').strip())
            annotations[(variant, task)] = (notes, cells[7] if len(cells) == 8 else '')
    return annotations, remarks


def duration_label(seconds):
    rounded = int(seconds + 0.5)
    return f'{rounded // 60:02d}:{rounded % 60:02d}'


def column_summary(records, variant=None):
    summary = []
    for _, _, model in columns:
        group = [r for r in records if r['model'] == model and (variant is None or r['variant'] == variant)]
        successes = [r for r in group if r['success']]
        mean = math.fsum(r['elapsed_s'] for r in successes) / len(successes) if successes else None
        summary.append(dict(model=model, total=len(group), successes=len(successes),
            success_rate=len(successes) / len(group), mean_success_elapsed_s=mean,
            mean_display=duration_label(mean) if mean is not None else '—',
            rate_display=f'{len(successes)}/{len(group)}（{100 * len(successes) / len(group):.0f}%）'))
    return summary


annotations, remarks = existing_annotations(OUT / 'task_success_tables.md')
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
provenance, markdown = [], ['# 任务成功情况', '',
    '每张表为 10 个任务 × 6 种模型组合。✓ 成功，× 未成功；以环境原生成功标记为准。', '',
    '成功项勾号后的时间为完整实验耗时，格式为 **分:秒**，四舍五入到秒。包含语言模型思考、策略推理、网络等待、工具执行以及启动和收尾；不是视频帧时长。来源为 `benchmark_result.json` 的 `elapsed_s`。', '',
    '每列末尾的平均成功耗时只对成功项取算术平均；成功率 = 成功项数 / 该列任务数。无成功项时平均耗时记为“—”。', '',
    'GPT6 = gpt-6-astra / low；Kimi = kimi-code/k3.1-pro / high。本次 4/6 题重跑结果已更新；其余保留现有记录，步数预算并不完全相同。', '']

for variant, suite, title in [('original', 'libero_10', '原版场景'), ('swap', 'libero_10_swap', 'Swap 场景')]:
    matrix, timings, fallback = [], [], set()
    for task in range(10):
        row, times = [], []
        for col, (planner, policy, model) in enumerate(columns):
            name = f'long-{variant}-{task}-{planner}-{policy}'
            directory = DATA / name
            archived = not (directory / 'benchmark_result.json').is_file()
            if archived:
                assert name in {'long-swap-6-gpt6-pi05', 'long-swap-6-gpt6-gawm'}
                directory = old[name]
                fallback.add((task, col))
            result = read(directory / 'benchmark_result.json')
            assert (result['variant'], result['task'], result['planner'], result['policy']) == (variant, task, planner, policy)
            assert not result.get('error')
            row.append(bool(result['success']))
            elapsed = float(result['elapsed_s'])
            assert math.isfinite(elapsed) and elapsed > 0, name
            times.append(elapsed)
            config = read(directory / 'configuration.json')
            start, end = config['started_at'], config['finished_at']
            wall = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
            assert wall > 0, name
            provenance.append(dict(variant=variant, task=task, task_zh=labels[(suite, task)],
                model=model, success=bool(result['success']), archived_fallback=archived,
                result_file=str(directory / 'benchmark_result.json'),
                elapsed_s=elapsed, display_elapsed=duration_label(elapsed),
                timing_field='benchmark_result.elapsed_s', started_at=start, finished_at=end,
                configuration_wall_s=wall,
                video_duration_s=result['frames'] / result['fps']))
        matrix.append(row)
        timings.append(times)
    assert len(matrix) == 10 and all(len(r) == 6 for r in matrix)
    total = sum(map(sum, matrix))
    summary = column_summary(provenance, variant)
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    ax.text(0, .99, f'{title} · 任务成功情况', fontproperties=font, fontsize=24,
            weight='bold', color='#172b45', transform=ax.transAxes, va='top')
    ax.text(0, .932, f'10 个任务 × 6 种模型组合   |   成功 {total}/60   |   ✓ 成功　× 未成功   |   时间：完整实验耗时（分:秒）',
            fontproperties=font, fontsize=12, color='#536175', transform=ax.transAxes)
    cells = []
    for task, result_row in enumerate(matrix):
        text = labels[(suite, task)]
        if task in (4, 6):
            text = text.replace('，', '，\n')
        values = [r'$\checkmark$ ' + duration_label(timings[task][col]) if value else r'$\times$'
                  for col, value in enumerate(result_row)]
        cells.append([f'{task}  {text}', *values])
    cells.append(['平均成功耗时（分:秒）', *[s['mean_display'] for s in summary]])
    cells.append(['成功率', *[s['rate_display'] for s in summary]])
    table = ax.table(cellText=cells, colLabels=['中文任务名', *[m for _, _, m in columns]],
        cellLoc='center', colWidths=[.46, .09, .09, .09, .09, .09, .09], bbox=[0, .125, 1, .755])
    table.auto_set_font_size(False)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#dbe3ec')
        cell.set_linewidth(.7)
        cell.PAD = .022
        if r == 0:
            cell.set_facecolor('#213c5b')
            cell.set_text_props(fontproperties=font, fontsize=13, color='white')
        elif r > 10:
            cell.set_facecolor('#e8eff8')
            cell.set_text_props(fontproperties=font, fontsize=12.5, color='#213c5b',
                                ha='left' if c == 0 else 'center')
        elif c == 0:
            cell.set_facecolor('#f3f6fa' if r % 2 else '#ffffff')
            cell.set_text_props(fontproperties=font, fontsize=12.5, color='#253447', ha='left')
        else:
            success = matrix[r - 1][c - 1]
            cell.set_facecolor('#e9f5ee' if success else '#fff0ef')
            cell.set_text_props(fontfamily='DejaVu Sans', fontsize=18 if success else 23,
                                color='#197344' if success else '#ba3e3e')
            if (r - 1, c - 1) in fallback:
                cell.set_facecolor('#fff1cc')
    ax.text(0, .075, 'GPT6：gpt-6-astra / low    Kimi：kimi-code/k3.1-pro / high',
            fontproperties=font, fontsize=11, color='#536175', transform=ax.transAxes)
    note = ('注：黄色两格为第 6 题 GPT 的归档旧结果；这两项重跑暂缓，不能视为修复后的重跑结果。'
            if fallback else '注：采用当前有效实验记录；已完成的第 4、6 题重跑已更新，其余任务保留现有结果。')
    ax.text(0, .033, note, fontproperties=font, fontsize=10.5, color='#765a21' if fallback else '#536175', transform=ax.transAxes)
    fig.subplots_adjust(left=.03, right=.97, top=.97, bottom=.03)
    fig.savefig(OUT / f'task_success_{variant}.png', dpi=180, facecolor='white')
    plt.close(fig)
    headers = ['中文任务名', *[m for _, _, m in columns]]
    if variant in remarks:
        headers.append('备注')
    markdown += [f'## {title}', '', '| ' + ' | '.join(headers) + ' |',
                 '|' + '---|' * len(headers)]
    for task, row in enumerate(matrix):
        notes, remark = annotations.get((variant, task), ([''] * 6, ''))
        values = [('✓ ' + duration_label(timings[task][col]) if value else '×')
                  + ('<br>' + comment if comment else '')
                  for col, (value, comment) in enumerate(zip(row, notes))]
        if variant in remarks:
            values.append(remark)
        markdown.append('| ' + f'{task} · {labels[(suite, task)]}' + ' | ' + ' | '.join(values) + ' |')
    for label, key in [('平均成功耗时（分:秒）', 'mean_display'), ('成功率', 'rate_display')]:
        values = [s[key] for s in summary]
        if variant in remarks:
            values.append('')
        markdown.append('| **' + label + '** | ' + ' | '.join(values) + ' |')
    markdown_note = ('注：第 6 题两项 GPT 使用归档旧结果；这两项重跑暂缓，不能视为修复后的重跑结果。'
                     if fallback else note)
    markdown += ['', f'成功合计：{total}/60。', '', markdown_note, '']
    print(variant, f'{total}/60', 'archived_fallbacks', len(fallback))

assert sum(r['archived_fallback'] for r in provenance) == 2
summaries = {variant: column_summary(provenance, None if variant == 'all' else variant)
             for variant in ('original', 'swap', 'all')}
markdown += ['## 两种场景合计（每列 20 项）', '',
    '| 指标 | ' + ' | '.join(s['model'] for s in summaries['all']) + ' |', '|' + '---|' * 7]
for label, key in [('平均成功耗时（分:秒）', 'mean_display'), ('成功率', 'rate_display')]:
    markdown.append('| ' + label + ' | ' + ' | '.join(s[key] for s in summaries['all']) + ' |')
markdown += ['', '平均耗时是本次仿真实验的端到端耗时，包含模型与网络等待，也含启动和记录开销，不能直接等同真机运动时间。不同组合成功的任务集合不同，比较整体平均数时需同时参考成功率。', '']
(OUT / 'task_success_tables.md').write_text('\n'.join(markdown), encoding='utf-8')
(OUT / 'task_success_sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'task_success_timing_summary.json').write_text(json.dumps(dict(
    elapsed_field='benchmark_result.elapsed_s', mean_scope='successful runs only',
    rate_denominator='10 per variant; 20 pooled; two deferred GPT cases use archived results',
    groups=summaries), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
overview = ['# 每列平均成功耗时与成功率', '',
    '时间均为 **分:秒**，包含模型思考、策略推理、网络等待、工具执行、启动与收尾；不是视频时长。', '',
    '平均成功耗时 = 成功记录 elapsed_s 的算术平均；失败不计入。成功率分母为每场景每列 10 项，合计每列 20 项。', '',
    '| 模型组合 | 原版平均成功耗时 | 原版成功率 | Swap 平均成功耗时 | Swap 成功率 | 合计平均成功耗时 | 合计成功率 |',
    '|---|---|---|---|---|---|---|']
for index, (_, _, model) in enumerate(columns):
    values = [model]
    for variant in ('original', 'swap', 'all'):
        record = summaries[variant][index]
        values += [record['mean_display'], record['rate_display']]
    overview.append('| ' + ' | '.join(values) + ' |')
overview += ['', 'Swap 第 6 题两项 GPT 重跑仍暂缓；表中沿用其归档旧失败结果。因此是现有记录汇总，不是所有 120 项均重新运行后的结果。', '',
    '不同组合成功的任务集合不同，整体成功项平均耗时不能单独证明后端更快。应结合成功率，并优先比较同场景、同任务、同语言模型且两种策略都成功的记录。', '',
    '计时来源：`benchmark_result.json` 的 `elapsed_s`。正常运行从本地策略桥接启动前计到实验子进程退出，包含完整交互；起止时间 `configuration.json.started_at/finished_at` 还包含末尾校验和清理。本表 120 条记录的两种口径差值均小于 6 秒。', '',
    '这是仿真实验的实际耗时，还包含仿真启动、观测保存等开销，不能直接等同真机运动时间。该字段也不能单独拆出纯语言模型思考耗时。', '']
(OUT / 'task_success_timing_summary.md').write_text('\n'.join(overview), encoding='utf-8')
