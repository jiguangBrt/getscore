"""Render the latest original/swap 10-by-6 results, with explicit archived fallbacks."""
from __future__ import annotations

import json
from pathlib import Path

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
                notes.append(cell[1:].removeprefix('<br>').strip())
            annotations[(variant, task)] = (notes, cells[7] if len(cells) == 8 else '')
    return annotations, remarks


annotations, remarks = existing_annotations(OUT / 'task_success_tables.md')
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
provenance, markdown = [], ['# 任务成功情况', '',
    '每张表为 10 个任务 × 6 种模型组合。✓ 成功，× 未成功；以环境原生成功标记为准。', '',
    'GPT6 = gpt-6-astra / low；Kimi = kimi-code/k3.1-pro / high。本次 4/6 题重跑结果已更新；其余保留现有记录，步数预算并不完全相同。', '']

for variant, suite, title in [('original', 'libero_10', '原版场景'), ('swap', 'libero_10_swap', 'Swap 场景')]:
    matrix, fallback = [], set()
    for task in range(10):
        row = []
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
            provenance.append(dict(variant=variant, task=task, task_zh=labels[(suite, task)],
                model=model, success=bool(result['success']), archived_fallback=archived,
                result_file=str(directory / 'benchmark_result.json')))
        matrix.append(row)
    assert len(matrix) == 10 and all(len(r) == 6 for r in matrix)
    total = sum(map(sum, matrix))
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    ax.text(0, .99, f'{title} · 任务成功情况', fontproperties=font, fontsize=24,
            weight='bold', color='#172b45', transform=ax.transAxes, va='top')
    ax.text(0, .932, f'10 个任务 × 6 种模型组合   |   成功 {total}/60   |   ✓ 成功　× 未成功',
            fontproperties=font, fontsize=12, color='#536175', transform=ax.transAxes)
    cells = []
    for task, result_row in enumerate(matrix):
        text = labels[(suite, task)]
        if task in (4, 6):
            text = text.replace('，', '，\n')
        cells.append([f'{task}  {text}', *[r'$\checkmark$' if v else r'$\times$' for v in result_row]])
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
        elif c == 0:
            cell.set_facecolor('#f3f6fa' if r % 2 else '#ffffff')
            cell.set_text_props(fontproperties=font, fontsize=12.5, color='#253447', ha='left')
        else:
            success = matrix[r - 1][c - 1]
            cell.set_facecolor('#e9f5ee' if success else '#fff0ef')
            cell.set_text_props(fontfamily='DejaVu Sans', fontsize=23,
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
        values = [('✓' if value else '×') + ('<br>' + comment if comment else '')
                  for value, comment in zip(row, notes)]
        if variant in remarks:
            values.append(remark)
        markdown.append('| ' + f'{task} · {labels[(suite, task)]}' + ' | ' + ' | '.join(values) + ' |')
    markdown_note = ('注：第 6 题两项 GPT 使用归档旧结果；这两项重跑暂缓，不能视为修复后的重跑结果。'
                     if fallback else note)
    markdown += ['', f'成功合计：{total}/60。', '', markdown_note, '']
    print(variant, f'{total}/60', 'archived_fallbacks', len(fallback))

assert sum(r['archived_fallback'] for r in provenance) == 2
(OUT / 'task_success_tables.md').write_text('\n'.join(markdown), encoding='utf-8')
(OUT / 'task_success_sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
