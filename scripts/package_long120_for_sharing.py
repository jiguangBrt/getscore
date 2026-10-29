"""Bundle the 120 canonical recordings and the standard run_viewer for sharing."""
from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/long120_1027'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    directories = sorted(p for p in DATA.iterdir() if p.is_dir() and p.name.startswith('long-'))
    assert len(directories) == 120
    entries, canonical = [], []
    for directory in directories:
        row = read(directory / 'benchmark_result.json')
        trajectory = list(directory.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*/episode.json'))
        assert len(trajectory) == 1
        assert not (directory / 'tool_calls-full.jsonl').exists(), directory
        transcripts = list(directory.glob('transcript_*.json'))
        assert len(transcripts) == 1
        entry = dict(name=directory.name, root=directory.name, variant=row['variant'], task=row['task'],
                     planner=row['planner'], policy=row['policy'], success=row['success'],
                     video=f'{directory.name}/episode.mp4', transcript=f'{directory.name}/{transcripts[0].name}',
                     states=f'{directory.name}/states.json',
                     trajectory=trajectory[0].parent.relative_to(DATA).as_posix(),
                     actions=row.get('total_episode_actions', row['actions']))
        if (directory / 'recording-provenance.json').is_file():
            canonical.append(directory.name)
            assert row['actions'] == row['frames'] == entry['actions']
            entry['recording_scope'] = 'full_episode'
        entries.append(entry)
    assert len(canonical) == 3
    # This package needs only Python's standard library to view the records.
    shutil.copytree(ROOT / 'run_viewer', DATA / 'run_viewer', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (DATA / 'run-viewer.py').write_text(
        '"""Open these saved experiments with the bundled run_viewer."""\n'
        'from pathlib import Path\nimport sys\nfrom run_viewer.server import main\n\n'
        'if "--default-path" not in sys.argv:\n'
        '    sys.argv += ["--default-path", str(Path(__file__).resolve().parent)]\n'
        'main()\n')
    stats = read(DATA / 'statistics/tool_counts.json')
    for row in stats['per_run']:
        if row['name'] in canonical:
            row['execution_mode'] = 'single_episode'
    save(DATA / 'statistics/tool_counts.json', stats)
    save(DATA / 'sharing-manifest.json', dict(format_version=2,
         created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), runs=120,
         language_model_runs=80, function_types=15, model_tool_calls=stats['total'],
         paths_relative_to='this_directory', external_archive_required=False,
         viewer='run_viewer', launch='python3 run-viewer.py', canonical_recordings=canonical, entries=entries))
    rows = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>120 次实验</title>',
            '<style>body{max-width:1200px;margin:30px auto;font:16px/1.7 system-ui}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}a{color:#1462a6}</style>',
            '<h1>120 次实验 · 标准 run_viewer 记录</h1>',
            '<p>所有实验均直接读取原有文件名。下载整个目录后运行 <code>python3 run-viewer.py</code>，打开终端显示的网址，即可用 run_viewer 查看完整历史。3 次原续跑已整理成整轮记录。</p>',
            '<p><a href="statistics/functions-and-counts.html">全部函数说明与统计</a> · <a href="sharing-manifest.json">相对路径清单</a></p>',
            '<table><tr><th>实验</th><th>结果</th><th>动作数</th><th>原始记录</th></tr>']
    for entry in entries:
        links = ' · '.join(f'<a href="{quote(entry[field])}">{label}</a>' for field, label in [('video', '视频'), ('transcript', '对话'), ('states', '状态')])
        rows.append(f'<tr><td>{html.escape(entry["name"])}</td><td>{"成功" if entry["success"] else "未成功"}</td><td>{entry["actions"]}</td><td>{links}</td></tr>')
    rows.append('</table></html>')
    (DATA / 'SHARE.html').write_text('\n'.join(rows))
    description = ('# 120 次实验：完整历史记录\n\n'
        '本目录可整体上传、下载。所有实验都直接使用标准 `kimi_tool_calls.jsonl` / Codex 事件流、`states.json`、`command_trace.jsonl`、原始轨迹和 `episode.mp4`。'
        '3 次原续跑已统一观测编号、图像文件、工具参数、轨迹索引和动作视频范围，按一轮连续实验读取；不需要另选前后段或另选 full 文件。\n\n'
        '## 用 run_viewer 打开\n\n在下载后的本目录运行：\n\n```bash\npython3 run-viewer.py\n```\n\n'
        '然后打开终端显示的本机网址。查看器仅使用 Python 标准库，无需安装模型或联网。也可以使用已有 run_viewer，直接选择本目录。\n\n'
        '80 次语言模型实验：15 种函数、4208 次调用。逐任务和逐实验统计位于 `statistics/`。'
        '原始分段证据在原机器上另外归档，当前文件夹已包含完整历史，无需访问该归档。'
        '合并记录中的 `recording-provenance.json` 只用于追溯整理来源。\n\n'
        '运行 `python3 verify-sharing.py` 可重新校验文件和查看器解析结果。\n')
    (DATA / 'README-sharing.md').write_text(description)
    (DATA / 'README.md').write_text(description + '\n当前任务成功 63/120。保留结果的步数预算并不完全相同；部分重跑存在初始家具布局变化，因此不能把全部组合视为严格固定布局的配对实验。\n')
    print('Packaged 120 standard recordings and the standalone run_viewer')


if __name__ == '__main__':
    main()
