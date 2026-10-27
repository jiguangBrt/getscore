"""Move the latest 107+13 recordings into one real, link-free catalog."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_viewer.server import discover_runs
from run_viewer.parser import parse_run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def fingerprint(directory):
    checks = json.loads((directory / 'verification.json').read_text())['sha256']
    for name, expected in checks.items():
        paths = [directory / name] if (directory / name).is_file() else list((directory / 'trajectory').rglob(name))
        assert len(paths) == 1, (directory, name)
        digest = hashlib.sha256()
        with paths[0].open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, (directory, name, 'hash mismatch')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    outputs = ROOT / 'outputs'
    target = outputs / 'long120_1027'
    assert not target.exists(), target
    runs = discover_runs(outputs)
    assert len(runs) == 120
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    records = outputs / '_archive/records' / f'consolidate-long120_1027-{stamp}'
    archive = outputs / '_archive/batches' / f'before-long120_1027-{stamp}'
    entries = [dict(name=r['name'], old=r['path'], source=str(Path(r['path']).resolve()),
                    destination=str(target / r['name']), was_alias=Path(r['path']).is_symlink()) for r in runs]
    assert sum(e['was_alias'] for e in entries) == 13
    others = [p for p in outputs.iterdir() if p.name != '_archive' and p.name not in {e['name'] for e in entries}]
    moves = {Path(e['source']): Path(e['destination']) for e in entries}
    moves.update({p: archive / p.name for p in others})
    aliases = []
    for base, dirs, files in os.walk(outputs, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            if path.is_symlink():
                aliases.append(dict(path=str(path), target=str(path.resolve()), link=os.readlink(path)))
    manifest = dict(status='planned', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    latest_directory=str(target), archived_other_outputs=str(archive), entries=entries,
                    removed_aliases=aliases, other_moves=[dict(old=str(p), new=str(archive / p.name)) for p in others])
    if not args.apply:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    save(records / 'manifest.json', manifest)
    for entry in entries:
        entry['recorded_hashes'] = fingerprint(Path(entry['source']))
    manifest['status'] = 'preflight_verified'
    save(records / 'manifest.json', manifest)
    # Exact aliases precede directory-prefix replacements, retaining historical
    # batch references to the original 1000-step recordings where appropriate.
    replacements = {str(old): str(new) for old, new in moves.items()}
    def replace_path(text):
        for old in sorted(replacements, key=len, reverse=True):
            if text == old or text.startswith(old + '/'):
                return replacements[old] + text[len(old):]
        return text
    for alias in aliases:
        replacements[alias['path']] = replace_path(alias['target'])
    target.mkdir()
    for entry in entries:
        Path(entry['source']).rename(entry['destination'])
    # Remove only recorded directory aliases; no experiment contents are deleted.
    for alias in aliases:
        path = Path(alias['path'])
        assert path.is_symlink(), path
        path.unlink()
    archive.mkdir(parents=True)
    for path in others:
        path.rename(archive / path.name)
    manifest['status'] = 'moved'
    save(records / 'manifest.json', manifest)

    def remap(value):
        if isinstance(value, dict): return {key: remap(part) for key, part in value.items()}
        if isinstance(value, list): return [remap(part) for part in value]
        if isinstance(value, str): return replace_path(value)
        return value
    # Rewrite derived references, keeping immutable execution evidence intact.
    metadata_names = {'configuration.json', 'benchmark_result.json', 'artifact-completeness.json',
                      'recovery.json', 'full-video-verification.json'}
    files = [path / name for path in target.iterdir() for name in metadata_names if (path / name).is_file()]
    for moved in archive.iterdir():
        if moved.is_dir():
            files += [p for p in moved.glob('*.json') if p.name not in {'wall-time-recovery.json'}]
    # Archived source configurations still describe their original execution;
    # only top-level derived batch references are adjusted above.
    manifest['metadata_updates'] = []
    for path in files:
        before = path.read_text()
        data = json.loads(before)
        after = remap(data)
        if data != after:
            backup = records / 'metadata-before' / path.relative_to(outputs)
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(before)
            save(path, after)
            manifest['metadata_updates'].append(str(path))
    for moved in archive.iterdir():
        if not moved.is_dir(): continue
        for name in ['report.md', 'report.txt', 'comparison.csv', 'success-matrices.md']:
            path = moved / name
            if not path.is_file(): continue
            before = path.read_text(); after = before
            # Substitute each path once, avoiding recursive replacements.
            import re
            pattern = re.compile('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
            after = pattern.sub(lambda match: replacements[match.group()], before)
            if before != after:
                backup = records / 'metadata-before' / path.relative_to(outputs)
                backup.parent.mkdir(parents=True, exist_ok=True); backup.write_text(before)
                path.write_text(after)
    historical = archive / 'long-original-swap-benchmark-20261023'
    save(historical / 'case-locations.json', [dict(previous_path=a['path'], recording=replace_path(a['target']))
         for a in aliases if '/long-original-swap-benchmark-20261023/cases/' in a['path']])
    rows = []
    checks = []
    for entry in entries:
        directory = Path(entry['destination'])
        assert not directory.is_symlink()
        assert fingerprint(directory) == entry['recorded_hashes']
        row = json.loads((directory / 'benchmark_result.json').read_text())
        report = parse_run(directory)
        assert report['warnings'] == [], (directory, report['warnings'])
        assert report['status'] == ('success' if row['success'] else 'failure')
        assert report['metrics']['tool_calls'] > 0
        assert report['video'] and report['observations']
        assert sum(e.get('video_range', {}).get('frames', 0) for e in report['events']) == row['frames']
        assert row['directory'] == str(directory)
        assert all(Path(row[k]).is_file() if k in {'video', 'transcript'} else Path(row[k]).is_dir()
                   for k in ['directory', 'video', 'transcript', 'trajectory'])
        rows.append(row)
        checks.append(dict(name=entry['name'], success=row['success'], warnings=[], tool_calls=report['metrics']['tool_calls']))
    assert len(discover_runs(target)) == 120
    assert sum(bool(row['success']) for row in rows) == 60
    assert not [p for p in target.rglob('*') if p.is_symlink()]
    assert {p.name for p in outputs.iterdir()} == {'long120_1027', '_archive'}
    groups = defaultdict(lambda: dict(successes=0, total=0, errors=0))
    for row in rows:
        group = groups[f'{row["variant"]}-{row["planner"]}-{row["policy"]}']
        group['total'] += 1; group['successes'] += bool(row['success']); group['errors'] += bool(row['error'])
    for group in groups.values(): group['success_rate'] = group['successes'] / group['total']
    save(target / 'summary.json', dict(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
         completed=120, successes=60, groups=groups, results=rows,
         composition=dict(unchanged_original_runs=107, selected_new_runs=13, resumed=3, restarted=10),
         budget_note='107 original runs retain their original budgets; 13 selected runs use a total budget of 2000 actions.'))
    fields = ['name', 'variant', 'task', 'planner', 'policy', 'success', 'actions', 'total_episode_actions',
              'execution_mode', 'directory', 'video', 'transcript', 'trajectory']
    with (target / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    matrix = archive / 'selected-original-2000step-20261026/success-matrices.md'
    if matrix.exists(): (target / 'success-matrices.md').write_text(matrix.read_text())
    (target / 'README.md').write_text(
        '# 最新 120 项实验\n\n本目录包含 120 个真实实验文件夹，没有软链接：原有 107 项，加上本次更新的 13 项（3 项续跑、10 项重跑）。\n\n'
        '原版与交换场景各 60 项，当前共 60 项成功。107 项保留原步数预算；更新的 13 项总预算为 2000 步。'
        '续跑项的轨迹、日志与 episode.mp4 对应追加段，episode-full.mp4 为原 1000 步加追加段的连续视频。\n\n'
        f'run_viewer 默认打开本目录。统计见 summary.json、results.csv；成功矩阵见 success-matrices.md。'
        f'历史批次与旧结果在 ../_archive/，迁移清单：{records / "manifest.json"}。\n')
    manifest.update(status='verified', real_experiment_directories=120, latest_symlinks=0,
                    successes=60, all_recorded_hashes_preserved=True, all_viewer_warnings_empty=True, checks=checks)
    save(records / 'manifest.json', manifest)
    save(target / 'consolidation.json', {key: manifest[key] for key in ['status', 'real_experiment_directories',
         'latest_symlinks', 'successes', 'all_recorded_hashes_preserved', 'all_viewer_warnings_empty']}
         | dict(manifest=str(records / 'manifest.json'), historical_outputs=str(archive)))
    print(json.dumps(dict(status='verified', latest=str(target), experiments=120, symlinks=0, archive=str(archive)), ensure_ascii=False))


if __name__ == '__main__': main()
