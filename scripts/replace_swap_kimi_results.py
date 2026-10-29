"""Archive the selected old swap recordings and promote their audited reruns."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.consolidate_long120_1027 import fingerprint, save
from scripts.package_selected_long_2000 import prepare_case
from run_viewer.parser import parse_run
from run_viewer.server import discover_runs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outputs = ROOT / 'outputs'
    target = outputs / 'long120_1027'
    batch = outputs / 'swap-kimi-2000step-20261027'
    selection = json.loads((batch / 'selection.json').read_text())
    assert json.loads((batch / 'status.json').read_text())['status'] == 'complete'
    names = sorted(Path(item['source_run']).name for item in selection['cases'])
    assert len(names) == len(set(names)) == 13
    assert len(discover_runs(target)) == 120
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    archive = outputs / '_archive/batches' / f'swap-before-2000step-{stamp}'
    records = outputs / '_archive/records' / f'replace-swap-2000step-{stamp}'
    records.mkdir(parents=True)
    # Snapshot all derived metadata before any packaging or movement.
    for base in [target, batch]:
        for path in base.iterdir():
            if path.is_file() and path.suffix in {'.json', '.md', '.csv'}:
                dest = records / 'metadata-before' / path.relative_to(outputs)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    untouched = {p.name: digest(p / 'benchmark_result.json') for p in target.iterdir()
                 if p.is_dir() and (p / 'benchmark_result.json').exists() and p.name not in names}
    assert len(untouched) == 107
    entries = []
    for item in selection['cases']:
        name = Path(item['source_run']).name
        old, new = target / name, batch / 'cases' / name
        assert old.is_dir() and new.is_dir() and not old.is_symlink() and not new.is_symlink()
        assert digest(old / 'benchmark_result.json') == item['source_result_sha256']
        row = json.loads((new / 'benchmark_result.json').read_text())
        config = json.loads((new / 'configuration.json').read_text())
        assert not json.loads((old / 'benchmark_result.json').read_text())['success']
        assert config['model'] == 'kimi-code/k3.1-pro' and config['max_episode_steps'] == 2000
        assert row['variant'] == 'swap' and row['planner'] == 'kimi' and row['error'] is None
        entries.append(dict(name=name, source=str(new), destination=str(old), archived_original=str(archive / name),
                            original_result_sha256=item['source_result_sha256'],
                            original_recorded_hashes=fingerprint(old), new_recorded_hashes=fingerprint(new)))
        for filename in ['configuration.json', 'benchmark_result.json']:
            dest = records / 'metadata-before' / new.relative_to(outputs) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new / filename, dest)
    manifest = dict(status='preflight_verified', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    latest_directory=str(target), archived_originals=str(archive), entries=entries,
                    untouched_result_hashes=untouched,
                    archive_note='Original directories are preserved byte-for-byte; embedded historical absolute paths remain execution provenance.')
    save(records / 'manifest.json', manifest)
    # Validate the full viewer layout before replacing any canonical recording.
    for entry in entries:
        source = Path(entry['source'])
        row = json.loads((source / 'benchmark_result.json').read_text())
        row['trajectory_scope'] = 'full_episode'
        save(source / 'benchmark_result.json', row)
        config = json.loads((source / 'configuration.json').read_text())
        # The legacy packager predates migration; retain the recorded endpoint.
        command = config['command']
        prepare_case(source, batch)
        updated = json.loads((source / 'configuration.json').read_text())
        updated['config']['simulation']['sam3_endpoint'] = command[command.index('--sam3-endpoint') + 1]
        save(source / 'configuration.json', updated)
        print('PACKAGED', source.name, flush=True)
    archive.mkdir(parents=True)
    for entry in entries:
        Path(entry['destination']).rename(entry['archived_original'])
        Path(entry['source']).rename(entry['destination'])
    manifest['status'] = 'moved'
    save(records / 'manifest.json', manifest)

    replacements = {entry['source']: entry['destination'] for entry in entries}
    replacements.update({entry['destination']: entry['archived_original'] for entry in entries})
    def replace_path(value):
        for before in sorted(replacements, key=len, reverse=True):
            if value == before or value.startswith(before + '/'):
                return replacements[before] + value[len(before):]
        return value
    def remap(value):
        if isinstance(value, dict): return {k: remap(v) for k, v in value.items()}
        if isinstance(value, list): return [remap(v) for v in value]
        return replace_path(value) if isinstance(value, str) else value
    for entry in entries:
        directory = Path(entry['destination'])
        for name in ['configuration.json', 'benchmark_result.json', 'artifact-completeness.json']:
            path = directory / name
            save(path, remap(json.loads(path.read_text())))
        assert fingerprint(directory) == entry['new_recorded_hashes']
        old = Path(entry['archived_original'])
        assert fingerprint(old) == entry['original_recorded_hashes']
        assert digest(old / 'benchmark_result.json') == entry['original_result_sha256']
    for filename in ['selection.json', 'summary.json', 'status.json']:
        path = batch / filename
        data = remap(json.loads(path.read_text()))
        data['promoted_to'] = str(target)
        data['replacement_manifest'] = str(records / 'manifest.json')
        save(path, data)
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
    report_path = batch / 'report.md'
    report = pattern.sub(lambda match: replacements[match.group()], report_path.read_text())
    report = report.replace('原始 long120_1027 结果保持原样。',
                            f'这 13 项已替换进入 `{target}`；被替换的原记录完整保存在 `{archive}`。')
    report_path.write_text(report)
    save(batch / 'case-locations.json', dict(promoted_to=str(target), archived_originals=str(archive), entries=entries))
    (batch / 'cases/README.md').write_text(f'13 项已移入 {target}。对应旧实验在 {archive}。详见 ../case-locations.json。\n')
    checks, rows = [], []
    for run in discover_runs(target):
        directory = Path(run['path'])
        row = json.loads((directory / 'benchmark_result.json').read_text())
        report = parse_run(directory)
        assert report['warnings'] == [], (directory, report['warnings'])
        assert report['status'] == ('success' if row['success'] else 'failure')
        assert report['metrics']['tool_calls'] > 0 and report['video'] and report['observations']
        assert sum(e.get('video_range', {}).get('frames', 0) for e in report['events']) == row['frames']
        assert row['directory'] == str(directory)
        for key in ['video', 'transcript', 'trajectory']:
            assert Path(row[key]).exists(), (directory, key)
        if directory.name in untouched:
            assert digest(directory / 'benchmark_result.json') == untouched[directory.name]
        rows.append(row)
        checks.append(dict(name=directory.name, success=row['success'], warnings=[],
                           tool_calls=report['metrics']['tool_calls'], frames=row['frames']))
    assert len(rows) == 120 and sum(bool(r['success']) for r in rows) == 63
    assert sum(r['success'] for r in rows if r['variant'] == 'swap') == 18
    assert len(list(archive.iterdir())) == 13
    assert not any(p.is_symlink() for p in target.rglob('*'))
    groups = defaultdict(lambda: dict(successes=0, total=0, errors=0))
    for row in rows:
        group = groups[f'{row["variant"]}-{row["planner"]}-{row["policy"]}']
        group['total'] += 1
        group['successes'] += bool(row['success'])
        group['errors'] += bool(row['error'])
    for group in groups.values(): group['success_rate'] = group['successes'] / group['total']
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    composition = dict(unchanged_original_runs=94, selected_new_runs=26, resumed=3, restarted=10, fresh_swap_reruns=13)
    note = '94 runs retain their original budgets; 13 selected original-scene runs and 13 selected swap Kimi reruns use a 2000-action total budget.'
    initialization_note = '8 swap reruns differ in initial RGB due to existing fixture-layout randomization; nominal task/seed/init-file and robot state match. Not strictly fixed-layout paired trials.'
    save(target / 'summary.json', dict(updated_at=now, completed=120, successes=63, groups=groups, results=rows,
         composition=composition, budget_note=note, swap_initialization_note=initialization_note,
         replacement_manifest=str(records / 'manifest.json')))
    fields = ['name', 'variant', 'task', 'planner', 'policy', 'success', 'actions', 'total_episode_actions',
              'execution_mode', 'directory', 'video', 'transcript', 'trajectory']
    with (target / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    prior_matrix = (records / 'metadata-before/long120_1027/success-matrices.md').read_text()
    labels = re.findall(r'^\| (\d+ · [^|]+) \|', prior_matrix, re.M)[:10]
    assert len(labels) == 10
    columns = [('none', 'pi05'), ('none', 'gawm'), ('gpt6', 'pi05'), ('gpt6', 'gawm'), ('kimi', 'pi05'), ('kimi', 'gawm')]
    matrix = '原版场景此前更新 13 项；交换场景本次替换 Kimi 失败项 13 项。其余保留原结果。✅ 成功，❌ 失败。步数上限混合，并非全部以 2000 步重测。\n\n'
    for variant, title in [('original', '原版场景'), ('swap', '交换场景（Kimi 13 项已替换为 2000 步重跑结果）')]:
        matrix += title + '\n\n| 任务 | 纯 π0.5 | 纯 GAWM | GPT6+π0.5 | GPT6+GAWM | Kimi+π0.5 | Kimi+GAWM |\n|---|---|---|---|---|---|---|\n'
        lookup = {(r['task'], r['planner'], r['policy']): r for r in rows if r['variant'] == variant}
        assert len(lookup) == 60
        for task in range(10):
            matrix += '| ' + labels[task].strip() + ' | ' + ' | '.join('✅' if lookup[task, p, a]['success'] else '❌' for p, a in columns) + ' |\n'
        matrix += '| **成功 / 全部** | ' + ' | '.join(f'**{sum(lookup[t, p, a]["success"] for t in range(10))}/10**' for p, a in columns) + ' |\n\n'
    matrix += f'本次 8 项重跑存在初始家具布局差异；任务、seed、初态文件及机器人初态已核对一致，不能视为严格固定布局的配对实验。详见 [重跑报告]({batch / "report.md"})。\n'
    (target / 'success-matrices.md').write_text(matrix)
    (target / 'README.md').write_text(
        '# 最新 120 项实验\n\n本目录包含 120 个真实实验文件夹，无软链接。原版场景此前更新 13 项（3 项续跑、10 项重跑）；交换场景本次更新 13 项 Kimi 失败实验（模型 kimi-code/k3.1-pro，从头重跑）。其余 94 项保留原结果。\n\n'
        '原版场景 45/60、交换场景 18/60，合计 63/120 成功。交换场景 Kimi+π0.5 与 Kimi+GAWM 均为 5/10。26 项更新实验总预算为 2000 步，其余保留原预算。续跑项 episode.mp4 和轨迹对应追加段，episode-full.mp4 是完整连续视频。\n\n'
        f'本次被替换的 13 项旧实验完整归档于 [{archive.name}]({archive})；[替换清单]({records / "manifest.json"})。\n\n'
        f'本次 8 项初始家具布局与旧记录不同；名义任务、seed、初态文件及机器人初态相同，不能视为严格固定布局的配对实验。详情见 [重跑报告]({batch / "report.md"}) 和 [初始化审计]({batch / "final-audit/initialization-note.md"})。\n\n'
        'run_viewer 默认打开本目录。统计见 summary.json、results.csv；成功矩阵见 success-matrices.md。\n')
    manifest.update(status='verified', verified_at=now, real_experiment_directories=120, replaced=13,
                    archived_originals_count=13, successes=63, all_recorded_hashes_preserved=True,
                    untouched_experiments=107, all_viewer_warnings_empty=True, latest_symlinks=0, checks=checks)
    save(records / 'manifest.json', manifest)
    save(target / 'consolidation.json', {k: manifest[k] for k in ['status', 'real_experiment_directories',
         'latest_symlinks', 'successes', 'all_recorded_hashes_preserved', 'all_viewer_warnings_empty']}
         | dict(manifest=str(records / 'manifest.json'), historical_outputs=str(outputs / '_archive'),
                replaced_swap_runs=13, archived_originals=str(archive), viewer_parse_verified=120,
                previous_consolidation=str(records / 'metadata-before/long120_1027/consolidation.json')))
    print(json.dumps(dict(status='verified', replaced=13, archived=str(archive), successes=63,
                          experiments=120, manifest=str(records / 'manifest.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
