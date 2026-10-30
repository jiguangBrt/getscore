"""Archive and freshly replace the 16 task-4/6 planner runs in the canonical catalog."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
from scripts import run_long_benchmark as runner
from scripts.consolidate_long120_1027 import fingerprint
from scripts.package_selected_long_2000 import prepare_case
from scripts.run_swap_kimi_2000 import verify_initialization


def read(path):
    return json.loads(path.read_text())


def prepare(batch, target, archive):
    selection_path = batch / 'selection.json'
    if selection_path.exists():
        selection = read(selection_path)
        assert selection['target'] == str(target) and selection['archive'] == str(archive)
        return selection
    assert not archive.exists(), archive
    items = []
    for variant in ('original', 'swap'):
        for task in (4, 6):
            for planner in ('gpt6', 'kimi'):
                for policy in ('pi05', 'gawm'):
                    name = f'long-{variant}-{task}-{planner}-{policy}'
                    source = target / name
                    config, row = read(source / 'configuration.json'), read(source / 'benchmark_result.json')
                    assert source.is_dir() and not source.is_symlink()
                    assert (config['variant'], config['task'], config['planner'], config['policy']) == (variant, task, planner, policy)
                    assert config['seed'] == config['initial_state_index'] == 0
                    assert config['model'] == {'gpt6': 'gpt-6-astra', 'kimi': 'kimi-code/k3.1-pro'}[planner]
                    assert config['reasoning_effort'] == {'gpt6': 'low', 'kimi': 'high'}[planner]
                    for key in ('bddl_files', 'init_files'):
                        assert runner.sha(Path(config[key]['path'])) == config[key]['sha256']
                    items.append(dict(name=name, policy=policy,
                        case={k: config[k] for k in ('variant', 'suite', 'task', 'name', 'policy_prompt', 'planner')},
                        archived_source=str(archive / name), old_result=row,
                        recorded_hashes=fingerprint(source)))
    names = {item['name'] for item in items}
    untouched = {p.name: runner.sha(p / 'benchmark_result.json') for p in target.glob('long-*')
                 if p.is_dir() and p.name not in names}
    assert len(items) == 16 and len(untouched) == 104
    selection = dict(created_at=runner.now(), target=str(target), archive=str(archive), expected=16,
        mode='fresh', max_episode_steps=2000, max_turns=160,
        planner_timeout_s=dict(gpt6=2400, kimi=7200), seed=0, initial_state_index=0,
        runtime_commit=subprocess.check_output(['git', '-C', str(ROOT / 'RPent'), 'rev-parse', 'HEAD'], text=True).strip(),
        cases=items, untouched_result_hashes=untouched)
    runner.save(selection_path, selection)
    return selection


def archive_originals(batch, selection):
    target, archive = Path(selection['target']), Path(selection['archive'])
    archive.mkdir(parents=True, exist_ok=True)
    metadata = archive / 'catalog-metadata-before'
    if not metadata.exists():
        metadata.mkdir()
        for path in target.iterdir():
            if path.is_file():
                shutil.copy2(path, metadata / path.name)
        shutil.copytree(target / 'statistics', metadata / 'statistics')
    for item in selection['cases']:
        old, dest = target / item['name'], Path(item['archived_source'])
        if not dest.exists():
            assert fingerprint(old) == item['recorded_hashes']
            old.rename(dest)
        assert fingerprint(dest) == item['recorded_hashes']
    runner.save(batch / 'archive-verification.json', dict(status='verified', count=16,
        recorded_hashes_preserved=True, archive=str(archive), verified_at=runner.now()))


def update_report(batch, selection, completed, errors, status):
    rows = sorted(completed.values(), key=lambda r: r['name'])
    by_planner = {p: sum(r['planner'] == p for r in rows) for p in ('kimi', 'gpt6')}
    runner.save(batch / 'status.json', dict(status=status, expected=16, completed=len(rows),
        completed_by_planner=by_planner, deferred_planners=selection.get('deferred_planners', []),
        successes=sum(bool(r['success']) for r in rows), errors=errors, updated_at=runner.now()))
    lines = ['# 左右方向修复后：4 / 6 的 16 项重跑', '',
        f'完成 {len(rows)}/16，成功 {sum(bool(r["success"]) for r in rows)}。', '',
        '原版/swap × 4/6 × GPT/Kimi × π0.5/GAWM；从头运行，seed 0、初态索引 0，最多 2000 原生动作。',
        'GPT：gpt-6-astra / low，2400 秒；Kimi：kimi-code/k3.1-pro / high，7200 秒；max_turns 160。',
        '成功仅以原生环境 terminated 为准。任务失败不自动重试。',
        f'旧记录完整归档在 {selection["archive"]}；其余 104 项保留。', '',
        '| 实验 | 旧成功 | 新成功 | 动作数 | 初始画面逐像素一致 |', '|---|---|---|---:|---|']
    originals = {item['name']: item['old_result'] for item in selection['cases']}
    for row in rows:
        lines.append(f'| {row["name"]} | {originals[row["name"]]["success"]} | {row["success"]} | {row["actions"]} | {row["initial_observation_exact"]} |')
    if errors:
        lines += ['', '基础设施或记录校验异常：', *[f'- {k}: {v}' for k, v in sorted(errors.items())]]
    lines += ['', '家具初始化可能随机变化。任务、seed、初始化文件和机器人初态均与归档核对；画面是否一致单独记录。']
    (batch / 'report.md').write_text('\n'.join(lines) + '\n')


def adopt_running(output, item, record):
    """Keep an existing episode alive while transferring its supervisor."""
    pid = record['runner_pid']
    wait_code = """
import os, select, sys
from pathlib import Path
pid, expected, timeout = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
try:
    fd = os.pidfd_open(pid)
except ProcessLookupError:
    sys.exit(0)
try:
    try:
        actual = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        sys.exit(0)
    assert actual == expected, 'Runner PID was reused'
    sys.exit(0 if select.select([fd], [], [], timeout)[0] else 124)
finally:
    os.close(fd)
"""
    config = read(output / 'configuration.json')
    elapsed = (datetime.now().astimezone() - datetime.fromisoformat(record['started_at'])).total_seconds()
    remaining = max(1, config['planner_timeout_s'] + 600 - elapsed)
    print(runner.now(), 'ADOPT', output.name, 'pid=' + str(pid), flush=True)
    wait = subprocess.run(['/usr/bin/python3', '-c', wait_code, str(pid),
                           record['runner_start_ticks'], str(remaining)], check=False)
    try:
        if wait.returncode:
            raise RuntimeError(f'Adopted runner wait failed: {wait.returncode}')
        elapsed = (datetime.now().astimezone() - datetime.fromisoformat(record['started_at'])).total_seconds()
        row = runner.verify_case(output, item['case'], item['policy'], elapsed, returncode=None)
        config.update(run_status='finished', task_success=row['success'], error=row.get('error'),
            exit_code=None, finished_at=runner.now(), supervisor_takeover=record,
            exit_code_note='Original runner was adopted after its supervisor was replaced; OS exit code is unavailable. Native trajectory and transcript were verified.')
        runner.save(output / 'configuration.json', config)
    finally:
        for kind in ('runner', 'bridge'):
            child = record[kind + '_pid']
            try:
                ticks = Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()[19]
                if ticks == record[kind + '_start_ticks']:
                    os.killpg(child, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass


def run_one(batch, selection, item, env, inference_config, adopted=None):
    target = Path(selection['target'])
    output, original = target / item['name'], Path(item['archived_source'])
    if not (output / 'rerun-verification.json').exists():
        if adopted and not (output / 'benchmark_result.json').exists():
            adopt_running(output, item, adopted)
        args = SimpleNamespace(max_steps=2000, max_turns=160,
            timeout=selection['planner_timeout_s'][item['case']['planner']],
            inference_config=inference_config, output_root=target)
        runner.run_case(batch, item['case'], item['policy'], env, args)
        config = read(output / 'configuration.json')
        if config.get('error'):
            raise RuntimeError(config['error'])
        row = read(output / 'benchmark_result.json')
        assert not row.get('error') and 0 < row['actions'] <= 2000, row
        assert not row['truncated'] or row['success'] or row['actions'] == 2000
        old_row = item['old_result']
        archived_row = {**old_row, 'trajectory': str(original / Path(old_row['trajectory']).relative_to(old_row['directory']))}
        audit = verify_initialization(original, output, archived_row, row)
        if item['case']['planner'] == 'kimi':
            settings = read(output / 'kimi_session.json')['confirmed_settings']
            assert settings['model']['currentValue'] == 'kimi-code/k3.1-pro'
            assert settings['thinking']['currentValue'] == 'high'
        if item['policy'] == 'gawm':
            assert config['policy_status']['metadata']['checkpoint_sha256'] == 'f698e76ae129ccfbe8349418acd79044f2a1ca6366d098546afb477fd79e7e13'
        row.update(execution_mode='fresh', source_run=str(original), resumed_from_step=0,
            total_episode_actions=row['actions'], target_total_steps=2000, trajectory_scope='full_episode',
            initial_observation_exact=audit['initial_observation_exact'], runtime_commit=selection['runtime_commit'])
        runner.save(output / 'benchmark_result.json', row)
        prepare_case(output, batch)
        # The legacy viewer packager carries a historical SAM3 endpoint default.
        config = read(output / 'configuration.json')
        command = config['command']
        config['config']['simulation']['sam3_endpoint'] = command[command.index('--sam3-endpoint') + 1]
        config['runtime_commit'] = selection['runtime_commit']
        runner.save(output / 'configuration.json', config)
        runner.save(output / 'rerun-verification.json', dict(status='verified',
            max_episode_steps=2000, execution_mode='fresh', initialized_against=str(original),
            runtime_commit=selection['runtime_commit'], checked_at=runner.now()))
    fingerprint(output)
    return read(output / 'benchmark_result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=ROOT / 'outputs/long120_1027')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inference-config', type=Path, default=ROOT / 'configs/liberopro-228.toml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--adopt-running', type=Path)
    parser.add_argument('--defer-gpt', action='store_true', help='Finish Kimi only and leave pending GPT runs untouched')
    parser.add_argument('--finalize-only', action='store_true', help='Refresh metadata from verified recordings without running models')
    args = parser.parse_args()
    batch, target, archive = args.batch.resolve(), args.target.resolve(), args.archive.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / '.execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = prepare(batch, target, archive)
        if args.defer_gpt:
            selection.update(deferred_planners=['gpt6'], scheduling='Finish all Kimi tasks; pending GPT tasks deferred by user')
            runner.save(batch / 'selection.json', selection)
        if args.prepare_only:
            print(json.dumps(dict(expected=16, batch=str(batch), archive=str(archive),
                runtime_commit=selection['runtime_commit']), indent=2))
            return
        completed = {item['name']: read(target / item['name'] / 'benchmark_result.json')
            for item in selection['cases'] if (target / item['name'] / 'rerun-verification.json').exists()}
        errors = {}
        adopted = {r['name']: r for r in read(args.adopt_running)['runs']} if args.adopt_running else {}
        guard = threading.Lock()
        try:
            update_report(batch, selection, completed, errors, 'archiving')
            archive_originals(batch, selection)
            (batch / 'cases').mkdir(exist_ok=True)
            (batch / 'empty-memory/global').mkdir(parents=True, exist_ok=True)
            for p in [batch / 'empty-memory/MEMORY.md', batch / 'empty-memory/global/GLOBAL_MEMORY.md']:
                p.write_text('No task-specific or episode-specific external memory is provided.\n')
            env = runner.environment()
            def worker(policy, planner):
                for item in selection['cases']:
                    if item['policy'] != policy or item['case']['planner'] != planner:
                        continue
                    try:
                        row = run_one(batch, selection, item, env, args.inference_config, adopted.get(item['name']))
                        with guard:
                            completed[item['name']] = row
                    except Exception as exc:
                        with guard:
                            errors[item['name']] = repr(exc)
                    with guard:
                        update_report(batch, selection, completed, errors, 'running')
            update_report(batch, selection, completed, errors, 'running')
            for planner in (() if args.finalize_only else ('kimi',) if args.defer_gpt else ('kimi', 'gpt6')):
                # Complete both Kimi policy queues before starting any pending GPT run.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(worker, policy, planner) for policy in ('pi05', 'gawm')]
                    for future in futures:
                        future.result()
                if planner == 'kimi' and any(item['case']['planner'] == 'kimi' and item['name'] not in completed
                                            for item in selection['cases']):
                    break
            for name, digest in selection['untouched_result_hashes'].items():
                assert runner.sha(target / name / 'benchmark_result.json') == digest, name
            deferred = [item['name'] for item in selection['cases']
                if args.defer_gpt and item['case']['planner'] == 'gpt6' and item['name'] not in completed]
            kimi_complete = sum(r['planner'] == 'kimi' for r in completed.values()) == 8
            status = ('kimi_complete_gpt_deferred' if args.defer_gpt and kimi_complete and not errors
                      else 'complete' if len(completed) == 16 and not errors else 'requires_infrastructure_review')
            catalog = runner.refresh(target)
            catalog.update(successes=sum(bool(r['success']) for r in catalog['results']),
                left_right_rerun=dict(batch=str(batch), completed=len(completed), expected=16,
                    status=status, deferred_tasks=deferred))
            runner.save(target / 'summary.json', catalog)
            update_report(batch, selection, completed, errors, status)
            runner.save(batch / 'final-verification.json', dict(status=status, expected=16,
                completed=len(completed), errors=errors, untouched_cases=104,
                kimi_completed=sum(r['planner'] == 'kimi' for r in completed.values()),
                gpt_deferred=deferred,
                untouched_result_hashes_preserved=True, archive=str(archive),
                followup='Refresh catalog matrices, sharing manifest and tool statistics after the completion callback.'))
        except BaseException as exc:
            errors['supervisor'] = repr(exc)
            update_report(batch, selection, completed, errors, 'requires_infrastructure_review')
            raise


if __name__ == '__main__':
    main()
