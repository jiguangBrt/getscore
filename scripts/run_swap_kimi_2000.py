"""Fresh 2000-action reruns of the failed Kimi swap cases in long120_1027."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import socket
import sys
import threading
import time
import zipfile
from urllib.parse import urlsplit
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
from scripts import run_long_benchmark as runner


def initial_state(path):
    """Read the first proprioceptive state without expanding episode images."""
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open('states.npy') as stream:
        version=np.lib.format.read_magic(stream)
        header=np.lib.format.read_array_header_1_0 if version[0]==1 else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=header(stream)
        if fortran or dtype.hasobject:
            raise RuntimeError('Unsupported trajectory state representation')
        return np.frombuffer(stream.read(int(np.prod(shape[1:]))*dtype.itemsize),dtype=dtype).reshape(shape[1:])


def verify_initialization(source, output, source_row, row):
    """Require matching task inputs, and expose known fixture randomization."""
    import numpy as np
    old=json.loads((source/'configuration.json').read_text())
    new=json.loads((output/'configuration.json').read_text())
    for key in ('variant','suite','task','seed','initial_state_index'):
        if old[key]!=new[key]:raise RuntimeError(f'Initial condition changed: {key}')
    for key in ('bddl_files','init_files'):
        if old[key]['sha256']!=new[key]['sha256']:
            raise RuntimeError(f'Source task assets changed: {key}')
    prior=initial_state(Path(source_row['trajectory'])/'transitions.npz')
    current=initial_state(Path(row['trajectory'])/'transitions.npz')
    if not np.array_equal(prior,current):
        raise RuntimeError('Initial robot proprioception differs from source')
    references=[json.loads(p.read_text()) for p in source.parent.glob(
        f'long-{old["variant"]}-{old["task"]}-*/benchmark_result.json')]
    distinct=len({r['initial_observation_sha256'] for r in references})
    exact=source_row['initial_observation_sha256']==row['initial_observation_sha256']
    if not exact and distinct<=1:
        raise RuntimeError('Unexpected initial-image mismatch in a visually deterministic task')
    audit=dict(task_seed_and_init_index_match=True,task_asset_hashes_match=True,
               initial_robot_state_exact=True,initial_observation_exact=exact,
               original_reference_count=len(references),original_distinct_initial_observations=distinct,
               comparison_scope='nominal task/seed/init-file pairing; fixture layout is not serialized in init-state data')
    runner.save(output/'initialization-audit.json',audit)
    return audit


def prepare(batch, source):
    batch.mkdir(parents=True, exist_ok=True)
    selection_path = batch / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection['source_directory'] != str(source):
            raise RuntimeError('Existing batch has a different source directory')
        return selection
    cases = []
    for directory in sorted(source.glob('long-swap-*-kimi-*')):
        row = json.loads((directory / 'benchmark_result.json').read_text())
        config = json.loads((directory / 'configuration.json').read_text())
        if row['success']:
            continue
        if row.get('error') or config['model'] != 'kimi-code/k3.1-pro':
            raise RuntimeError(f'Unexpected source configuration: {directory}')
        cases.append(dict(source_run=str(directory), policy=row['policy'],
                          source_actions=row['actions'], source_truncated=row['truncated'],
                          source_result_sha256=runner.sha(directory / 'benchmark_result.json'),
                          case={key: config[key] for key in
                                ('variant', 'suite', 'task', 'name', 'policy_prompt', 'planner')}))
    expected = {'pi05': {0, 3, 4, 5, 6, 8, 9}, 'gawm': {0, 3, 5, 6, 8, 9}}
    actual = {policy: {item['case']['task'] for item in cases if item['policy'] == policy}
              for policy in expected}
    if len(cases) != 13 or actual != expected:
        raise RuntimeError(f'Selection differs from the requested 13 failures: {actual}')
    selection = dict(created_at=runner.now(), source_directory=str(source),
                     execution_mode='fresh', max_episode_steps=2000, max_turns=160,
                     planner_timeout_s=7200, model='kimi-code/k3.1-pro',
                     reasoning_effort='high', seed=0, initial_state_index=0,
                     expected=13, cases=cases)
    runner.save(selection_path, selection)
    (batch / 'cases').mkdir(exist_ok=True)
    (batch / 'empty-memory/global').mkdir(parents=True, exist_ok=True)
    for path in [batch / 'empty-memory/MEMORY.md', batch / 'empty-memory/global/GLOBAL_MEMORY.md']:
        path.write_text('No task-specific or episode-specific external memory is provided.\n')
    (batch / 'completion-prompt.txt').write_text(
        '用户授权的 swap Kimi 2000 步重跑进程已经退出，这是一次性完成事件回调。'
        f'请读取 {batch}/status.json、selection.json、summary.json、report.md 及 completion-hook.json，'
        '核验 13 次任务的完成情况、3.1pro 模型、2000 步上限、原生成功结果和视频轨迹完整性，'
        '然后在本会话用中文报告重跑结果及相对原结果的变化。'
        '原始目录 outputs/long120_1027 应保持原样；若有基础设施问题，处理问题并继续未完成项，'
        '不要重复运行已有有效结果。用户要求无需持续监控，请不要启动周期监控。\n')
    return selection


def update_report(batch, selection, completed, errors, status):
    with runner.LOCK:
        ordered = sorted(completed, key=lambda row: (row['policy'], row['task']))
        runner.save(batch / 'status.json', dict(status=status, expected=13,
                    completed=len(ordered), successes=sum(row['success'] for row in ordered),
                    errors=errors, updated_at=runner.now()))
        lines = ['# Swap Kimi 2000 步重跑', '',
                 f'完成 {len(ordered)}/13，成功 {sum(row["success"] for row in ordered)}。', '',
                 '模型 kimi-code/k3.1-pro，high；seed 0、初始状态 0；从头重跑，单次最多 2000 动作，规划时间预算 7200 秒。',
                 '原始 long120_1027 结果保持原样。成功以原生环境终止为准。', '',
                 '| 任务 | 策略 | 成功 | 动作数 | 视频 |', '|---|---|---|---:|---|']
        for row in ordered:
            lines.append(f'| {row["task"]} | {row["policy"]} | {row["success"]} | {row["actions"]} | [episode.mp4]({row["video"]}) |')
        if errors:
            lines += ['', '运行异常：'] + [f'- {name}: {error}' for name, error in sorted(errors.items())]
        varied=[row for row in ordered if not row.get('initial_observation_exact',True)]
        if varied:
            lines += ['',f'初始画面限制：{len(varied)} 项与旧记录不逐像素一致。旧批次同任务同 seed 的 6 种方法也存在不同初始画面；本次核对了任务、seed、初态索引、场景文件和机器人初态一致。固定家具位置在原初始化流程中仍可能随机变化，因此不是严格固定画面的配对实验。']
        (batch / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/long120_1027')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inference-config', type=Path)
    parser.add_argument('--wait-for-services', action='store_true',
                        help='Wait in the background for policy/perception ports before running')
    args = parser.parse_args()
    batch, source = args.batch.resolve(), args.source.resolve()
    selection = prepare(batch, source)
    if args.prepare_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    with (batch / '.execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed, errors = [], {}
        stop = threading.Event()
        update_report(batch, selection, completed, errors, 'starting')
        try:
            if args.wait_for_services:
                while True:
                    services = {}
                    configured=runner.inference_services(args.inference_config)
                    sam=urlsplit(configured['simulation']['sam3_endpoint'])
                    endpoints=[(configured[key]['host'],configured[key]['port']) for key in ('openpi','gawm')]
                    endpoints.append((sam.hostname,sam.port or 80))
                    for host,port in endpoints:
                        try:
                            with socket.create_connection((host, port), timeout=5):
                                services[f'{host}:{port}'] = True
                        except OSError:
                            services[f'{host}:{port}'] = False
                    runner.save(batch / 'service-readiness.json', dict(
                        endpoints=services, checked_at=runner.now()))
                    if all(services.values()):
                        break
                    update_report(batch, selection, completed, errors, 'waiting_for_services')
                    time.sleep(60)
            env = runner.environment()
            run_args = SimpleNamespace(max_steps=2000, max_turns=160, timeout=7200,
                                       inference_config=args.inference_config)

            def worker(policy):
                for item in selection['cases']:
                    if item['policy'] != policy or stop.is_set():
                        continue
                    original = Path(item['source_run'])
                    name = original.name
                    output = batch / 'cases' / name
                    try:
                        if runner.sha(original / 'benchmark_result.json') != item['source_result_sha256']:
                            raise RuntimeError('Source result changed after selection')
                        runner.run_case(batch, item['case'], policy, env, run_args)
                        config = json.loads((output / 'configuration.json').read_text())
                        error = config.get('error')
                        if error:
                            raise RuntimeError(error)
                        row = json.loads((output / 'benchmark_result.json').read_text())
                        source_row = json.loads((original / 'benchmark_result.json').read_text())
                        if row.get('error') or not 0 < row['actions'] <= 2000:
                            raise RuntimeError(f'Invalid result: {row.get("error")} actions={row["actions"]}')
                        if row['truncated'] and not row['success'] and row['actions'] != 2000:
                            raise RuntimeError('Episode truncated before its 2000-action limit')
                        initialization=verify_initialization(original,output,source_row,row)
                        session = json.loads((output / 'kimi_session.json').read_text())
                        if session['confirmed_settings']['model']['currentValue'] != selection['model']:
                            raise RuntimeError('Kimi did not confirm the requested model')
                        row.update(source_run=str(original), execution_mode='fresh',
                                   target_total_steps=2000, total_episode_actions=row['actions'],
                                   initial_observation_exact=initialization['initial_observation_exact'],
                                   initialization_audit=str(output/'initialization-audit.json'))
                        runner.save(output / 'benchmark_result.json', row)
                        with runner.LOCK:
                            completed.append(row)
                    except Exception as exc:
                        with runner.LOCK:
                            errors[name] = str(exc)
                        text = str(exc).lower()
                        if any(marker in text for marker in ('quota', 'usage limit', 'provider infrastructure', '429')):
                            stop.set()
                    update_report(batch, selection, completed, errors, 'running')

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, policy) for policy in ('pi05', 'gawm')]
                for future in futures:
                    future.result()
            runner.refresh(batch)
            status = 'complete' if len(completed) == 13 and not errors else 'requires_infrastructure_review'
            update_report(batch, selection, completed, errors, status)
            print(runner.now(), status, f'{len(completed)}/13', flush=True)
        except Exception as exc:
            errors['supervisor'] = repr(exc)
            update_report(batch, selection, completed, errors, 'requires_infrastructure_review')
            raise


if __name__ == '__main__':
    main()
