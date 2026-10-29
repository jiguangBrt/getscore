"""Read-only source audit and full-episode MCP-call statistics."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/long120_1027'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
from rpent.flywheel.episode import validate_episode
from robots.libero.flywheel import LIBERO_SPEC
from run_viewer.parser import parse_run


def read(path):
    return json.loads(path.read_text())


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tool_calls(directory, planner, transcript):
    streams = sorted(p for p in directory.glob('*.stream.jsonl') if p.name != 'kimi_acp.stream.jsonl')
    assert len(streams) == 1, ('normalized stream count', directory, streams)
    stream = streams[0]
    calls, shell, starts, ends, seen = [], [], {}, set(), set()
    for number, line in enumerate(stream.open(), 1):
        raw = json.loads(line)
        item = raw.get('payload', {}).get('item', {})
        kind, method, identifier = item.get('type'), raw.get('method'), item.get('id')
        if kind not in {'mcpToolCall', 'commandExecution'}:
            continue
        if method == 'item/started':
            starts[identifier] = item
        if method != 'item/completed' or identifier in seen:
            continue
        seen.add(identifier)
        ends.add(identifier)
        call = dict(tool=item.get('tool'), status=item.get('status'), arguments=item.get('arguments'),
                    source=str(stream), line=number, id=identifier)
        if kind == 'mcpToolCall':
            calls.append(call)
        else:
            shell.append(dict(command=item.get('command'), exit_code=item.get('exit_code'), **call))
    pending = [dict(id=k, tool=v.get('tool'), type=v.get('type')) for k, v in starts.items() if k not in ends]
    assert not pending, ('unclosed tool invocation', directory, pending)
    if planner == 'kimi':
        native_path = directory / 'kimi_tool_calls.jsonl'
        native = [json.loads(line) for line in native_path.open()]
        assert len(native) == len(calls), ('Kimi native count', directory)
        for index, (a, b) in enumerate(zip(calls, native)):
            assert a['tool'] == b['tool'] and a['arguments'] == b['arguments'], (directory, index)
            assert (a['status'] == 'failed') == bool(b['is_error']), (directory, index, 'status')
            a.update(native_source=str(native_path), native_line=index + 1)
    recorded = transcript.get('stats', {}).get('tool_calls')
    assert recorded == len(calls) + len(shell), ('transcript tool count', directory, recorded, len(calls), len(shell))
    return calls, shell


def audit_segment(directory, decode=True):
    row = read(directory / 'benchmark_result.json')
    config = read(directory / 'configuration.json')
    completeness = read(directory / 'artifact-completeness.json')
    verification = read(directory / 'verification.json')
    transcripts = list(directory.glob('transcript_*.json'))
    assert len(transcripts) == 1, directory
    transcript = read(transcripts[0])
    missing = [name for name in completeness['required_files'] if not (directory / name).is_file()]
    assert not missing, ('required files missing', directory, missing)
    assert completeness['complete'] and config['run_status'] == 'finished', directory
    assert not list(directory.rglob('*.partial')), ('partial trajectories', directory)
    episodes = list(directory.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*/episode.json'))
    assert len(episodes) == 1, ('trajectory count', directory)
    episode_path = episodes[0].parent
    metadata = validate_episode(episode_path, spec=LIBERO_SPEC)
    states = read(directory / 'states.json')['steps']
    assert len(states) > 0 and [s['step_idx'] for s in states] == list(range(len(states))), directory
    assert bool(states[-1]['terminated']) == row['success'] == metadata['is_success'], directory
    with np.load(episode_path / 'transitions.npz', allow_pickle=False) as data:
        actions = len(data['actions'])
        assert actions == row['actions'] == metadata['step_count'], directory
        assert bool(data['terminated'].any()) == row['success'], directory
        assert bool(data['truncated'].any()) == row['truncated'] == bool(states[-1].get('truncated')), directory
    assert metadata['proposal_count'] == row['policy_queries'], directory
    counts = read(directory / 'native-action-counts.json')
    assert counts['total_frames'] == actions == sum(s['frames'] for s in counts['steps'].values()), directory
    traces = [json.loads(line) for line in (directory / 'command_trace.jsonl').open()]
    commands = [s for s in states if s.get('command')]
    assert len(traces) == len(commands), ('trace/state count', directory)
    for trace, state in zip(traces, commands):
        assert trace['step'] == state['step_idx'] and trace['command'] == state['command'], directory
    observation_files = 0
    for state in states:
        for name in state.get('artifacts', []):
            p = directory / name
            if p.is_dir():
                p = p / f"{state['step_idx']:02d}{Path(name).suffix}"
            assert p.is_file(), ('missing state artifact', directory, state['step_idx'], name)
            observation_files += 1
    for name, expected in verification['sha256'].items():
        candidates = [directory / name] if (directory / name).is_file() else list(episode_path.glob(name))
        assert len(candidates) == 1 and sha(candidates[0]) == expected, ('hash', directory, name)
    video_frames = row.get('total_episode_actions', actions)
    if decode:
        proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-threads', '1',
                               '-i', str(directory / 'episode.mp4'), '-map', '0:v:0', '-f', 'null', '-',
                               '-progress', 'pipe:1'], capture_output=True, text=True, timeout=120)
        frames = [int(x) for x in re.findall(r'^frame=(\d+)$', proc.stdout, re.M)]
        assert proc.returncode == 0 and not proc.stderr.strip() and frames[-1] == video_frames, ('video decode', directory, proc.stderr)
    viewer = parse_run(directory)
    assert not viewer['warnings'], ('viewer warnings', directory, viewer['warnings'])
    assert viewer['video']['frames'] == video_frames, directory
    calls, shell = tool_calls(directory, row['planner'], transcript)
    anomalies = []
    if transcript.get('error') or row.get('exit_code') or row.get('planner_completion_error'):
        anomalies.append(dict(kind='planner_completion_error', transcript_error=transcript.get('error'),
                              result_error=row.get('error'), exit_code=row.get('exit_code'),
                              native_success=row['success'], native_truncated=row['truncated'], finish=transcript.get('finish')))
    return dict(directory=str(directory), actions=actions, video_frames=video_frames, success=row['success'],
                sha256_files_checked=len(verification['sha256']), observation_files_checked=observation_files,
                raw_trajectory_valid=True, video_fully_decoded=decode, required_files_checked=len(completeness['required_files']),
                viewer_warnings=[], calls=calls, shell=shell, anomalies=anomalies,
                model=config.get('model'), execution_mode=row.get('execution_mode', 'fresh'))


def audit_run(directory):
    row = read(directory / 'benchmark_result.json')
    current = audit_segment(directory)
    segments = []
    if row.get('execution_mode') == 'resume':
        source = Path(row['source_run'])
        assert source.is_dir(), ('missing prefix archive', source)
        prefix = audit_segment(source)
        assert prefix['actions'] == row['resumed_from_step'], directory
        recovery = read(directory / 'recovery.json')
        assert recovery['initial_images_exact'] and recovery['final_images_exact'] and recovery['first_replay_mismatch_step'] is None, directory
        assert prefix['actions'] + current['actions'] == row['total_episode_actions'], directory
        prefix['role'] = 'archived_prefix'
        segments.append(prefix)
    current['role'] = 'continuation' if segments else 'full_episode'
    segments.append(current)
    calls = [c for segment in segments for c in segment['calls']]
    return dict(name=directory.name, variant=row['variant'], task=row['task'], task_language=row['policy_prompt'],
                planner=row['planner'], policy=row['policy'], success=row['success'],
                execution_mode=current['execution_mode'], full_episode_mcp_counts=dict(Counter(c['tool'] for c in calls)),
                failed_mcp_counts=dict(Counter(c['tool'] for c in calls if c['status'] == 'failed')),
                shell_command_count=sum(len(s['shell']) for s in segments), segments=segments,
                data_complete=True, planner_clean=not any(s['anomalies'] for s in segments))


def main():
    directories = sorted(p for p in SOURCE.iterdir() if p.is_dir() and p.name.startswith('long-'))
    expected = {f'long-{v}-{t}-{p}-{b}' for v in ['original', 'swap'] for t in range(10)
                for p in ['gpt6', 'kimi', 'none'] for b in ['gawm', 'pi05']}
    assert {p.name for p in directories} == expected
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(audit_run, d): d for d in directories}
        for future in as_completed(futures):
            directory = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                failures.append(dict(name=directory.name, error=f'{type(exc).__name__}: {exc}'))
                print('FAILED', directory.name, str(exc), flush=True)
            if (len(results) + len(failures)) % 10 == 0:
                print('AUDITED', len(results) + len(failures), '/120; failures', len(failures), flush=True)
    results.sort(key=lambda r: r['name'])
    save('audit.json', dict(source=str(SOURCE), checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                            expected_runs=120, audited_runs=len(results), failures=failures, runs=results))
    if failures:
        print(json.dumps(failures, ensure_ascii=False), flush=True)
        return
    llm = [r for r in results if r['planner'] != 'none']
    assert len(llm) == 80
    tools = sorted({tool for r in llm for tool in r['full_episode_mcp_counts']})
    groups = []
    for variant in ['all', 'original', 'swap']:
        for task in range(10):
            runs = [r for r in llm if r['task'] == task and (variant == 'all' or r['variant'] == variant)]
            counts = Counter()
            failed = Counter()
            for r in runs:
                counts.update(r['full_episode_mcp_counts'])
                failed.update(r['failed_mcp_counts'])
            groups.append(dict(variant=variant, task=task, experiments=len(runs), task_language=runs[0]['task_language'],
                               calls={t: counts[t] for t in tools}, failed_calls={t: failed[t] for t in tools}))
    total = Counter()
    for r in llm:
        total.update(r['full_episode_mcp_counts'])
    stats = dict(experiments=80, planners=dict(Counter(r['planner'] for r in llm)), tools=tools,
                 total_calls=dict(total), total=sum(total.values()), by_task=groups,
                 per_run=[{k: v for k, v in r.items() if k != 'segments'} for r in llm],
                 counting_rule='Full-episode completed MCP invocations including returned errors; one item ID once; '
                 'archived prefix plus continuation for three resumed runs; restarts only use the latest fresh episode; '
                 'local shell commands and resumed pending action chunks are excluded from MCP counts.')
    save('tool_counts.json', stats)
    print('COMPLETE', json.dumps(dict(audited=len(results), llm_runs=len(llm), total_mcp=stats['total'], tools=tools), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
