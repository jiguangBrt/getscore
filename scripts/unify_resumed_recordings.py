"""Publish resumed episodes as normal, self-contained run_viewer recordings."""
from __future__ import annotations

from collections import Counter
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/long120_1027'
sys.path[:0] = [str(ROOT), str(ROOT / 'RPent')]
from run_viewer.parser import parse_run
from scripts.organize_long_benchmark import normalize_events
from rpent.flywheel.episode import validate_episode
from robots.libero.flywheel import LIBERO_SPEC

NAMES = ['long-original-4-kimi-pi05', 'long-original-6-kimi-gawm', 'long-original-6-kimi-pi05']


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def records(path):
    return [json.loads(line) for line in path.open()]


def save_lines(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def episode(directory):
    matches = list(directory.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*/episode.json'))
    assert len(matches) == 1, directory
    return matches[0].parent


def remap(value, offset, artifacts):
    if isinstance(value, list):
        return [remap(item, offset, artifacts) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {'step', 'step_idx'} and type(item) is int and item >= 0:
                result[key] = item + offset
            else:
                result[key] = remap(item, offset, artifacts)
        return result
    if isinstance(value, str):
        return artifacts.get(value, value)
    return value


def merge_trajectory(prefix, suffix, destination):
    a_path, b_path = episode(prefix), episode(suffix)
    a_meta, b_meta = read(a_path / 'episode.json'), read(b_path / 'episode.json')
    native_offset = a_meta['step_count']
    primitive_offset = len(a_meta['primitive_names'])
    proposal_offset = a_meta['proposal_count']
    with np.load(a_path / 'transitions.npz') as left, np.load(b_path / 'transitions.npz') as right:
        merged = {}
        for key in left.files:
            a, b = left[key], right[key]
            if key in {'main_images', 'wrist_images', 'states'}:
                if key == 'states':
                    np.testing.assert_allclose(a[-1], b[0], rtol=0, atol=1e-6)
                else:
                    assert np.array_equal(a[-1], b[0]), (prefix.name, key)
                merged[key] = np.concatenate([a, b[1:]])
            else:
                if key == 'primitive_id':
                    b = np.where(b >= 0, b + primitive_offset, b)
                elif key == 'vla_chunk_id':
                    b = np.where(b >= 0, b + proposal_offset, b)
                elif key == 'truncated':
                    merged['segment_budget_truncated'] = np.concatenate([a, b])
                    a = np.zeros_like(a)
                merged[key] = np.concatenate([a, b])
        count = len(merged['actions'])
        training = np.flatnonzero(merged['terminated'])
        metadata = dict(b_meta, step_count=count, proposal_count=proposal_offset + b_meta['proposal_count'],
                        primitive_names=a_meta['primitive_names'] + b_meta['primitive_names'],
                        training_step_count=int(training[0] + 1) if len(training) else 0,
                        recording_scope='single_continuous_episode',
                        recording_note='Lossless recorded observations/actions concatenated at their verified common state. '
                                       'Old intermediate budget truncation is retained in segment_budget_truncated; '
                                       'truncated describes the final episode budget only.')
        destination.mkdir(parents=True)
        np.savez_compressed(destination / 'transitions.npz', **merged)
        del merged
    with np.load(a_path / 'proposals.npz') as left, np.load(b_path / 'proposals.npz') as right:
        proposals = {}
        for key in left.files:
            a, b = left[key], right[key]
            if key == 'created_step':
                b = b + native_offset
            elif key == 'primitive_id':
                b = np.where(b >= 0, b + primitive_offset, b)
            proposals[key] = np.concatenate([a, b])
        np.savez_compressed(destination / 'proposals.npz', **proposals)
    save(destination / 'episode.json', metadata)
    validate_episode(destination, spec=LIBERO_SPEC)
    return metadata


def prepare(name, archive):
    source = DATA / name
    prefix = source / 'recording-prefix'
    assert prefix.is_dir() and read(source / 'benchmark_result.json')['execution_mode'] == 'resume'
    stage = archive / 'staging' / name
    stage.parent.mkdir(parents=True, exist_ok=True)
    excluded = {'recording-prefix', 'trajectory', 'states.json', 'kimi_tool_calls.jsonl', '00-planner.stream.jsonl',
                'command_trace.jsonl', 'native-action-counts.json', 'video-timeline.json', 'recovery.json',
                'episode-full.mp4', 'episode-continuation.mp4', 'full-video-verification.json',
                'tool_calls-full.jsonl', 'command_trace-full.jsonl', 'states-full.json', 'transcript-full.json',
                'full-process.html', 'full-process-manifest.json', 'README-sharing.md'}
    for path in source.iterdir():
        if path.is_dir():
            excluded.add(path.name)
    shutil.copytree(source, stage, ignore=lambda directory, names: set(names) & excluded if Path(directory) == source else set())
    original = read(source / 'benchmark_result.json')
    before = read(prefix / 'benchmark_result.json')
    prefix_states = read(prefix / 'states.json')['steps']
    observation_offset = len(prefix_states) - 1
    max_segment = max([int(m[1]) for p in prefix.iterdir() if (m := re.fullmatch(r'segment_(?:overlay_)?(\d+)\.(?:json|png)', p.name))] or [-1])
    artifact_names = {}
    for path in source.iterdir():
        match = re.fullmatch(r'(segment_(?:overlay_)?)(\d+)(\.(?:json|png))', path.name)
        if match:
            artifact_names[path.name] = f'{match[1]}{int(match[2]) + max_segment + 1:02d}{match[3]}'
    states, tools, traces, acp, messages, stats_segments = [], [], [], [], [], []
    counts = {}
    provenance = []
    for label, directory, offset, artifact_map, native_offset in [
        ('prefix', prefix, 0, {}, 0), ('continuation', source, observation_offset, artifact_names, before['actions'])
    ]:
        old_states = read(directory / 'states.json')['steps']
        native = read(directory / 'native-action-counts.json')
        for old in old_states:
            new = remap(old, offset, artifact_map)
            new['recording_source'] = dict(segment=label, local_step=old['step_idx'])
            if label == 'continuation' and old['step_idx'] == 0:
                states[-1]['artifacts'] = sorted(set(states[-1].get('artifacts', [])) | set(new.get('artifacts', [])))
                continue
            if label == 'prefix':
                if new.get('truncated'):
                    new['recording_source']['original_budget_truncated'] = True
                new['truncated'] = False
            states.append(new)
        for number, entry in enumerate(records(directory / 'kimi_tool_calls.jsonl'), 1):
            tools.append(dict(remap(entry, offset, artifact_map), recording_source=dict(segment=label, line=number)))
        for entry in records(directory / 'command_trace.jsonl'):
            traces.append(dict(remap(entry, offset, artifact_map), recording_source=dict(segment=label, local_step=entry['step'])))
        for key, entry in native['steps'].items():
            counts[str(int(key) + offset)] = entry
        for entry in records(directory / 'kimi_acp.stream.jsonl'):
            acp.append(dict(entry, recording_segment=label))
        transcript = read(next(directory.glob('transcript_*.json')))
        messages += [dict(remap(m, offset, artifact_map), recording_segment=label) for m in transcript['messages']]
        stats_segments.append(transcript['stats'])
        provenance.append(dict(segment=label, original_actions=native['total_frames'], original_observations=len(old_states),
                               original_tool_calls=len(records(directory / 'kimi_tool_calls.jsonl')),
                               observation_index_offset=offset, native_action_offset=native_offset,
                               source_sha256={p.name: sha(p) for p in [directory / 'states.json', directory / 'kimi_tool_calls.jsonl',
                                                                      directory / '00-planner.stream.jsonl', next(directory.glob('transcript_*.json'))]},
                               artifact_renames=artifact_map))
        # Every observation directory is addressed by state number. Preserve
        # the image/depth bytes, remap filenames, and remap JSON state metadata.
        for folder in directory.iterdir():
            if not folder.is_dir() or folder.suffix not in {'.png', '.npz', '.json'}:
                continue
            destination = stage / artifact_map.get(folder.name, folder.name)
            destination.mkdir(exist_ok=True)
            for path in folder.iterdir():
                match = re.fullmatch(r'(\d+)(\.[^.]+)', path.name)
                assert match, path
                index = int(match[1])
                target = destination / f'{index + offset:02d}{match[2]}'
                if target.exists() and label == 'continuation' and index == 0:
                    continue  # Verified common boundary observation.
                assert not target.exists(), target
                if path.suffix == '.json':
                    save(target, remap(read(path), offset, artifact_map))
                else:
                    shutil.copy2(path, target)
    assert [s['step_idx'] for s in states] == list(range(len(states)))
    continuation_prompt = next(e['time_unix'] for e in acp
                               if e['recording_segment'] == 'continuation'
                               and e.get('direction') == 'client'
                               and e.get('message', {}).get('method') == 'session/prompt')
    for state in states:
        if (state.get('command') or {}).get('action') == 'resume_pending_policy_actions':
            state.update(recorded_at_ms=continuation_prompt * 1000,
                         recorded_at_basis='Adjacent next recorded model prompt; no separate action wall-clock timestamp was recorded.')
    save(stage / 'states.json', dict(run_artifacts=['episode.mp4'], steps=states))
    save_lines(stage / 'kimi_tool_calls.jsonl', tools)
    save_lines(stage / 'command_trace.jsonl', traces)
    save_lines(stage / 'kimi_acp.stream.jsonl', acp)
    save(stage / 'native-action-counts.json', dict(total_frames=original['total_episode_actions'], steps=counts,
          source='Merged native primitive IDs; continuous state numbering and video frames from the episode start.'))
    transcript_path = next(stage.glob('transcript_*.json'))
    transcript = read(transcript_path)
    transcript['messages'] = messages
    transcript['stats']['tool_calls'] = len(tools)
    transcript['stats']['elapsed_s'] = round(sum(s.get('elapsed_s', 0) for s in stats_segments), 2)
    transcript['stats'].update(output_path=str(source / 'kimi.txt'), raw_stream_path=str(source / 'kimi_acp.stream.jsonl'),
                               recording_scope='single_episode', original_sessions=stats_segments)
    for key in ['recovery_mode', 'resumed_from_steps']:
        transcript['stats'].pop(key, None)
    transcript['elapsed_s'] = round(read(next(prefix.glob('transcript_*.json')))['elapsed_s'] + transcript['elapsed_s'], 2)
    transcript['environment_success'] = original['success']
    save(transcript_path, transcript)
    for filename in ['run.log', 'env_server.log', 'console.log', 'bridge.log', 'kimi.txt', 'kimi_stderr.log', 'kimi_prompt.txt']:
        paths = [p / filename for p in [prefix, source] if (p / filename).is_file()]
        if paths:
            (stage / filename).write_text('\n'.join(p.read_text(errors='replace') for p in paths))
    for filename in ['network-timing.jsonl']:
        save_lines(stage / filename, [r for d in [prefix, source] for r in records(d / filename)])
    session = read(stage / 'kimi_session.json')
    session['recorded_sessions'] = [read(d / 'kimi_session.json') for d in [prefix, source]]
    save(stage / 'kimi_session.json', session)
    normalize_events(stage, 'kimi')
    suffix_episode = episode(source)
    new_episode = stage / suffix_episode.relative_to(source)
    print('MERGING TRAJECTORY', name, flush=True)
    metadata = merge_trajectory(prefix, source, new_episode)
    row = dict(original, actions=metadata['step_count'], frames=metadata['step_count'],
               total_episode_actions=metadata['step_count'], policy_queries=metadata['proposal_count'],
               initial_observation_sha256=before['initial_observation_sha256'],
               elapsed_s=round(before['elapsed_s'] + original['elapsed_s'], 2),
               execution_mode='single_episode', trajectory_scope='full_episode', video_scope='full_episode',
               directory=str(source), trajectory=str(source / new_episode.relative_to(stage)),
               transcript=str(source / transcript_path.name), recording_scope='full_episode')
    for key in ['source_run', 'resumed_from_step', 'new_actions', 'recovery_verification', 'continuation_frames', 'continuation_video', 'full_video']:
        row.pop(key, None)
    save(stage / 'benchmark_result.json', row)
    config = read(stage / 'configuration.json')
    config.update(stats=transcript['stats'], started_at=read(prefix / 'configuration.json')['started_at'],
                  execution_mode='single_episode', total_episode_actions=row['actions'], recording_scope='full_episode')
    for key in ['source_run', 'resumed_from_step']:
        config.pop(key, None)
    save(stage / 'configuration.json', config)
    files = [stage / 'episode.mp4', stage / 'states.json', transcript_path, stage / 'kimi_tool_calls.jsonl',
             stage / '00-planner.stream.jsonl', stage / 'command_trace.jsonl', new_episode / 'transitions.npz', new_episode / 'proposals.npz']
    save(stage / 'verification.json', dict(success_agrees_with_native=True, frames_equal_actions=True, raw_trajectory_valid=True,
          frames=row['frames'], actions=row['actions'], sha256={p.name: sha(p) for p in files}))
    tag = transcript_path.stem.removeprefix('transcript_')
    audit = read(stage / f'{tag}.json')
    if isinstance(audit.get('strategy_notes'), dict):
        audit['strategy_notes'].update(actions=row['actions'], policy_queries=row['policy_queries'],
                                        recording_scope='full_episode')
    save(stage / f'{tag}.json', audit)
    recipe = stage / f'{tag}_recipe.jsonl'
    if recipe.is_file():
        # Exact recorded-action replay includes the old partial chunk without
        # introducing a nonexistent model tool or depending on old budgets.
        with np.load(new_episode / 'transitions.npz') as raw:
            save_lines(recipe, [dict(action='execute_action', values=action.tolist()) for action in raw['actions']])
    report = parse_run(stage)
    assert report['warnings'] == [], (name, report['warnings'])
    assert report['metrics']['tool_calls'] == report['metrics']['recorded_tool_calls'] == len(tools)
    assert report['status'] == ('success' if row['success'] else 'failure')
    actions = [e for e in report['events'] if e.get('action')]
    assert sum(e['video_range']['frames'] for e in actions) == row['actions']
    assert not any(e.get('video_prefix') for e in report['events'])
    ordered = sorted(actions, key=lambda e: e['after_state'])
    cursor = 0
    for event in ordered:
        clip = event['video_range']
        assert abs(clip['start_s'] * row['fps'] - cursor) < 1e-6, (name, event['title'])
        cursor += clip['frames']
    assert cursor == row['actions']
    required = [p.name for p in stage.iterdir() if p.is_file() and p.name not in {'artifact-completeness.json', 'recording-provenance.json'}]
    save(stage / 'artifact-completeness.json', dict(complete=True, format_version=4, directory=str(source),
          required_files=sorted(required), video_frames=row['frames'], native_actions=row['actions'],
          raw_trajectory=row['trajectory'], recording_scope='full_episode',
          viewer=dict(video_loaded=True, observations=len(states), tool_events=len(tools),
                      action_events=len(actions), action_clip_frames=cursor, warnings=[])))
    save(stage / 'recording-provenance.json', dict(format_version=1, presentation='single_episode',
          original_segments=provenance, original_recordings_archive=str(archive / 'before' / name),
          intermediate_budget_truncation='Only the boundary flags in canonical states/trajectory were cleared; original tool-return evidence is retained.',
          native_actions=row['actions'], tool_calls=len(tools), no_new_model_inference=True, no_new_simulation=True))
    print('PREPARED', name, 'actions', cursor, 'observations', len(states), 'tools', len(tools), flush=True)
    return row


def main():
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    archive = ROOT / 'outputs/_archive/records' / f'unified-runviewer-{stamp}'
    archive.mkdir(parents=True)
    metadata_backup = archive / 'metadata-before'
    metadata_backup.mkdir()
    for path in DATA.iterdir():
        if path.is_file() and path.suffix in {'.json', '.md', '.html', '.csv', '.py'}:
            shutil.copy2(path, metadata_backup / path.name)
    rows = [prepare(name, archive) for name in NAMES]
    before = archive / 'before'
    before.mkdir()
    for name in NAMES:
        (DATA / name).rename(before / name)
        (archive / 'staging' / name).rename(DATA / name)
    summary = read(DATA / 'summary.json')
    replacements = {r['name']: r for r in rows}
    summary['results'] = [replacements.get(r['name'], r) for r in summary['results']]
    summary['unified_runviewer_recordings'] = NAMES
    save(DATA / 'summary.json', summary)
    with (DATA / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        headers = csv.DictReader(stream).fieldnames
    with (DATA / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary['results'])
    consolidation = read(DATA / 'consolidation.json')
    consolidation['unified_runviewer_recordings'] = dict(verified=3, original_archive=str(archive), canonical_files=True)
    save(DATA / 'consolidation.json', consolidation)
    save(archive / 'manifest.json', dict(status='canonical_recordings_published', names=NAMES, results=rows, backups=str(before)))
    print('PUBLISHED', archive, flush=True)


if __name__ == '__main__':
    main()
