"""Render complete resumed-episode videos from recorded RGB, preserving evidence."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_viewer.parser import parse_run, video_metadata
from run_viewer.server import discover_runs
from scripts.consolidate_long120_1027 import fingerprint, save


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def trajectory(directory):
    files = list(directory.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*/transitions.npz'))
    assert len(files) == 1, directory
    return files[0]


def main():
    target = ROOT / 'outputs/long120_1027'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    archive = ROOT / 'outputs/_archive/records' / f'full-episode-videos-{stamp}'
    runs = discover_runs(target)
    assert len(runs) == 120
    candidates = []
    for run in runs:
        directory = Path(run['path'])
        row = json.loads((directory / 'benchmark_result.json').read_text())
        actual = video_metadata(directory / 'episode.mp4')
        assert actual is not None
        expected = row.get('total_episode_actions', row['actions'])
        if actual['frames'] != expected:
            assert row.get('execution_mode') == 'resume'
            assert actual['frames'] == row['actions'] == row['new_actions']
            assert actual['frames'] + row['resumed_from_step'] == expected
            candidates.append((directory, row))
    assert len(candidates) == 3, [str(p) for p, _ in candidates]
    archive.mkdir(parents=True)
    for path in target.iterdir():
        if path.is_file() and path.suffix in {'.json', '.csv', '.md'}:
            shutil.copy2(path, archive / path.name)
    entries = []
    for directory, row in candidates:
        source = Path(row['source_run'])
        original_hashes = fingerprint(source)
        recorded_hashes = fingerprint(directory)
        backup = archive / directory.name
        backup.mkdir()
        for name in ['episode.mp4', 'episode-full.mp4', 'benchmark_result.json', 'configuration.json',
                     'verification.json', 'full-video-verification.json', 'artifact-completeness.json',
                     'native-action-counts.json']:
            path = directory / name
            if path.exists(): shutil.copy2(path, backup / name)
        entry = dict(name=directory.name, directory=str(directory), archived_previous_video=str(backup / 'episode.mp4'),
                     prefix_source=str(source), prefix_trajectory=str(trajectory(source)),
                     continuation_trajectory=str(trajectory(directory)), original_recorded_hashes=recorded_hashes,
                     prefix_recorded_hashes=original_hashes, prefix_frames=row['resumed_from_step'],
                     continuation_frames=row['actions'], total_frames=row['total_episode_actions'], fps=row['fps'])
        entries.append(entry)
    manifest = dict(status='preflight_verified', created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    entries=entries, renderer='recorded main_images[1:] from prefix and continuation transitions.npz; CPU libx264 CRF18',
                    no_new_inference=True, no_new_simulation=True)
    save(archive / 'manifest.json', manifest)

    # Render and decode every new video before replacing any default video.
    for entry in entries:
        temporary = archive / entry['name'] / 'rendered-full.mp4'
        samples = {}
        cursor = 0
        last = None
        with imageio.get_writer(temporary, fps=entry['fps'], codec='libx264', pixelformat='yuv420p',
                                ffmpeg_log_level='error', output_params=['-crf', '18', '-threads', '2', '-movflags', '+faststart']) as writer:
            for index, key in enumerate(['prefix_trajectory', 'continuation_trajectory']):
                with np.load(entry[key]) as data:
                    images = data['main_images']
                    assert len(images) == len(data['actions']) + 1
                    assert len(data['actions']) == entry['prefix_frames' if index == 0 else 'continuation_frames']
                    if index:
                        assert np.array_equal(last['main'], images[0])
                        assert np.array_equal(last['wrist'], data['wrist_images'][0])
                        np.testing.assert_allclose(last['state'], data['states'][0], rtol=0, atol=1e-6)
                    else:
                        last = dict(main=images[-1].copy(), wrist=data['wrist_images'][-1].copy(), state=data['states'][-1].copy())
                    for frame in images[1:]:
                        if cursor in {0, entry['prefix_frames'] - 1, entry['prefix_frames'], entry['total_frames'] - 1}:
                            samples[cursor] = frame.copy()
                        writer.append_data(frame)
                        cursor += 1
        assert cursor == entry['total_frames']
        count, errors = 0, {}
        with imageio.get_reader(temporary) as reader:
            assert abs(reader.get_meta_data()['fps'] - entry['fps']) < 1e-6
            for index, frame in enumerate(reader):
                assert frame.shape == (256, 256, 3)
                if index in samples:
                    error = float(np.abs(frame.astype(np.int16) - samples[index].astype(np.int16)).mean())
                    assert error < 8, (entry['name'], index, error)
                    errors[str(index)] = error
                count += 1
        assert count == entry['total_frames'] and len(errors) == 4
        entry.update(rendered_sha256=sha(temporary), all_frames_decoded=True, boundary_images_exact=True,
                     sample_rgb_mean_absolute_errors=errors)
        print('RENDERED', entry['name'], count, 'frames', flush=True)
    manifest['status'] = 'rendered_and_decoded'
    save(archive / 'manifest.json', manifest)

    for entry in entries:
        directory = Path(entry['directory'])
        suffix = directory / 'episode-continuation.mp4'
        assert not suffix.exists()
        (directory / 'episode.mp4').rename(suffix)
        shutil.copy2(archive / entry['name'] / 'rendered-full.mp4', directory / 'episode.mp4')
        shutil.copy2(directory / 'episode.mp4', directory / 'episode-full.mp4')
        row = json.loads((directory / 'benchmark_result.json').read_text())
        row.update(frames=entry['total_frames'], continuation_frames=entry['continuation_frames'],
                   continuation_video=str(suffix), full_video=str(directory / 'episode.mp4'),
                   video_scope='full_episode', video_start_step=0, video_end_step=entry['total_frames'])
        save(directory / 'benchmark_result.json', row)
        save(directory / 'video-timeline.json', dict(version=1, video='episode.mp4',
             prefix_frames=entry['prefix_frames'], continuation_frames=entry['continuation_frames'],
             total_frames=entry['total_frames'], prefix_source=entry['prefix_source'], continuation_video=suffix.name,
             note='Video covers the complete episode. Saved states, tool events and raw trajectory in this directory cover only the continuation.'))
        verification = json.loads((directory / 'verification.json').read_text())
        verification['sha256']['episode-continuation.mp4'] = verification['sha256'].pop('episode.mp4')
        verification['sha256']['episode.mp4'] = entry['rendered_sha256']
        verification.update(recorded_video=suffix.name, recorded_video_scope='continuation suffix',
                            full_video_frames=entry['total_frames'], full_video_verification='full-video-verification.json')
        save(directory / 'verification.json', verification)
        save(directory / 'full-video-verification.json', dict(path=str(directory / 'episode.mp4'),
             alternate_path=str(directory / 'episode-full.mp4'), frames=entry['total_frames'], fps=entry['fps'],
             all_frames_decoded=True, source_prefix=str(Path(entry['prefix_source']) / 'episode.mp4'),
             continuation_suffix=str(suffix), sha256=entry['rendered_sha256'],
             source_trajectories=[entry['prefix_trajectory'], entry['continuation_trajectory']],
             boundary_images_exact=True, sample_rgb_mean_absolute_errors=entry['sample_rgb_mean_absolute_errors'],
             role='full episode rendered from original lossless recorded RGB; native trajectories and suffix video preserved',
             manifest=str(archive / 'manifest.json')))
        report = parse_run(directory)
        assert report['warnings'] == [], (directory, report['warnings'])
        assert report['video']['frames'] == entry['total_frames']
        assert sum(e.get('video_range', {}).get('frames', 0) for e in report['events']) == entry['total_frames']
        actions = [e for e in report['events'] if e.get('action')]
        assert sum(e['video_range']['frames'] for e in actions) == entry['continuation_frames']
        assert min(e['video_range']['start_s'] for e in actions) == entry['prefix_frames'] / entry['fps']
        check = json.loads((directory / 'artifact-completeness.json').read_text())
        check.update(video_frames=entry['total_frames'], continuation_frames=entry['continuation_frames'],
                     video_scope='full_episode', recorded_video=str(suffix))
        check['viewer'].update(prefix_clip_frames=entry['prefix_frames'], covered_video_frames=entry['total_frames'], warnings=[])
        check['recipe_note'] = 'Recipes and raw trajectory retain continuation scope; video-timeline.json maps events into the full default video.'
        check['required_files'] = list(dict.fromkeys(check['required_files'] + ['episode-continuation.mp4', 'video-timeline.json', 'full-video-verification.json']))
        save(directory / 'artifact-completeness.json', check)
        hashes = fingerprint(directory)
        assert hashes['episode-continuation.mp4'] == entry['original_recorded_hashes']['episode.mp4']
        for name, expected in entry['original_recorded_hashes'].items():
            if name != 'episode.mp4': assert hashes[name] == expected
        assert fingerprint(Path(entry['prefix_source'])) == entry['prefix_recorded_hashes']

    checks, rows = [], []
    for run in discover_runs(target):
        directory = Path(run['path'])
        row = json.loads((directory / 'benchmark_result.json').read_text())
        report = parse_run(directory)
        assert not report['warnings'], (directory, report['warnings'])
        expected = row.get('total_episode_actions', row['actions'])
        assert report['video']['frames'] == row['frames'] == expected
        assert report['status'] == ('success' if row['success'] else 'failure')
        assert sum(e.get('video_range', {}).get('frames', 0) for e in report['events']) == expected
        rows.append(row)
        checks.append(dict(name=directory.name, video_frames=expected, warnings=[]))
    assert len(rows) == 120 and sum(r['success'] for r in rows) == 63
    summary_path = target / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(results=rows, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), full_video_repair_manifest=str(archive / 'manifest.json'))
    save(summary_path, summary)
    with (target / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        fields = csv.DictReader(stream).fieldnames
    fields += [name for name in ['frames', 'video_scope', 'continuation_frames', 'continuation_video', 'full_video'] if name not in fields]
    with (target / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    readme = target / 'README.md'
    text = readme.read_text().replace('续跑项 episode.mp4 和轨迹对应追加段，episode-full.mp4 是完整连续视频。',
        '续跑项的 episode.mp4 已覆盖为从第 0 步到实际结束的完整视频，episode-full.mp4 内容相同；原续跑段保留为 episode-continuation.mp4。原始轨迹、状态和工具日志仍对应续跑段，video-timeline.json 记录完整视频中的时间偏移。')
    text += f'\n3 项续跑视频修复、旧视频备份及核验记录：[{archive.name}]({archive})。\n'
    readme.write_text(text)
    metadata = target / 'consolidation.json'
    consolidation = json.loads(metadata.read_text())
    consolidation.pop('live_viewer_verified', None)
    consolidation.pop('live_verification', None)
    consolidation.update(full_video_repair_manifest=str(archive / 'manifest.json'), viewer_parse_verified=120,
                         all_default_videos_cover_complete_episodes=True)
    save(metadata, consolidation)
    manifest.update(status='verified', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    complete_default_videos=120, warnings_empty=True, raw_recordings_unchanged=True, checks=checks)
    save(archive / 'manifest.json', manifest)
    print(json.dumps(dict(status='verified', repaired=3, complete_default_videos=120, archive=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
