"""Apply the original 120-run viewer layout to the 13 selected long runs."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.organize_long_benchmark import FORMAT_VERSION, package_case, write
from run_viewer.parser import parse_run


def recorded_hashes(source):
    """Check immutable recorded evidence against the original verification."""
    verification = json.loads((source / 'verification.json').read_text())
    for name, expected in verification['sha256'].items():
        files = [source / name] if (source / name).is_file() else list((source / 'trajectory').rglob(name))
        if len(files) != 1:
            raise RuntimeError(f'{source.name}: ambiguous/missing recorded artifact {name}')
        digest = hashlib.sha256()
        with files[0].open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f'{source.name}: recorded artifact changed: {name}')
    return verification['sha256']


def prepare_case(source, batch):
    source = Path(source).resolve()
    batch = Path(batch).resolve()
    config_path = source / 'configuration.json'
    original_config = config_path.read_text()
    reference = ROOT / 'outputs/_archive/runs/liberopro-pi05-gpt6astra-20261020-104215'
    recorded_hashes(source)
    backup = ROOT / 'outputs/_archive/records/viewer-packaging' / batch.name / source.name / 'configuration.json'
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(original_config)
    packaged = package_case(source, batch, reference)
    report = parse_run(source)
    row = packaged['result']
    commands = [state for state in json.loads((source / 'states.json').read_text())['steps'] if state.get('command')]
    actions = [event for event in report['events'] if event.get('action')]
    clips = [event['video_range'] for event in actions if 'video_range' in event]
    expected_status = 'success' if row['success'] else 'failure'
    if report['warnings'] or report['backend'] != packaged['backend'] or report['status'] != expected_status:
        raise RuntimeError(f'{source.name}: viewer interpretation failed: {report["warnings"]}, {report["status"]}')
    if not report['video'] or not report['observations'] or not report['metrics']['tool_calls']:
        raise RuntimeError(f'{source.name}: missing viewer video/observations/tool events')
    if len(actions) != len(commands) or len(clips) != len(commands) or sum(clip['frames'] for clip in clips) != row['frames']:
        raise RuntimeError(f'{source.name}: action clips do not cover the recorded video exactly')
    if report['metrics']['recorded_tool_calls'] is not None and report['metrics']['tool_calls'] != report['metrics']['recorded_tool_calls']:
        raise RuntimeError(f'{source.name}: normalized tools disagree with planner tool count')
    expected_steps = [state['step_idx'] for state in commands]
    if sorted(event['after_state'] for event in actions) != sorted(expected_steps):
        raise RuntimeError(f'{source.name}: action observations are not matched exactly once')
    if any(not obs['images'].get('agentview') or not obs['images'].get('wrist') for obs in report['observations'].values()):
        raise RuntimeError(f'{source.name}: missing observation camera images')
    recorded_hashes(source)
    check = dict(
        complete=True, format_version=FORMAT_VERSION, reference_directory=str(reference), directory=str(source),
        required_files=packaged['required_files'], reference_observation_directories=packaged['required_dirs'],
        video_frames=row['frames'], native_actions=row['actions'], total_episode_actions=row['total_episode_actions'],
        execution_mode=row['execution_mode'], trajectory_scope=row['trajectory_scope'],
        recipe_expected=row['success'],
        recipe_note='Recipes exist only for native successes. A resumed run retains its suffix trace and separate full replay.',
        raw_trajectory=row['trajectory'], recorded_artifact_hashes_preserved=True,
        viewer=dict(video_loaded=True, observations=len(report['observations']), tool_events=report['metrics']['tool_calls'],
                    action_events=len(actions), action_clip_frames=sum(clip['frames'] for clip in clips),
                    recovery_action_events=sum(bool(event.get('recovery_prefix')) for event in actions),
                    backend=report['backend'], warnings=report['warnings']),
    )
    write(source / 'artifact-completeness.json', check)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    latest = ROOT / 'outputs/long120_1027'
    parser.add_argument('--batch', type=Path, default=latest if latest.is_dir() else ROOT / 'outputs/selected-original-2000step-20261026')
    args = parser.parse_args()
    batch = args.batch.resolve()
    if (batch / 'selection.json').exists():
        selection = json.loads((batch / 'selection.json').read_text())
        sources = [batch / 'cases' / Path(item['source_run']).name for item in selection['cases']]
    else:
        sources = [p for p in batch.iterdir() if p.is_dir() and (p / 'configuration.json').is_file()
                   and json.loads((p / 'configuration.json').read_text()).get('execution_mode') in {'resume', 'restart'}]
    assert len(sources) == 13
    checks = []
    for source in sorted(sources):
        checks.append(prepare_case(source, batch))
        print(f'PASS {source.name}: tools={checks[-1]["viewer"]["tool_events"]}, frames={checks[-1]["video_frames"]}', flush=True)
    write(batch / 'output-layout-audit.json', dict(
        checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        completed_cases=len(checks), all_completed_cases_have_video=True,
        all_completed_cases_match_reference=True, all_viewer_warnings_empty=True,
        all_raw_artifact_hashes_preserved=True, cases=checks,
    ))


if __name__ == '__main__':
    main()
