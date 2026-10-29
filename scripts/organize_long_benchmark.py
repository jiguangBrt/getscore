"""Promote completed cases into outputs/ and audit the reference artifact layout."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import tomllib


FORMAT_VERSION=4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_viewer.naming import canonical_name, case_directories


def normalize_events(source,planner):
    """Use the viewer's event envelope while retaining the true backend origin."""
    events=[]
    if planner=='none':
        p=source/'pure_policy_events.jsonl'
        for i,line in enumerate(p.read_text().splitlines()):
            e=json.loads(line)
            events.append((None,dict(id=f'pure-{i}',type='mcpToolCall',tool=e['tool'],
                arguments=e['arguments'],result=e['result'],status='completed'),p.name,i+1))
    elif planner=='kimi':
        p=source/'kimi_tool_calls.jsonl'
        for i,line in enumerate(p.read_text().splitlines()):
            e=json.loads(line)
            events.append((e['time_unix']*1000,dict(id=f'kimi-tool-{i}',type='mcpToolCall',
                tool=e['tool'],arguments=e['arguments'],result=e['result'],
                status='failed' if e.get('is_error') or (isinstance(e['result'],dict) and e['result'].get('error')) else 'completed'),p.name,i+1))
        p=source/'kimi_acp.stream.jsonl'
        raw=[json.loads(line) for line in p.read_text().splitlines()]
        # session/load replays historical messages. Only the new prompt's
        # responses belong to this continuation's timeline.
        prompt_times={}
        for e in raw:
            if e.get('direction')=='client' and e.get('message',{}).get('method')=='session/prompt':
                prompt_times.setdefault(e.get('recording_segment','single'),e['time_unix'])
        for i,e in enumerate(raw):
            prompt_at=prompt_times.get(e.get('recording_segment','single'))
            if prompt_at is not None and e['time_unix']<prompt_at:continue
            m=e.get('message',{});u=m.get('params',{}).get('update',{})
            c=u.get('content',{})
            if m.get('method')=='session/update' and u.get('sessionUpdate')=='agent_message_chunk' and c.get('type')=='text':
                events.append((e['time_unix']*1000,dict(id=f'kimi-text-{i}',type='agentMessage',
                    text=c['text']),p.name,i+1))
        events.sort(key=lambda e:e[0])
        merged=[]
        for event in events:
            if merged and event[1]['type']=='agentMessage' and merged[-1][1]['type']=='agentMessage':
                merged[-1][1]['text']+=event[1]['text']
            else:merged.append(event)
        events=merged
    else:return None
    p=source/'00-planner.stream.jsonl'
    p.write_text(''.join(json.dumps(dict(method='item/completed',origin=planner,
        normalization='Viewer-compatible envelope; source files are preserved unchanged.',
        source=dict(file=file,line=line),payload=dict(completed_at_ms=at,item=item)),ensure_ascii=False,default=str)+'\n'
        for at,item,file,line in events))
    return p.name


def write(path, value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(path)


def package_case(source, batch, reference):
    """Write the original benchmark's derived viewer artifacts without moving a run.

    Logs, transcripts, states, raw trajectories, and videos remain unchanged.
    """
    source=Path(source).resolve();batch=Path(batch).resolve();reference=Path(reference).resolve()
    config_path=source/'configuration.json'
    config=json.loads(config_path.read_text())
    required_dirs=[p.name for p in reference.iterdir() if p.is_dir()]
    result=json.loads((source/'benchmark_result.json').read_text())
    transcripts=list(source.glob('transcript_*.json'))
    if len(transcripts)!=1:raise RuntimeError(f'{source}: transcript count')
    transcript=json.loads(transcripts[0].read_text())
    bridge_config=tomllib.loads((source/'bridge.toml').read_text())
    planner=config['planner'];backend='gawm' if config['policy']=='gawm' else 'openpi'
    if planner=='none':config['model']=transcript.get('model','pure-policy')
    config.update(policy_backend=backend,reasoning_enabled=planner!='none',
        auth={'none':'not applicable: pure policy','gpt6':'existing ChatGPT login','kimi':'existing Kimi Code login'}[planner],
        api_key_used=False,mode='sync',planner_finish=transcript.get('finish'),
        planner_error=transcript.get('error'),stats=transcript.get('stats',{}),
        limits={k:config[k] for k in ['max_turns','planner_timeout_s','max_episode_steps']},
        memory_dir=config.get('memory_dir',str(batch/'empty-memory')),
        config={**bridge_config,'policy_backend':backend,
            'simulation':dict(suite=config['suite'],task=config['task'],seed=0,
                initial_state_index=0,max_episode_steps=config['max_episode_steps'],
                execute_horizon=config['policy_horizon'],cuda_device=0,sam3_endpoint='http://192.168.50.73:8114'),
            'planner':dict(backend=planner,model=config['model'],reasoning_effort=config['reasoning_effort'])})
    write(config_path,config)
    normalized=normalize_events(source,planner)
    tag=transcripts[0].stem.removeprefix('transcript_')
    states=json.loads((source/'states.json').read_text())['steps']
    import numpy as np
    episode_path=Path(result['trajectory'])
    episode=json.loads((episode_path/'episode.json').read_text())
    commands=[s for s in states if s.get('command')]
    if [s['command']['action'] for s in commands]!=episode['primitive_names']:
        raise RuntimeError(f'{source}: raw primitives do not align with state commands')
    with np.load(episode_path/'transitions.npz') as data:
        ids,counts=np.unique(data['primitive_id'],return_counts=True)
        if (ids<0).any():raise RuntimeError(f'{source}: native actions lack primitive ids')
        frames_by_id=dict(zip(ids.tolist(),counts.tolist()))
    frame_counts={str(s['step_idx']):dict(action=s['command']['action'],frames=frames_by_id.get(i,0)) for i,s in enumerate(commands)}
    recorded_frames=result.get('continuation_frames',result['frames'])
    if sum(e['frames'] for e in frame_counts.values())!=recorded_frames:raise RuntimeError(f'{source}: action-frame mismatch')
    write(source/'native-action-counts.json',dict(total_frames=recorded_frames,steps=frame_counts,
        source='trajectory transitions.npz primitive_id; matched against episode.json primitive_names and states.json commands'))
    audit=source/f'{tag}.json'
    if not audit.exists():
        policy_steps=[s for s in states if (s.get('command') or {}).get('action') in ['pi0_pick','pi0_doubled']]
        write(audit,dict(suite=result['suite'],task_id=result['task'],seed=0,
            regime='pure_policy' if result['planner']=='none' else 'strict_perception',
            strategy_notes=dict(generated_by='artifact packaging from recorded state trace; no new model inference',
                task_language=result['policy_prompt'],planner=result['planner'],policy=result['policy'],
                single_episode=True,resets=0,actions=result['actions'],policy_queries=result['policy_queries'],
                finish=transcript.get('finish'),error=transcript.get('error'),
                command_trace='command_trace.jsonl',transcript=transcripts[0].name),
            pick_result=policy_steps[-1].get('result') if policy_steps else None,
            final_state=states[-1]['state'],terminated=result['success']))
    trace=source/'command_trace.jsonl'
    if not trace.exists():
        trace.write_text(''.join(json.dumps(dict(step=s.get('step_idx',i),command=s.get('command'),
            result=s.get('result'),terminated=s.get('terminated')),ensure_ascii=False,default=str)+'\n'
            for i,s in enumerate(states) if s.get('command')))
    required_files=['episode.mp4','configuration.json','console.log','run.log','env_server.log',
        'bridge.log','network-timing.jsonl','states.json',transcripts[0].name,audit.name,
        'benchmark_result.json','verification.json','command_trace.jsonl','native-action-counts.json']
    if result['success']:required_files.append(f'{tag}_recipe.jsonl')
    if result['planner']=='gpt6':
        required_files.extend([f'codex_{tag}.txt',f'codex_{tag}.txt.last',f'codex_{tag}.txt.stream.jsonl'])
    elif result['planner']=='kimi':
        required_files.extend(['kimi.txt','kimi_acp.stream.jsonl','kimi_session.json','kimi_tool_calls.jsonl'])
    else:required_files.extend(['pure_policy_events.jsonl','pure_policy_prompt.txt'])
    if normalized:required_files.append(normalized)
    missing=[n for n in required_files if not (source/n).is_file()]
    missing.extend(n for n in required_dirs if not (source/n).is_dir() or not any((source/n).iterdir()))
    if missing:raise RuntimeError(f'{source.name}: missing required artifacts {missing}')
    return dict(result=result,backend=backend,required_files=required_files,required_dirs=required_dirs)


def organize(batch):
    root=batch.parent
    reference=root/'_archive/runs/liberopro-pi05-gpt6astra-20261020-104215'
    if not reference.is_dir():reference=root/'liberopro-pi05-gpt6astra-20261020-104215'
    required_dirs=[p.name for p in reference.iterdir() if p.is_dir()]
    rows=[]
    for source in case_directories(batch):
        config_path=source/'configuration.json'
        if not config_path.exists():continue
        config=json.loads(config_path.read_text())
        if config.get('run_status')!='finished' or not (source/'benchmark_result.json').exists():continue
        prior=json.loads((source/'artifact-completeness.json').read_text()) if (source/'artifact-completeness.json').exists() else {}
        if source.is_symlink() and prior.get('format_version')==FORMAT_VERSION:
            destination=root/canonical_name(source.name)
            rows.append(dict(name=destination.name,directory=str(destination),video=str(destination/'episode.mp4'),complete=True))
            continue
        packaged=package_case(source,batch,reference)
        result=packaged['result'];backend=packaged['backend'];required_files=packaged['required_files']
        destination=root/canonical_name(source.name)
        if not source.is_symlink():
            if destination.exists():raise FileExistsError(destination)
            source.rename(destination)
            source.symlink_to(Path('../..')/destination.name,target_is_directory=True)
        elif source.resolve()!=destination.resolve():raise RuntimeError(f'Unexpected case link: {source}')
        old=str(source);new=str(destination)
        for key in ['directory','video','trajectory','transcript']:
            if result.get(key,'').startswith(old):result[key]=new+result[key][len(old):]
        write(destination/'benchmark_result.json',result)
        from run_viewer.parser import parse_run
        parsed=parse_run(destination)
        if parsed['video'] is None or not parsed['observations']:raise RuntimeError(f'{destination}: viewer cannot load video/observations')
        if parsed['backend']!=backend:raise RuntimeError(f'{destination}: viewer backend mismatch')
        check=dict(complete=True,format_version=FORMAT_VERSION,reference_directory=str(reference),directory=str(destination),
            required_files=required_files,reference_observation_directories=required_dirs,
            video_frames=result['frames'],native_actions=result['actions'],
            recipe_expected=result['success'],
            recipe_note='RPent exports replay recipes only for native successes; failed command sequences remain in states.json and command_trace.jsonl.',
            raw_trajectory=result['trajectory'],viewer=dict(video_loaded=True,observations=len(parsed['observations']),
                tool_events=parsed['metrics']['tool_calls'],backend=parsed['backend'],warnings=parsed['warnings']))
        write(destination/'artifact-completeness.json',check)
        rows.append(dict(name=destination.name,directory=str(destination),video=str(destination/'episode.mp4'),complete=True))
    write(batch/'output-layout-audit.json',dict(checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        completed_cases=len(rows),all_completed_cases_have_video=True,all_completed_cases_match_reference=True,
        cases=rows,incomplete_cases_remain_under=str(batch/'cases'),
        compatibility_note='cases/ entries for completed cases are symlinks; original recorded paths remain readable.'))
    return len(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--watch-pid',type=int)
    a=p.parse_args();batch=a.batch.resolve();previous=-1
    while True:
        with (batch/'.organize.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            count=organize(batch)
        if count!=previous:
            print(f'Organized {count} completed cases into {batch.parent}',flush=True);previous=count
            report=subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_long_benchmark.py')),str(batch)],capture_output=True,text=True)
            (batch/'current-report-build.log').write_text(report.stdout+report.stderr)
            if report.returncode:raise RuntimeError('Live report audit failed; see current-report-build.log')
        if not a.watch_pid:break
        try:
            if Path(f'/proc/{a.watch_pid}/stat').read_text().split()[2]=='Z':break
        except FileNotFoundError:break
        time.sleep(3)


if __name__=='__main__':main()
