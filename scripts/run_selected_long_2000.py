"""Run the screenshot-selected cases, Kimi first, without changing original runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from scripts import run_long_benchmark as runner


def finalize_result(output,source):
    """Preserve a completed native episode even if planner shutdown timed out."""
    import numpy as np
    result_path=output/'benchmark_result.json'
    if not result_path.exists():return False
    row=json.loads(result_path.read_text())
    config=json.loads((output/'configuration.json').read_text())
    recovery=json.loads((output/'recovery.json').read_text())
    prefix=int(recovery['resumed_steps'])
    episode=next(output.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*'))
    with np.load(episode/'transitions.npz') as data:
        count=len(data['actions']);native=bool(data['terminated'].any())
        truncated=bool(data['truncated'][-1]) if count else False
    if prefix+count>2000:raise RuntimeError('Episode exceeded 2000 actions')
    error=str(row.get('error') or config.get('error') or '')
    if error:
        # A wall timeout below the action cap remains an infrastructure interruption.
        # Terminal native success or the complete 2000-step budget is authoritative.
        terminal=native or (prefix+count==2000 and truncated)
        if not terminal or not error.startswith('TimeoutError: Kimi ACP session/prompt exceeded '):return False
        verification=json.loads((output/'verification.json').read_text())
        assert verification['frames_equal_actions'] and verification['raw_trajectory_valid']
        assert verification['frames']==count==row['actions']
        paths={p.name:p for p in [output/'episode.mp4',output/'states.json',
            *output.glob('transcript_*.json'),episode/'transitions.npz',episode/'proposals.npz']}
        for name,digest in verification['sha256'].items():
            if runner.sha(paths[name])!=digest:raise RuntimeError('Archived verification hash changed: '+name)
        row.update(error=None,planner_completion_error=error,
            outcome_basis='native environment terminal before planner wall-time cleanup')
        config.update(error=None,planner_completion_error=error)
    states=json.loads((output/'states.json').read_text())['steps']
    if native!=bool(row['success']) or native!=bool(states[-1]['terminated']):
        raise RuntimeError('Result disagrees with native termination')
    row.update(execution_mode=recovery['mode'],source_run=str(source),resumed_from_step=prefix,
        new_actions=count,total_episode_actions=prefix+count,target_total_steps=2000,
        recovery_verification=str(output/'recovery.json'),trajectory_scope='continuation suffix' if prefix else 'full fresh episode')
    runner.save(result_path,row)
    config.update(execution_mode=recovery['mode'],source_run=str(source),resumed_from_step=prefix,total_episode_actions=row['total_episode_actions'])
    runner.save(output/'configuration.json',config)
    from scripts.package_selected_long_2000 import prepare_case
    prepare_case(output, output.parent.parent)
    print(runner.now(),'VERIFIED',output.name,recovery['mode'],'success='+str(native),
        'new='+str(count),'total='+str(row['total_episode_actions']),flush=True)
    return True


def summary(batch,selection):
    rows=[]
    for item in selection['cases']:
        path=batch/'cases'/Path(item['source_run']).name/'benchmark_result.json'
        if path.exists():
            row=json.loads(path.read_text())
            if not row.get('error') and 'total_episode_actions' in row:rows.append(row)
    runner.save(batch/'selected-summary.json',dict(updated_at=runner.now(),expected=13,completed=len(rows),
        successes=sum(bool(r['success']) for r in rows),resumed=sum(r.get('execution_mode')=='resume' for r in rows),results=rows))
    lines=['2000步追加实验',f'更新时间：{runner.now()}',f'已完成 {len(rows)}/13，成功 {sum(bool(r["success"]) for r in rows)}。',
        '仅处理用户截图的空白格；先Kimi，再GPT；原始120项结果不变。','',
        '| 实验 | 执行方式 | 原生成功 | 本次动作 | episode累计动作 |','|---|---|---|---:|---:|']
    for r in rows:lines.append(f'| {r["name"]} | {r.get("execution_mode","pending verification")} | {r["success"]} | {r["actions"]} | {r.get("total_episode_actions",r["actions"])} |')
    (batch/'report.md').write_text('\n'.join(lines)+'\n')
    return rows


def run_one(batch,item,env,stop):
    if stop.is_set():return
    source=Path(item['source_run']);output=batch/'cases'/source.name
    if (output/'benchmark_result.json').exists():
        prior=json.loads((output/'benchmark_result.json').read_text())
        if not prior.get('error') and 'total_episode_actions' in prior:return
        if finalize_result(output,source):return
        archive=batch/'incomplete'/f'{source.name}-invalid-result-{int(time.time())}'
        archive.parent.mkdir(exist_ok=True);output.rename(archive)
    original=json.loads((source/'configuration.json').read_text())
    case={k:original[k] for k in ('variant','suite','task','name','policy_prompt','planner')}
    args=SimpleNamespace(max_steps=2000,max_turns=160,timeout=7200 if item['planner']=='kimi' else 2400,
        entrypoint=ROOT/'scripts/long_recovery_entrypoint.py',recovery_source=source,recovery_disabled=False)
    for attempt in range(2):
        runner.run_case(batch,case,item['policy'],env,args)
        result_path=output/'benchmark_result.json'
        config=json.loads((output/'configuration.json').read_text())
        error=str(config.get('error') or '')
        if result_path.exists():
            row=json.loads(result_path.read_text());error=str(row.get('error') or error)
        else:row=None
        if row is not None and finalize_result(output,source):return
        if 'RECOVERY_SESSION_UNAVAILABLE' in error and attempt==0:
            archive=batch/'incomplete'/f'{source.name}-session-restore-{int(time.time())}'
            archive.parent.mkdir(exist_ok=True);output.rename(archive)
            args.recovery_disabled=True
            continue
        if row is None or error:
            stop.set()
            raise RuntimeError(f'{source.name}: infrastructure/planner error, not a benchmark failure: {error}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=ROOT/'outputs/selected-original-2000step-20261026')
    args=parser.parse_args();batch=args.batch.resolve()
    with (batch/'.execution.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(__import__('os').getpid()));lock.flush()
        selection=json.loads((batch/'selection.json').read_text())
        assert len(selection['cases'])==13
        (batch/'cases').mkdir(exist_ok=True);(batch/'empty-memory/global').mkdir(parents=True,exist_ok=True)
        for p in [batch/'empty-memory/MEMORY.md',batch/'empty-memory/global/GLOBAL_MEMORY.md']:
            p.write_text('No task-specific or episode-specific external memory is provided.\n')
        env=runner.environment();stop=threading.Event()
        selection.update(status='running',started_at=selection.get('started_at',runner.now()),resumed_at=runner.now())
        selection.pop('error',None)
        runner.save(batch/'selection.json',selection)
        try:
            for planner in ('kimi','gpt6'):
                def worker(policy):
                    for item in selection['cases']:
                        if item['planner']==planner and item['policy']==policy:
                            run_one(batch,item,env,stop)
                            summary(batch,selection)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(worker,policy) for policy in ('pi05','gawm')]
                    for future in futures:future.result()
                if stop.is_set():break
            rows=summary(batch,selection)
            runner.refresh(batch)
            selection.update(status='complete' if len(rows)==13 else 'incomplete',completed=len(rows),finished_at=runner.now())
            runner.save(batch/'selection.json',selection)
            print(runner.now(),'SELECTED BATCH',selection['status'],len(rows),flush=True)
        except Exception as exc:
            summary(batch,selection)
            selection.update(status='requires_infrastructure_review',error=str(exc),updated_at=runner.now())
            runner.save(batch/'selection.json',selection)
            raise


if __name__=='__main__':main()
