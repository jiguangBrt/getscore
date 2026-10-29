"""Run 80 paired Long cases and conditionally 40 Kimi cases, with resumable results."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from run_viewer.naming import case_directories, experiment_name
PYTHON=ROOT/'.venv/bin/python'
PRO=ROOT/'data/liberopro-runtime/liberopro/liberopro'
TZ=ZoneInfo('Asia/Shanghai')
LOCK=threading.Lock()


def now():return datetime.now(TZ).isoformat(timespec='seconds')
def save(path,value):
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    temporary.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def inference_services(path=None):
    if path is None:
        return dict(openpi=dict(host='192.168.50.73',port=8008),
                    gawm=dict(host='192.168.50.73',port=8115),
                    simulation=dict(sam3_endpoint='http://192.168.50.73:8114'))
    return tomllib.loads(Path(path).read_text())


def environment():
    env={**os.environ}
    for key in ['PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_PREFIX','OPENAI_API_KEY','OPENAI_BASE_URL','CODEX_API_KEY','CODEX_BASE_URL','LIBERO_SUFFIX','LIBERO_PERTURBATION']:
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_TYPE='pro',
        LIBERO_PRO_ASSET_PATH=str(ROOT/'data/libero-pro/assets'),LIBERO_ASSET_PATH=str(ROOT/'data/libero-pro/assets'),
        LIBERO_CONFIG_PATH=str(ROOT/'data/libero-pro/config'),LIBERO_PRO_CONFIG_PATH=str(ROOT/'data/libero-pro/config-pro'),
        RPENT_RLINF_ROOT=str(ROOT/'data/rlinf-runtime'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    from codex_cli_bin import bundled_codex_path
    env['CODEX_BIN']=str(bundled_codex_path())
    for key in ['NO_PROXY','no_proxy']:
        env[key]=','.join(filter(None,[env.get(key),'127.0.0.1','localhost','192.168.50.73']))
    return env


def cases(planners):
    mapping=ast.literal_eval(ast.parse((PRO/'benchmark/libero_suite_task_map.py').read_text()).body[0].value)
    for planner in planners:
        for variant,suite in [('original','libero_10'),('swap','libero_10_swap')]:
            for task,name in enumerate(mapping['libero_10']):
                canonical=name.split('SCENE',1)[1].split('_',1)[1].replace('_',' ')
                yield dict(variant=variant,suite=suite,task=task,name=name,policy_prompt=canonical,planner=planner)


def refresh(batch):
    with LOCK, (batch/'.organize.lock').open('a') as filesystem_lock:
        fcntl.flock(filesystem_lock,fcntl.LOCK_EX)
        results=[]
        for directory in case_directories(batch):
            p=directory/'benchmark_result.json'
            if not p.is_file():continue
            row=json.loads(p.read_text());results.append(row)
        groups={}
        for row in results:
            key=f"{row['variant']}-{row['planner']}-{row['policy']}"
            group=groups.setdefault(key,dict(successes=0,total=0,errors=0))
            group['total']+=1;group['successes']+=int(row['success']);group['errors']+=int(bool(row.get('error')))
        for g in groups.values():g['success_rate']=g['successes']/g['total']
        summary=dict(updated_at=now(),completed=len(results),groups=groups,results=results)
        save(batch/'summary.json',summary)
        csv_temporary=batch/f'results.csv.{os.getpid()}.tmp'
        with csv_temporary.open('w',encoding='utf-8-sig',newline='') as f:
            fields=['name','variant','suite','task','planner','policy','success','terminated','truncated','actions','policy_queries','elapsed_s','error','video','directory']
            w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(results)
        csv_temporary.replace(batch/'results.csv')
        return summary


def verify_case(output,case,policy,elapsed,returncode):
    import numpy as np
    import imageio.v2 as imageio
    transcripts=list(output.glob('transcript_*.json'))
    if len(transcripts)!=1:raise RuntimeError('Expected exactly one RPent transcript')
    transcript=json.loads(transcripts[0].read_text())
    planner_error=str(transcript.get('error') or '')
    states=json.loads((output/'states.json').read_text())['steps']
    if 'Kimi provider infrastructure interruption' in planner_error and not states[-1]['terminated']:
        raise RuntimeError(planner_error)
    if ('usage limit' in planner_error.lower() or 'usageLimitExceeded' in planner_error) and not states[-1]['terminated']:
        raise RuntimeError('Planner infrastructure interruption: '+planner_error)
    episodes=list((output/'trajectory').glob('raw/libero/*/task_*/seed_*/episode_*/episode.json'))
    if len(episodes)!=1:raise RuntimeError('Expected exactly one complete raw trajectory')
    episode_path=episodes[0].parent
    from robots.libero.flywheel import LIBERO_SPEC
    from rpent.flywheel.episode import validate_episode
    episode=validate_episode(episode_path,spec=LIBERO_SPEC)
    with np.load(episode_path/'transitions.npz') as data:
        count=len(data['actions']);native=bool(data['terminated'].any());truncated=bool(data['truncated'].any())
        if native!=bool(states[-1]['terminated']):raise RuntimeError('Native success disagrees with state trace')
        initial_hash=hashlib.sha256(data['states'][0].tobytes()+data['main_images'][0].tobytes()).hexdigest()
    video=output/'episode.mp4'
    with imageio.get_reader(video) as reader:
        frames=sum(1 for _ in reader);fps=reader.get_meta_data()['fps']
    if frames!=count:raise RuntimeError(f'Video frames {frames} != action count {count}')
    row=dict(**{**case,'name':output.name},policy=policy,success=native,terminated=native,truncated=truncated,
        actions=count,policy_queries=episode['proposal_count'],elapsed_s=round(elapsed,2),
        error=transcript.get('error'),exit_code=returncode,video=str(video),directory=str(output),
        frames=frames,fps=fps,initial_observation_sha256=initial_hash,trajectory=str(episode_path),
        transcript=str(transcripts[0]),finished_at=now())
    save(output/'verification.json',dict(success_agrees_with_native=True,frames_equal_actions=True,
        raw_trajectory_valid=True,frames=frames,actions=count,sha256={p.name:sha(p) for p in [video,transcripts[0],output/'states.json',episode_path/'transitions.npz',episode_path/'proposals.npz']}))
    save(output/'benchmark_result.json',row)
    return row


def run_case(batch,case,policy,env,args):
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.rpc.http_rpc import HttpRpcClient
    name=experiment_name(f"long-{case['variant']}",case['task'],case['planner'],policy)
    output=batch/'cases'/name
    if (output/'benchmark_result.json').exists():return
    if output.exists():
        archive=batch/'incomplete'/f'{name}-{int(time.time())}'
        archive.parent.mkdir(exist_ok=True);output.rename(archive)
    output.mkdir(parents=True)
    port=free_port();endpoint=f'http://127.0.0.1:{port}'
    backend='openpi' if policy=='pi05' else 'gawm'
    horizon=5 if policy=='pi05' else 8
    rpc_timeout=330 if policy=='pi05' else 150
    request_timeout=300 if policy=='pi05' else 120
    services=inference_services(getattr(args,'inference_config',None))
    policy_service=services[backend]
    config=output/'bridge.toml'
    config.write_text(f'[{backend}]\nhost = {json.dumps(policy_service["host"])}\nport = {int(policy_service["port"])}\nconnect_timeout_s = 20.0\nrequest_timeout_s = {request_timeout}.0\n'+
        ('execute_horizon = 8\n' if policy=='gawm' else '')+
        f'\n[bridge]\nhost = "127.0.0.1"\nport = {port}\nrpc_timeout_s = {rpc_timeout}.0\n')
    command=[str(PYTHON),str(getattr(args,'entrypoint',ROOT/'scripts/long_benchmark_entrypoint.py')),'--robot','libero','--libero-type','pro',
        '--suite',case['suite'],'--task',str(case['task']),'--seed','0','--cuda-device','0',
        '--max-episode-steps',str(args.max_steps),'--vla-endpoint',endpoint,'--vla-timeout-s',str(rpc_timeout),
        '--sam3-endpoint',services['simulation']['sam3_endpoint'],'--planner',{'gpt6':'codex','none':'none','kimi':'kimi'}[case['planner']],
        '--model',{'gpt6':'gpt-6-astra','none':'pure-policy','kimi':'kimi-code/k3.1-pro'}[case['planner']],
        '--reasoning-effort',{'gpt6':'low','none':'none','kimi':'high'}[case['planner']],
        '--max-turns',str(args.max_turns),'--planner-timeout-s',str(args.timeout),
        '--memory-profile','local','--memory-dir',str(batch/'empty-memory'),
        '--collect-flywheel-data','--flywheel-root',str(output/'trajectory'),'--output-dir',str(output)]
    run_env={**env,'LONG_POLICY_BACKEND':policy,'LONG_POLICY_PROMPT':case['policy_prompt'],
        'LONG_POLICY_HORIZON':str(horizon),'LONG_MAX_STEPS':str(args.max_steps)}
    if getattr(args,'recovery_source',None):
        run_env.update(LONG_RECOVERY_SOURCE=str(args.recovery_source),LONG_RECOVERY_OUTPUT=str(output),
            LONG_RECOVERY_DISABLED=str(int(getattr(args,'recovery_disabled',False))))
    manifest=dict(**case,policy=policy,seed=0,initial_state_index=0,model={'gpt6':'gpt-6-astra','none':None,'kimi':'kimi-code/k3.1-pro'}[case['planner']],
        reasoning_effort={'gpt6':'low','none':None,'kimi':'high'}[case['planner']],command=command,
        run_status='starting',started_at=now(),max_episode_steps=args.max_steps,max_turns=args.max_turns,planner_timeout_s=args.timeout,
        memory='empty, read-only, no per-task external memory',policy_horizon=horizon,task_success=None)
    if getattr(args,'inference_config',None):
        manifest['inference_config']=dict(path=str(Path(args.inference_config).resolve()),sha256=sha(Path(args.inference_config)))
    for kind,ext in [('bddl_files','.bddl'),('init_files','.pruned_init')]:
        path=PRO/kind/case['suite']/(case['name']+ext)
        manifest[kind]=dict(path=str(path),sha256=sha(path))
    save(output/'configuration.json',manifest)
    print(now(),'START',name,flush=True)
    rpc=HttpRpcClient(endpoint);started=time.monotonic();runner=None
    with (output/'bridge.log').open('w') as log:
        bridge=subprocess.Popen([str(PYTHON),'-m',f'scripts.{backend}_rpent_bridge','--config',str(config),'--timing-log',str(output/'network-timing.jsonl')],
            cwd=ROOT,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            wait_for_ready(rpc,timeout_s=40)
            manifest['policy_status']=rpc.call('vla.status');manifest['run_status']='running';save(output/'configuration.json',manifest)
            with (output/'console.log').open('w') as console:
                runner=subprocess.Popen(command,cwd=ROOT/'RPent',env=run_env,stdin=subprocess.DEVNULL,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
                returncode=runner.wait(timeout=args.timeout+600 if case['planner']!='none' else 3600)
            result=verify_case(output,case,policy,time.monotonic()-started,returncode)
            manifest.update(run_status='finished',task_success=result['success'],error=result['error'],exit_code=returncode)
            print(now(),'DONE',name,'success='+str(result['success']),'actions='+str(result['actions']),flush=True)
        except Exception as exc:
            manifest.update(run_status='infrastructure_error',error=f'{type(exc).__name__}: {exc}')
            print(now(),'ERROR',name,manifest['error'],flush=True)
            save(output/'infrastructure_error.json',dict(error=manifest['error'],at=now()))
        finally:
            if runner and runner.poll() is None:
                os.killpg(runner.pid,signal.SIGTERM)
                try:runner.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(runner.pid,signal.SIGKILL)
            if bridge.poll() is None:
                try:rpc.call('shutdown',timeout_s=3)
                except Exception:os.killpg(bridge.pid,signal.SIGTERM)
            try:bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(bridge.pid,signal.SIGKILL);bridge.wait()
            manifest['finished_at']=now();save(output/'configuration.json',manifest)
            refresh(batch)


def worker(batch,policy,planners,env,args):
    for index,case in enumerate(cases(planners)):
        if args.pilot and index>0:break
        run_case(batch,case,policy,env,args)
        config=batch/'cases'/experiment_name(f"long-{case['variant']}",case['task'],case['planner'],policy)/'configuration.json'
        if config.exists() and 'usage limit' in str(json.loads(config.read_text()).get('error','')).lower():
            print(now(),'STOP',policy,'planner quota exhausted; remaining cases retained for resume',flush=True)
            break
        if config.exists() and 'Kimi provider infrastructure interruption' in str(json.loads(config.read_text()).get('error','')):
            print(now(),'STOP',policy,'Kimi provider interruption; remaining cases retained for resume',flush=True)
            break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--max-steps',type=int,default=1000)
    parser.add_argument('--max-turns',type=int,default=80);parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--inference-config',type=Path)
    parser.add_argument('--only-policy',choices=['pi05','gawm']);parser.add_argument('--only-planner',choices=['none','gpt6','kimi'])
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();batch=args.output.resolve();batch.mkdir(parents=True,exist_ok=True);(batch/'cases').mkdir(exist_ok=True)
    (batch/'empty-memory/global').mkdir(parents=True,exist_ok=True)
    for p in [batch/'empty-memory/MEMORY.md',batch/'empty-memory/global/GLOBAL_MEMORY.md']:
        p.write_text('No task-specific or episode-specific external memory is provided.\n')
    env=environment()
    previous_protocol=json.loads((batch/'protocol.json').read_text()) if (batch/'protocol.json').exists() else {}
    new_protocol=dict(seed=0,initial_state_index=0,suites=['libero_10','libero_10_swap'],
        base_planners=['none','gpt6'],policies=['pi05','gawm'],expected_base_cases=80,max_episode_steps=args.max_steps,
        max_turns=args.max_turns,planner_timeout_s=args.timeout,gpt_model='gpt-6-astra',gpt_reasoning='low',
        kimi_model='kimi-code/k3.1-pro',kimi_context=1048576,kimi_reasoning='high',
        kimi_trigger='GPT successes >60% of 40 OR absolute GPT-minus-none success-rate difference >=20 percentage points, pooled or within any matched scene/policy group',
        success_criterion='native environment termination',single_attempt=True,external_memory='empty',
        original_environment='PRO renderer base suite; original BDDL and init files verified byte-identical to standard checkout')
    save(batch/'protocol.json',{**previous_protocol,**new_protocol,'started_at':previous_protocol.get('started_at',now()),'resumed_at':now()})
    planners=[args.only_planner] if args.only_planner else ['none','gpt6']
    policies=[args.only_policy] if args.only_policy else ['pi05','gawm']
    with ThreadPoolExecutor(max_workers=len(policies)) as pool:
        futures=[pool.submit(worker,batch,policy,planners,env,args) for policy in policies]
        for future in futures:future.result()
    summary=refresh(batch)
    if args.only_planner or args.only_policy or args.pilot:return
    base=[r for r in summary['results'] if r['planner'] in ['none','gpt6']]
    if len(base)!=80:
        save(batch/'status.json',dict(status='incomplete',completed_base=len(base),expected=80,at=now()))
        return
    gpt=[r for r in base if r['planner']=='gpt6'];pure=[r for r in base if r['planner']=='none']
    gr=sum(r['success'] for r in gpt)/40;nr=sum(r['success'] for r in pure)/40
    paired_differences=[]
    for variant in ['original','swap']:
        for policy in ['pi05','gawm']:
            gg=[r for r in gpt if r['variant']==variant and r['policy']==policy]
            nn=[r for r in pure if r['variant']==variant and r['policy']==policy]
            gs=sum(r['success'] for r in gg);ns=sum(r['success'] for r in nn)
            paired_differences.append(dict(variant=variant,policy=policy,gpt_successes=gs,pure_successes=ns,total=10,difference=(gs-ns)/10))
    trigger=sum(r['success'] for r in gpt)>24 or abs(sum(r['success'] for r in gpt)-sum(r['success'] for r in pure))>=8 or any(abs(d['gpt_successes']-d['pure_successes'])>=2 for d in paired_differences)
    save(batch/'kimi_trigger.json',dict(triggered=trigger,gpt_success_rate=gr,pure_success_rate=nr,difference=gr-nr,matched_groups=paired_differences,at=now()))
    if trigger:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,batch,policy,['kimi'],env,args) for policy in ['pi05','gawm']]
            for future in futures:future.result()
    summary=refresh(batch)
    expected=120 if trigger else 80
    save(batch/'status.json',dict(status='complete' if summary['completed']==expected else 'incomplete',completed=summary['completed'],expected=expected,kimi_triggered=trigger,at=now()))
    report=['Long 原版 / Swap 单次评测报告',f'完成时间：{now()}',f'已完成：{summary["completed"]}/{expected}',
        '统一 seed 0、初始状态 0；1000 步动作上限；成功只看原生终止标记。',
        'GPT：gpt-6-astra / low；Kimi（触发时）：kimi-code/k3.1-pro / 1M / high。',
        '每个任务独立一次，无重置；外部任务记忆为空。','']
    for group,g in sorted(summary['groups'].items()):report.append(f'{group}: {g["successes"]}/{g["total"]} ({g["success_rate"]:.1%}), planner errors={g["errors"]}')
    report.extend(['',f'Kimi 追加条件触发：{trigger}',f'GPT 总成功率：{gr:.1%}；纯策略：{nr:.1%}；差值：{gr-nr:+.1%}',
        '逐任务结果见 results.csv，完整配置、日志、过程记录、原始动作轨迹及 episode.mp4 见 cases/。'])
    (batch/'report.txt').write_text('\n'.join(report)+'\n')
    print(now(),'BATCH FINISHED',summary['completed'],'/',expected,flush=True)


if __name__=='__main__':main()
