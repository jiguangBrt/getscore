"""Relay this authorized remote run's metrics to W&B; training stays server-side."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/rl_skills_v3_20261104/overnight'
HOST = '49.233.66.228'
REMOTE = '/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-overnight-20261104-to-20261105-0900'
HARD_DEADLINE = 1793840400
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', HOST]


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def snapshot(after):
    code = '''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);after=int(sys.argv[2])
def read(name,default=None):
 try:return json.loads((p/name).read_text())
 except (OSError,ValueError):return default
rows=[]
if (p/'metrics.jsonl').exists():
 for line in (p/'metrics.jsonl').read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r['update']>after:
   m=dict(r['metrics'])
   for a in r['rollouts']:
    for key in ('seconds','inference_seconds','environment_seconds'):
     m[f"rollout/gpu{a['physical_gpu']}_{key}"]=a[key]
   rows.append(dict(update=r['update'],metrics=m))
evaluations=[]
for f in sorted(p.glob('evaluation-*.json')):
 e=read(f.name,{})
 if 'metrics' in e:evaluations.append(dict(tag=e['tag'],update=e['update'],metrics=e['metrics']))
print(json.dumps(dict(rows=rows,evaluations=evaluations,status=read('status.json',{}),supervisor=read('supervisor-status.json',{}),config=read('effective-config.json',{}),report=(p/'RESULTS.md').read_text() if (p/'RESULTS.md').exists() else '',results=read('RESULTS.json',{}))))
'''
    command = 'python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(REMOTE) + ' ' + str(after)
    result = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def copy_artifacts():
    for name in ('learning-curves.png', 'evaluation-baseline.json', 'evaluation-final.json'):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                        f'{HOST}:{REMOTE}/{name}', str(REPORT / name)],
                       capture_output=True, timeout=30)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = open(REPORT / 'relay.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import wandb
    state_path = REPORT / 'relay-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(last_update=256, evaluations=[])
    first = snapshot(state['last_update'])
    run = wandb.init(entity=wandb.Api(timeout=15).default_entity,
                     project='rpent-pi05-skills', id='overnight20261104u257',
                     resume='allow', name='pi05-v3-overnight-20261104-to-0900',
                     group='pi05-v3-overnight-20261104', job_type='remote-metrics-monitor',
                     mode='online', dir=str(REPORT),
                     config=dict(first['config'], monitoring='Read-only SSH metrics relay; GPU2 is shared'),
                     settings=wandb.Settings(init_timeout=60, x_disable_stats=True))
    run.define_metric('update')
    for prefix in ('train/*', 'eval/*', 'rollout/*', 'coverage/*'):
        run.define_metric(prefix, step_metric='update')
    cloud = dict(url=run.url, run_id=run.id, entity=run.entity, project=run.project,
                 mode='online metrics relay; server retains full offline W&B run',
                 local_pid=os.getpid(), remote_run=REMOTE)
    save(REPORT / 'wandb-cloud.json', cloud)
    write_remote = "import sys,pathlib;pathlib.Path(" + repr(REMOTE + '/wandb-cloud.json') + ").write_text(sys.stdin.read())"
    subprocess.run(SSH + ['python3 -c ' + shlex.quote(write_remote)], input=json.dumps(cloud),
                   text=True, capture_output=True, timeout=20, check=True)
    try:
        while time.time() < HARD_DEADLINE + 600:
            try:
                data = snapshot(state['last_update'])
                for row in data['rows']:
                    run.log(dict(row['metrics'], update=row['update']))
                    state['last_update'] = max(state['last_update'], row['update'])
                for evaluation in data['evaluations']:
                    key = f"{evaluation['tag']}:{evaluation['update']}"
                    if key not in state['evaluations']:
                        run.log(dict(evaluation['metrics'], update=evaluation['update'],
                                     evaluation_tag=evaluation['tag']))
                        state['evaluations'].append(key)
                save(REPORT / 'remote-results.json', data['results'])
                (REPORT / 'RESULTS.md').write_text(data['report'])
                save(REPORT / 'evaluations.json', data['evaluations'])
                state.update(last_poll=time.time(), supervisor=data['supervisor'], cloud_url=run.url)
                save(state_path, state)
                copy_artifacts()
                terminal = data['supervisor'].get('state')
                if terminal in ('completed', 'failed', 'stopped', 'hard_deadline'):
                    run.summary.update(dict(terminal_state=terminal, final_update=state['last_update'],
                                            remote_status=data['status']))
                    break
            except Exception:
                traceback.print_exc()
            time.sleep(45)
    finally:
        run.finish()


if __name__ == '__main__':
    for attempt in range(3):
        try:
            main()
            break
        except Exception:
            traceback.print_exc()
            if attempt == 2:
                raise
            time.sleep(30)
