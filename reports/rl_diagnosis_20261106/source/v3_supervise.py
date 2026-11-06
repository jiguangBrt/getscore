"""Detached bounded supervisor; no credentials, external messages, or other users' processes."""
from config_contract import guard_runtime
guard_runtime('rpent-skills-v3-20261104')
import argparse,os,time,json,signal,subprocess,sys,traceback
from pathlib import Path
from v3_summary import summary
BASE=Path('/data/wangyihan/getscore-rl-skills')
PROFILE=Path(__file__).resolve().parents[1]

def save(path,data):
    q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');q.replace(path)
def kill_group(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=10)
def supervise(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'supervisor.lock','a+')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'supervisor-pid.json',dict(pid=os.getpid(),started=time.time(),out=str(out)))
    stopped=False;p=None
    def stop(sig,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    attempts_path=out/'attempt-count.json'
    used=json.loads(attempts_path.read_text()).get('started',0) if attempts_path.exists() else 0
    last_progress=0.;last_plot=0.
    for attempt in range(used,3):
        if time.time()>=json.loads((out/'run-meta.json').read_text())['deadline']:break
        save(attempts_path,dict(started=attempt+1,max_attempts=3))
        if stopped:break
        cmd=[str(PROFILE/'run.sh'),'v3_train.py','--out',str(out),'--hours',str(args.hours),'--envs',str(args.envs),'--chunks',str(args.chunks)]
        if args.preflight:cmd+=['--preflight','--max-updates','4']
        logfile=out/f'trainer-attempt{attempt}.log'
        with open(logfile,'a',buffering=1) as log:
            p=subprocess.Popen(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(out/'supervisor-status.json',dict(state='running',attempt=attempt,pid=p.pid,command=cmd,log=str(logfile)))
            while p.poll() is None:
                if stopped:kill_group(p);break
                meta=json.loads((out/'run-meta.json').read_text()) if (out/'run-meta.json').exists() else {}
                if meta and time.time()>=meta['reporting_deadline']:
                    save(out/'supervisor-status.json',dict(state='hard_deadline',attempt=attempt,pid=p.pid));kill_group(p);break
                event=out/'events.jsonl'
                if event.exists() and time.time()-event.stat().st_mtime>2100:
                    save(out/'supervisor-status.json',dict(state='stalled',attempt=attempt,pid=p.pid));kill_group(p);break
                try:
                    gpu=subprocess.run(['nvidia-smi','-i','0,1,2,3','--query-gpu=timestamp,index,uuid,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                    with open(out/'gpu.csv','a') as f:f.write(gpu.stdout)
                    summary(out)
                    now=time.time()
                    if now-last_plot>=300:
                        subprocess.run([str(PROFILE/'run.sh'),'plot_run.py',str(out)],cwd=BASE,timeout=90,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                        last_plot=now
                    milestones=list((out/'checkpoints').glob('milestone-*.pt'))
                    newest=max((x.stat().st_mtime for x in milestones),default=0.)
                    if newest>last_progress:
                        from v3_finalize import finalize
                        finalize(out,stage='3h checkpoint');last_progress=newest
                except Exception:
                    with open(out/'supervisor-errors.log','a') as f:f.write(traceback.format_exc())
                for _ in range(30):
                    if p.poll() is not None or stopped:break
                    time.sleep(1)
        rc=p.wait();kill_group(p)
        state={}
        try:state=json.loads((out/'status.json').read_text())
        except (OSError,ValueError):pass
        if rc==0 and state.get('state')=='completed':
            save(out/'supervisor-status.json',dict(state='completed',attempt=attempt,exit_code=rc,finished=time.time()));summary(out)
            if not args.preflight:
                from v3_finalize import finalize
                finalize(out)
            return 0
        save(out/'supervisor-status.json',dict(state='interrupted' if stopped else 'retry_pending',attempt=attempt,exit_code=rc,finished=time.time()))
        summary(out)
        text=logfile.read_text(errors='replace')[-30000:]
        if any(word in text for word in ('FloatingPointError','logprob mismatch','trust-region guard','foreign checkpoint','Invalid task configuration','AssertionError','fixed batch8 OOM','supports rpent-skills','base checkpoint hash mismatch')):
            save(out/'supervisor-status.json',dict(state='failed',reason='nonrecoverable_numerics_or_config',attempt=attempt,exit_code=rc,finished=time.time()));summary(out)
            if not args.preflight:
                from v3_finalize import finalize
                finalize(out)
            return 1
        if stopped:break
        meta=json.loads((out/'run-meta.json').read_text()) if (out/'run-meta.json').exists() else {}
        if meta and time.time()>=meta['deadline']:break
        time.sleep(10)
    save(out/'supervisor-status.json',dict(state='stopped' if stopped else ('hard_deadline' if time.time()>=json.loads((out/'run-meta.json').read_text())['reporting_deadline'] else 'failed'),finished=time.time(),last_exit_code=p.returncode if p else None));summary(out)
    if not args.preflight:
        from v3_finalize import finalize
        finalize(out)
    return 1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--hours',type=float,default=6);ap.add_argument('--envs',type=int,default=24);ap.add_argument('--chunks',type=int,default=32);ap.add_argument('--preflight',action='store_true');sys.exit(supervise(ap.parse_args()))
