"""Resume remaining Kimi trials after provider quota returns, without model monitoring."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_long_benchmark as runner


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path)
    p.add_argument('--start-now',action='store_true',help='Check quota and resume immediately before the regular retry interval')
    a=p.parse_args();batch=a.batch.resolve();root=Path(__file__).resolve().parents[1]
    unknown_failures=0
    while True:
        summary=runner.refresh(batch)
        if summary['completed']==120:break
        runner.save(batch/'status.json',dict(status='waiting_for_kimi_quota',completed=summary['completed'],expected=120,at=runner.now(),next_probe_after_s=1800))
        # A minimal provider request every 30 minutes; no Codex invocation is used.
        if a.start_now:a.start_now=False
        else:time.sleep(1800)
        probe=subprocess.run(['/home/jh/.kimi-code/bin/kimi','-m','kimi-code/k3.1-pro',
            '-p','Reply with exactly OK.','--output-format','stream-json'],cwd=batch,
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=120)
        text=probe.stdout+'\n'+probe.stderr
        quota=probe.returncode!=0 and ('429' in text or 'usage limit' in text.lower())
        runner.save(batch/'kimi-quota-probe.json',dict(at=runner.now(),exit_code=probe.returncode,
            quota_exhausted=quota,passed=probe.returncode==0 and 'OK' in probe.stdout))
        if quota:continue
        if probe.returncode or 'OK' not in probe.stdout:
            unknown_failures+=1
            if unknown_failures>=3:raise RuntimeError('Kimi recovery preflight failed three times; see kimi-quota-probe.json')
            continue
        runner.save(batch/'status.json',dict(status='running_remaining_kimi',completed=summary['completed'],expected=120,at=runner.now()))
        with (batch/'quota-recovery-worker.log').open('a') as log:
            result=subprocess.run([str(root/'.venv/bin/python'),'-u',str(root/'scripts/run_long_benchmark.py'),
                '--output',str(batch),'--only-planner','kimi'],cwd=root,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=False)
        subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/organize_long_benchmark.py'),str(batch)],cwd=root,check=True)
        updated=runner.refresh(batch)
        unknown_failures=unknown_failures+1 if updated['completed']==summary['completed'] else 0
        if result.returncode or unknown_failures>=3:
            raise RuntimeError('Kimi worker failed or repeatedly made no progress; see quota-recovery-worker.log')
    subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/organize_long_benchmark.py'),str(batch)],cwd=root,check=True)
    runner.refresh(batch)
    subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/summarize_long_benchmark.py'),str(batch)],cwd=root,check=True)
    audit=json.loads((batch/'final-verification.json').read_text())
    if not audit['complete']:raise RuntimeError('Final audit is incomplete')
    sources=batch/'sources';hashes={}
    for name in ['run_long_benchmark.py','long_benchmark_entrypoint.py','long_benchmark_kimi.py',
            'summarize_long_benchmark.py','finish_long_benchmark.py','organize_long_benchmark.py',
            'run_long_kimi_tail_case.py','resume_long_kimi_after_quota.py','notify_long_benchmark_completion.py',
            'gawm_rpent_bridge.py','openpi_rpent_bridge.py','openpi_transport.py']:
        source=root/'scripts'/name;shutil.copyfile(source,sources/name)
        hashes['scripts/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
    runner.save(sources/'sha256.json',hashes)
    runner.save(batch/'status.json',dict(status='complete',completed=120,expected=120,at=runner.now()))
    runner.save(batch/'completion-supervisor.json',dict(status='finished',completed=120,expected=120,
        complete=True,report_exit_code=0,at=runner.now(),quota_recovery=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        if len(sys.argv)>1:
            runner.save(Path(sys.argv[1]).resolve()/'completion-supervisor.json',dict(status='requires_infrastructure_review',error=repr(exc),at=runner.now()))
        raise
