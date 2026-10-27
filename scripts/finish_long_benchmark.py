"""After the active batch exits, finish the matched-group Kimi rule and audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path);args=parser.parse_args();batch=args.batch.resolve()
    pid=int((batch/'batch.pid').read_text())
    while True:
        try:
            state=Path(f'/proc/{pid}/stat').read_text().split()[2]
            if state=='Z':break
        except FileNotFoundError:break
        time.sleep(10)
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_long_benchmark import refresh
    summary=refresh(batch)
    base=[r for r in summary['results'] if r['planner'] in ['none','gpt6']]
    if len(base)!=80:
        (batch/'completion-supervisor.json').write_text(json.dumps(dict(status='requires_infrastructure_review',completed_base=len(base)),indent=2)+'\n')
        return
    status=dict(status='finishing_matched_group_rule',completed=summary['completed'],expected=120,
        reason='Original GAWM pure 10/10; GPT has at least three failures, so matched-group difference exceeds 20 percentage points.')
    (batch/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    with (batch/'continuation.log').open('a') as log:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_long_benchmark.py'),'--output',str(batch)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=False)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/organize_long_benchmark.py'),str(batch)],cwd=ROOT,check=True)
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_long_benchmark import refresh
    refresh(batch)
    report=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/summarize_long_benchmark.py'),str(batch)],cwd=ROOT,
        capture_output=True,text=True,check=False)
    (batch/'final-report-build.log').write_text(report.stdout+report.stderr)
    sources=batch/'sources';sources.mkdir(exist_ok=True)
    checks={}
    for name in ['run_long_benchmark.py','long_benchmark_entrypoint.py','long_benchmark_kimi.py','summarize_long_benchmark.py','finish_long_benchmark.py','organize_long_benchmark.py','run_long_kimi_tail_case.py','gawm_rpent_bridge.py','openpi_rpent_bridge.py','openpi_transport.py']:
        source=ROOT/'scripts'/name;destination=sources/name;shutil.copyfile(source,destination)
        checks['scripts/'+name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
    (batch/'completion-supervisor.json').write_text(json.dumps(dict(status='finished',batch_exit_code=result.returncode,
        report_exit_code=report.returncode),indent=2)+'\n')


if __name__=='__main__':main()
