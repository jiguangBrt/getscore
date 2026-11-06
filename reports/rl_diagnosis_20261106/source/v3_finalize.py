"""Export complete bounded-run results and a report-only stage commit."""
from pathlib import Path
import json,subprocess,argparse,time
BASE=Path('/data/wangyihan/getscore-rl-skills')
PROFILE=Path(__file__).resolve().parents[1]
def finalize(out,commit=True,stage="final"):
    from v3_summary import summary
    out=Path(out);result=summary(out)
    subprocess.run([str(PROFILE/'run.sh'),'plot_run.py',str(out)],cwd=BASE,check=True,timeout=180)
    report=BASE/'reports'/f'{out.name}-RESULTS.md';report.parent.mkdir(exist_ok=True)
    out=out.resolve();text=(out/'RESULTS.md').read_text().replace('](checkpoints/',f']({out}/checkpoints/').replace('](videos/',f']({out}/videos/')
    text+='\n\nFull immutable configuration, source hashes, native traces, metrics, evaluation tables, checkpoint inventory, videos and learning-curves.png are in: '+str(out)+'\n'
    report.write_text(text)
    if commit:
        branch=subprocess.run(['git','branch','--show-current'],cwd=BASE,text=True,capture_output=True,check=True).stdout.strip()
        if branch=='codex/pi05-three-skills':
            subprocess.run(['git','add',str(report)],cwd=BASE,check=True)
            p=subprocess.run(['git','commit','--only','-m',f'results: {stage} {out.name}','--',str(report)],cwd=BASE,text=True,capture_output=True)
            (out/'finalize-git.log').write_text(p.stdout+p.stderr)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--no-git',action='store_true');a=p.parse_args();finalize(a.out,not a.no_git)
