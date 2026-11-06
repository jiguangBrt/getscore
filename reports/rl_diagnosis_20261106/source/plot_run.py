"""CPU-only exported learning curves; independent of training and W&B credentials."""
import os,json,time,argparse
from pathlib import Path
BASE=Path('/data/wangyihan/getscore-rl-skills')
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib')
os.environ['CUDA_VISIBLE_DEVICES']=''

def generate(out):
    out=Path(out);path=out/'metrics.jsonl'
    if not path.exists():return
    rows=[]
    for l in path.read_text().splitlines():
        try:rows.append(json.loads(l))
        except ValueError:pass
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);colors={'pick':'#277da8','place':'#43aa8b','upright':'#f8961e'}
    xs=np.asarray([r['elapsed_seconds']/3600 for r in rows])
    def smooth(vals,window=15):
        result=[];v=None
        for x in vals:
            if x is None:result.append(float('nan'));continue
            v=float(x) if v is None else .15*float(x)+.85*v;result.append(v)
        return result
    for skill,color in colors.items():
        vals=[r['metrics'].get(f'train/{skill}_chunk_reward') for r in rows]
        axs[0,0].plot(xs,smooth(vals),label=skill,color=color)
    axs[0,0].set(title='Training reward (EMA)',xlabel='Elapsed hours',ylabel='Reward per executed action chunk');axs[0,0].legend()
    evals=[]
    for p in out.glob('evaluation-*.json'):
        if p.name=='evaluation-cases.json':continue
        try:e=json.loads(p.read_text())
        except ValueError:continue
        if 'metrics' not in e:continue
        idx=next((i for i,r in enumerate(rows) if r['update']==e['update']),None)
        h=0. if e['tag']=='baseline' else (float(xs[idx]) if idx is not None else float(xs[-1]))
        evals.append((h,e))
    for skill,color in colors.items():
        vals=sorted([(h,e['metrics'][f'eval/{skill}_success']) for h,e in evals]);axs[0,1].plot([x[0] for x in vals],[x[1] for x in vals],'-o',label=skill,color=color)
    axs[0,1].set(title='Fixed-case skill evaluations (valid resets)',xlabel='Elapsed hours',ylabel='Success proportion',ylim=(-.03,1.03));axs[0,1].legend()
    axs[1,0].plot(xs,[r['learner']['critic_loss'] for r in rows],color='#7650a4',alpha=.5,label='critic loss');axs[1,0].plot(xs,smooth([r['learner']['critic_loss'] for r in rows]),color='#7650a4',label='EMA');axs[1,0].set(title='Value learning',xlabel='Elapsed hours',ylabel='Critic loss');axs[1,0].legend()
    axs[1,1].plot(xs,[r['metrics']['train/native_steps_per_second'] for r in rows],color='#d65b5b');axs[1,1].set(title='End-to-end training throughput',xlabel='Elapsed hours',ylabel='Native actions / second')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle(f'pi0.5 shared skill PPO | {out.name} | updates={rows[-1]["update"]}, native steps={rows[-1]["native_steps"]}',fontsize=12)
    tmp=out/'learning-curves.png.tmp';fig.savefig(tmp,format='png',dpi=150);tmp.replace(out/'learning-curves.png');plt.close(fig)
    (out/'CURVES.md').write_text('# Learning curves\n\n![Training curves](learning-curves.png)\n\nSee RESULTS.md and per-case evaluation JSON for reset failures and task coverage. These custom skills are not the unmodified LIBERO benchmark.\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out');ap.add_argument('--watch',action='store_true');args=ap.parse_args();out=Path(args.out)
    if not args.watch:generate(out)
    else:
        while True:
            try:
                generate(out)
                p=out/'supervisor-status.json';s=json.loads(p.read_text()) if p.exists() else {}
                meta=json.loads((out/'run-meta.json').read_text())
                if s.get('state') in ['completed','failed','stopped'] or time.time()>meta['deadline']+1800:break
            except Exception as e:print(type(e).__name__,str(e),flush=True)
            time.sleep(300)
