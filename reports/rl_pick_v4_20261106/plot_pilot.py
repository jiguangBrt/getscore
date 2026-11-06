"""Plot native-stage audits and PPO diagnostics; no extra training dependencies."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(values,window=50):
    a=np.asarray(values,dtype=float)
    return np.array([np.mean(a[max(0,i-window+1):i+1]) for i in range(len(a))])


def plot(paths,out):
    fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for i,path in enumerate(paths):
        a=json.loads(path.read_text());name='Trial '+Path(a['run']).name.rsplit('-',1)[-1].upper();recs=a['training']['records'];updates=a['updates']
        color=f'C{i}'
        x=np.arange(1,len(recs)+1)
        axes[0,0].plot(x,smooth([r['reward'] for r in recs]),label=name,color=color)
        axes[0,1].plot(x,smooth([r['success'] for r in recs]),label=name,color=color)
        for key,label,ls in [('ever_verified','verified grasp','-'),('ever_lift_5cm','lift >=5cm','--'),('max_hold','hold steps',':')]:
            vals=[r[key]/40 if key=='max_hold' else r[key] for r in recs]
            axes[1,0].plot(x,smooth(vals),label=f'{name}: {label}',ls=ls,color=color)
        xx=[r['update'] for r in updates];lm=[r['learner'] for r in updates]
        axes[1,1].plot(xx,[max(r['approx_kl'],1e-10) for r in lm],label=name,color=color)
        axes[2,0].plot(xx,[r['value_mse_after'] for r in lm],label=f'{name}: GAE-target MSE',color=color)
        axes[2,1].plot(xx,[r['value_explained_variance_after'] for r in lm],label=name,color=color)
    for ax,title,xlab,ylab in zip(axes.flat,
       ['Complete-episode return (rolling50)','Training success (rolling50)',
        'Physical stages (rolling50; hold /40)','Post-update sampled-chain KL (all batch rows)',
        'Critic fit against current GAE targets','Critic explained variance against GAE targets'],
       ['Complete episodes']*3+['Policy update cycles']*3,
       ['Return','Success fraction','Fraction / hold progress','Approx. KL','MSE','Explained variance']):
        ax.set(title=title,xlabel=xlab,ylabel=ylab);ax.grid(alpha=.25);ax.legend(fontsize=7)
    axes[0,1].set_ylim(-.02,1.02);axes[1,0].set_ylim(-.02,1.02);axes[1,1].set_yscale('log')
    axes[2,1].axhline(0,color='gray',lw=.8)
    fig.savefig(out,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audits',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.audits,a.out)
