"""Reproduce scientific training figures from the compact exported run summary."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).with_name('learning-data.json'));p.add_argument('--final',action='store_true');a=p.parse_args()
data=json.loads(a.data.read_text());root=a.data.parent
warm=data['pick-v5-warmup-20261106-e']
runs=[data[n] for n in ['pick-v5-pilot800-20261106-h','pick-v5-pilot701-20261106-k'] if n in data]
metrics=[r for run in runs for r in run.get('metrics',[])];metrics.sort(key=lambda x:x['update'])
episodes=[r for run in runs for r in run.get('episodes',[])]
evals={r['update']:r['metrics'] for run in runs for r in run['evaluations'] if not r.get('reused_from')}
balanced=data.get('pick-v5-balanced200-20261106-m',{})
bmetrics=balanced.get('metrics',[]);bevals={r['update']:r['metrics'] for r in balanced.get('evaluations',[]) if not r.get('reused_from')}
keys=sorted(evals);types=warm['effective-config']['object_types'];short={'porcelain_mug':'white mug','alphabet_soup':'soup can','ketchup':'ketchup','tomato_sauce':'tomato can','butter':'butter','cream_cheese':'cheese','akita_black_bowl':'black bowl','white_yellow_mug':'white/yellow mug','black_book':'book','chocolate_pudding':'chocolate','moka_pot':'moka pot','yellow_moka_pot':'yellow moka'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
ax=axs[0,0];wm=warm.get('metrics',[]);ax.plot([r['step'] for r in wm],[r['flow_matching_loss'] for r in wm],color='#506882',alpha=.7,label='flow-matching loss');ax.set_yscale('log');ax.set_xlabel('supervised steps');ax.set_ylabel('loss');ax.set_title('A. Supervised initialization: loss and independent development success')
right=ax.twinx();we=sorted(warm['evaluations'],key=lambda r:r['update']);right.plot([r['update'] for r in we],[r['metrics']['success_rate'] for r in we],'o-',color='#b9472c',label='dev48 success');right.axvline(600,color='#b9472c',ls=':',alpha=.5);right.set_ylim(0,1);right.set_ylabel('success fraction');right.text(620,.9,'selected: 600 (coverage)',fontsize=8,color='#b9472c')
ax=axs[0,1];rewards=np.array([r['episode_reward'] for r in episodes]);success=np.array([r['success'] for r in episodes],float)
def roll(x):return np.array([np.mean(x[max(0,i-49):i+1]) for i in range(len(x))])
x=np.arange(1,len(rewards)+1);ax.plot(x,roll(success),color='#167765',label='success, rolling50');ax.set_ylim(0,1);ax.set_xlabel('complete PPO episodes (H + K)');ax.set_ylabel('success fraction');ax.set_title('B. Actual stochastic-policy training episodes (trailing up to 50)');right=ax.twinx();right.plot(x,roll(rewards),color='#b26e0e',alpha=.7,label='raw return, rolling50');right.set_ylabel('raw return');right.axhline(0,color='grey',lw=.6)
if balanced.get('episodes'):
 bs=[float(r['success']) for r in balanced['episodes']];ax.plot(np.arange(1,len(bs)+1),roll(bs),'--',color='#8b54a1',label='per-object normalized');ax.legend(fontsize=7,loc='lower right')
ax=axs[1,0];columns=[(f'C{k}',evals[k]) for k in keys]+[(f'B{k}',bevals[k]) for k in sorted(bevals)]
matrix=np.array([[m['by_object'][t]['successes']/m['by_object'][t]['valid'] for _,m in columns] for t in types]);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_yticks(range(len(types)),[short[t] for t in types]);ax.set_xticks(range(len(columns)),[k for k,m in columns],rotation=45);ax.set_xlabel('update: C=global control, B=per-object normalization');ax.set_title('C. Fixed evaluation: four cases per object per checkpoint');fig.colorbar(im,ax=ax,shrink=.8,label='success fraction')
for i in range(len(types)):
 for j in range(len(columns)):ax.text(j,i,str(round(matrix[i,j]*4)),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>.65 else '#183040')
ax=axs[1,1];ax.plot(keys,[evals[k]['mean_return'] for k in keys],'o-',color='#167765',label='mean raw return');ax.set_ylabel('raw return');ax.set_xlabel('PPO update');ax.set_title('D. Same fixed cases: return and unwanted gripper contact cost');ax.set_ylim(-.5,3.4);right=ax.twinx();right.plot(keys,[np.mean([r['collision_cost'] for r in evals[k]['by_object'].values()]) for k in keys],'s--',color='#b9472c');right.set_ylabel('mean collision cost');right.set_ylim(bottom=0)
if bevals:
 bk=sorted(bevals);ax.plot(bk,[bevals[k]['mean_return'] for k in bk],'o--',color='#8b54a1',label='per-object normalized');ax.legend(fontsize=7)
ax=axs[2,0];up=[r['update'] for r in metrics]
for key,label,c in [('value_explained_variance_before','before update','#9aabb6'),('value_explained_variance_after','after update','#167765')]:ax.plot(up,[np.nan if r['learner'][key] is None else r['learner'][key] for r in metrics],label=label,color=c)
ax.set_ylim(-1,1);ax.axhline(0,color='grey',lw=.6);ax.set_xlabel('PPO update');ax.set_ylabel('explained variance');ax.set_title('E. Critic fit to GAE targets (not a task-success metric)');ax.legend(loc='lower right',fontsize=8)
if bmetrics:ax.plot([r['update'] for r in bmetrics],[r['learner']['value_explained_variance_after'] for r in bmetrics],'--',color='#8b54a1',label='per-object normalized');ax.legend(loc='lower right',fontsize=7)
ax=axs[2,1];ax.plot(up,[max(1e-9,r['learner']['approx_kl']) for r in metrics],color='#506882');ax.axhline(.01,color='#b26e0e',ls='--',label='target .01');ax.axhline(.05,color='#b9472c',ls='--',label='reject .05');ax.set_yscale('log');ax.set_xlabel('PPO update');ax.set_ylabel('approximate chain KL');ax.set_title('F. Full-batch update guard');ax.legend(fontsize=8)
if bmetrics:ax.plot([r['update'] for r in bmetrics],[max(1e-9,r['learner']['approx_kl']) for r in bmetrics],'--',color='#8b54a1',label='per-object normalized');ax.legend(fontsize=7)
fig.suptitle('Single-object randomized grasp pilots — '+('completed' if a.final else 'in progress')+'\nSupervised initialization and PPO are reported separately; evaluation seeds differ between A and C/D.',fontsize=13)
fig.savefig(root/'learning-curves.png',dpi=170);plt.close(fig)
