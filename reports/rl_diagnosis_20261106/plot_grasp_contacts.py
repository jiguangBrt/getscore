from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = json.loads((root/'grasp-contact-traces/wall_dx0_z52.json').read_text())
control = json.loads((root/'grasp-contact-traces/wall_dx12_z52.json').read_text())
t = np.arange(1,len(rows)+1)/20
fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
axes[0].plot(t,[100*r['metrics']['clearance_m'] for r in rows],color='#167b98',label='Cup bottom above tabletop')
axes[0].axhline(5,color='#999',ls='--',label='5 cm lift threshold')
axes[0].set(ylabel='Clearance (cm)',title='CPU physics probe: a held cup missed by the pad-only grasp detector')
axes[0].legend(loc='upper left')
flags=[('Left finger (any collision geom)','finger_left'),('Right finger (any collision geom)','finger_right'),('Left designated pad','pad_left'),('Right designated pad','pad_right'),('Original grasped flag',None)]
matrix=np.array([[int(r[k] if k else r['metrics']['grasped']) for r in rows] for _,k in flags])
axes[1].imshow(matrix,aspect='auto',interpolation='nearest',extent=[t[0],t[-1],len(flags)-.5,-.5],cmap='Blues',vmin=0,vmax=1)
axes[1].set_yticks(range(len(flags)),[x[0] for x in flags]);axes[1].set_title('Dark = contact / true; white = no contact / false',fontsize=11)
axes[2].plot(t,[r['metrics']['linear_speed']*100 for r in rows],color='#bd6527',label='Cup linear speed')
axes[2].axhline(1,color='#999',ls='--',label='1 cm/s hold gate')
axes[2].set(xlabel='Time since scripted descent began (s)',ylabel='Speed (cm/s)')
axes[2].legend()
for ax in [axes[0],axes[2]]:
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'grasp-contact-probe.png',dpi=160)
plt.close(fig)
