"""Recoverable excursion and severe-boundary controls with the current reward."""
from multi_preflight import Probe
from pick_env import inventory
from runtime import save
import argparse
import json
from pathlib import Path
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);results=[]
    for control,distance in [('recover',.07),('severe',.23)]:
        e=Probe(dict(inventory()[0],skill='pick'),1210000,a.out/control,0.,True);rows=[]
        try:
            def step(action):
                if not e.done:rows.append(e.step(action)[-1])
            def move(target,g):
                for _ in range(90):
                    if e.done:break
                    d=np.asarray(target)-e.obs['robot0_eef_pos']
                    if np.linalg.norm(d)<.003:break
                    step(np.r_[np.clip(d/.05,-.5,.5),0,0,0,g])
            pos=e.sim.data.body_xpos[e.body].copy();move(pos+[0,0,.085],-1)
            for _ in range(25):step(np.r_[np.zeros(6),1])
            move(np.asarray(e.obs['robot0_eef_pos'])+[0,0,.10],1)
            origin=np.asarray(e.obs['robot0_eef_pos']).copy();move(origin+[distance,0,0],1)
            survived=not e.done
            if control=='recover':move(origin,1)
            for _ in range(200):step(np.r_[np.zeros(6),1])
            last=rows[-1];r=dict(control=control,success=last['skill_success'],reason=last['termination_reason'],
                max_xy=max(x['xy_offset_m'] for x in rows),survived_excursion=survived,
                failure=last['failure'],max_hold=max(x['hold_steps'] for x in rows),
                positive_soft_cost_steps=sum(x['reward_components']['soft_boundary']<0 for x in rows),
                returned_free=any(x['xy_offset_m']<=.03 and x['reward_components']['soft_boundary']==0
                     and any(y['xy_offset_m']>.05 for y in rows[:i]) for i,x in enumerate(rows)),
                reward=last['episode_reward'])
            results.append(r)
        finally:e.close()
    # This checks recoverability, not an exemption from the unchanged deadline.
    # A slow excursion may leave fewer than 40 stable steps before native200.
    assert results[0]['survived_excursion'] and results[0]['returned_free'] and not results[0]['failure']
    assert results[0]['positive_soft_cost_steps']>0
    assert results[1]['reason']=='severe_xy_boundary' and not results[1]['survived_excursion']
    save(a.out/'summary.json',dict(passed=True,results=results,role='scripted boundary controls, not policy successes'))
