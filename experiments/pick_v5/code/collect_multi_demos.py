"""Balanced verified multi-object demonstrations, explicitly auxiliary to PPO."""
import argparse
import json
from pathlib import Path
import numpy as np
from pick_env import PickEnv,inventory
from runtime import CFG,save


def offsets(typ):
    if typ in ('moka_pot','yellow_moka_pot'):
        return [(.035,0,.052),(-.035,0,.052),(0,0,.052),(0,-.06,.052)]
    if typ=='akita_black_bowl':return [(0,.04,.025),(0,-.04,.025),(.04,0,.025),(-.04,0,.025)]
    zs = (.085,.075,.065,.052) if typ in ('porcelain_mug','white_yellow_mug') else (
         (.04,.025,.01,0) if typ in ('alphabet_soup','tomato_sauce') else (
         (.052,.04,.025) if typ=='ketchup' else ((.065,.052,0) if typ=='black_book' else (0,.01))))
    return [(0,0,z) for z in zs]


def collect(out,count,types=None,seed_base=1240000,lift_min=.10,lift_max=.10):
    out.mkdir(parents=True,exist_ok=False);results=[];accepted={s['type']:0 for s in inventory()}
    for i,spec in enumerate(inventory()):
        if types and spec['type'] not in types:continue
        for trial in range(count*5):
            if accepted[spec['type']]>=count:break
            seed=seed_base+i*1000+trial
            for j,offset in enumerate(offsets(spec['type'])):
                e=None;observations=[];actions=[]
                try:
                    e=PickEnv(dict(spec,skill='pick'),seed,out/f"{spec['type']}-{seed}-try{j}",0.,False)
                    def step(action):
                        if e.done:return
                        observations.append(e.observation());actions.append(np.asarray(action,dtype=np.float32));e.step(action)
                    def move(target,g):
                        for _ in range(90):
                            if e.done:break
                            d=np.asarray(target)-e.obs['robot0_eef_pos']
                            if np.linalg.norm(d)<.003:break
                            step(np.r_[np.clip(d/.05,-.5,.5),0,0,0,g])
                    p=e.sim.data.body_xpos[e.body].copy();move(p+offset,-1)
                    for _ in range(25):step(np.r_[np.zeros(6),1])
                    lift=float(np.random.default_rng(seed).uniform(lift_min,lift_max))
                    move(np.asarray(e.obs['robot0_eef_pos'])+[0,0,lift],1)
                    for _ in range(200):step(np.r_[np.zeros(6),1])
                    final=json.loads(e.trace_file.read_text().splitlines()[-1])
                    r=dict(type=spec['type'],seed=seed,offset=offset,success=final['skill_success'],reward=final['episode_reward'],
                           prompt=e.prompt,scripted_lift_m=lift,collision_cost=final['cumulative_collision_cost'],steps=len(actions),
                           role='physically executed scripted demonstration; never a policy success')
                    if r['success'] and r['reward']>0:
                        path=e.out/'demo.npz'
                        np.savez_compressed(path,main_images=np.stack([o['main_images'][0] for o in observations]),
                            wrist_images=np.stack([o['wrist_images'][0] for o in observations]),
                            states=np.concatenate([o['states'] for o in observations]),actions=np.stack(actions))
                        r['data']=str(path);accepted[spec['type']]+=1
                except Exception:
                    import traceback
                    r=dict(type=spec['type'],seed=seed,offset=offset,success=False,error=traceback.format_exc())
                finally:
                    if e is not None:e.close()
                results.append(r);save(out/'summary.json',dict(config=CFG,accepted_by_object=accepted,accepted=sum(accepted.values()),results=results))
                print(json.dumps(r),flush=True)
                if r['success']:break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--per-object',type=int,default=6)
    p.add_argument('--types',nargs='*');p.add_argument('--seed-base',type=int,default=1240000)
    p.add_argument('--lift-min',type=float,default=.10);p.add_argument('--lift-max',type=float,default=.10)
    a=p.parse_args();collect(a.out,a.per_object,a.types,a.seed_base,a.lift_min,a.lift_max)
