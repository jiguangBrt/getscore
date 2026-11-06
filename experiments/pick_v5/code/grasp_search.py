"""Find physically executable demo recipes; distinct from policy evaluation."""
from multi_preflight import Probe
from pick_env import inventory
from runtime import save
import argparse
import json
from pathlib import Path
import numpy as np


def search(spec, out):
    results=[]
    # Relative offsets refer to the original asset body frame's WORLD position,
    # not the collision bottom; different assets have different body origins.
    if spec['type'] in ('moka_pot','yellow_moka_pot'):
        candidates=[(x,y,z) for z in (.052,.07,.025) for x,y in ((0,-.06),(0,.045),(.035,0),(-.035,0),(0,-.035),(0,0))]
    elif spec['type']=='akita_black_bowl':
        candidates=[(x,y,z) for z in (.025,.01,0.,-.01) for x,y in ((0,0),(.04,0),(-.04,0),(0,.04),(0,-.04))]
    else:
        candidates=[(0,0,z) for z in (0.,.01,.025,.04,.052,-.01)]
    for j,(dx,dy,z) in enumerate(candidates):
        e=None
        try:
            e=Probe(dict(spec,skill='pick'),1220000,out/f"{spec['type']}-{j}",0.,False)
            def step(a):
                if not e.done:e.step(a)
            def move(target,g):
                for _ in range(100):
                    if e.done:break
                    delta=np.asarray(target)-e.obs['robot0_eef_pos']
                    if np.linalg.norm(delta)<.003:break
                    step(np.r_[np.clip(delta/.05,-.5,.5),0,0,0,g])
            p=e.sim.data.body_xpos[e.body].copy()
            move(p+[dx,dy,z],-1)
            for _ in range(25):step(np.r_[np.zeros(6),1])
            move(np.asarray(e.obs['robot0_eef_pos'])+[0,0,.10],1)
            for _ in range(200):step(np.r_[np.zeros(6),1])
            final=json.loads(e.trace_file.read_text().splitlines()[-1])
            rows=[json.loads(x) for x in e.trace_file.read_text().splitlines()]
            r=dict(type=spec['type'],offset=[dx,dy,z],success=final['skill_success'],reward=final['episode_reward'],
                   max_hold=max(x['hold_steps'] for x in rows),max_height=max(x['clearance_m'] for x in rows),
                   max_verified=sum(x['grasped'] for x in rows),collision_cost=final['cumulative_collision_cost'])
        except Exception:
            import traceback
            r=dict(type=spec['type'],offset=[dx,dy,z],success=False,error=traceback.format_exc())
        finally:
            if e is not None:e.close()
        results.append(r);print(json.dumps(r),flush=True)
        save(out/f"{spec['type']}-summary.json",dict(results=results,selected=r if r['success'] else None))
        if r['success']:return r
    return results[-1]


if __name__=='__main__':
    import multiprocessing as mp
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--types',nargs='*');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    with mp.get_context('spawn').Pool(4) as pool:
        results=pool.starmap(search,[(s,a.out) for s in inventory() if not a.types or s['type'] in a.types])
    save(a.out/'summary.json',dict(results=results,role='scripted controls, never policy successes'))
