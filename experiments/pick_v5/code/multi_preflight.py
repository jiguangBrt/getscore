"""CPU physical controls; scripted outcomes are never policy success metrics."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
from pick_env import PickEnv, inventory
from runtime import save, CFG
from liberopro.liberopro.envs.env_wrapper import ControlEnv
import liberopro.liberopro.envs as env_module


def no_render(**kwargs):
    kwargs.update(use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False)
    return ControlEnv(**kwargs)


env_module.OffScreenRenderEnv=no_render


class Probe(PickEnv):
    def observation(self):return {}


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    results=[]
    for spec in inventory():
        for control in ('grasp','table_collision','open_lift'):
            e=None
            name=f"{spec['type']}-{control}"
            try:
                e=Probe(dict(spec,skill='pick'),1210000,out/name,0.,True)
                rows=[]
                def step(a):
                    if not e.done:rows.append(e.step(a)[-1])
                def move(target,g):
                    for _ in range(100):
                        if e.done:break
                        delta=np.asarray(target)-e.obs['robot0_eef_pos']
                        if np.linalg.norm(delta)<.003:break
                        step(np.r_[np.clip(delta/.05,-.5,.5),0,0,0,g])
                pos=e.sim.data.body_xpos[e.body].copy()
                bounds=[e.geom_bounds(g) for g in e.collision_ids]
                if control=='table_collision':
                    # Empty patch, so this must not be confused with target contact.
                    move([pos[0]+.14,pos[1],e.support_z+.015],-1)
                    for _ in range(10):step(np.r_[0,0,-.2,0,0,0,-1])
                else:
                    move(pos+[0,0,.052],-1)
                    g=1 if control=='grasp' else -1
                    for _ in range(25):step(np.r_[np.zeros(6),g])
                    move(np.asarray(e.obs['robot0_eef_pos'])+[0,0,.10],g)
                for _ in range(200):step(np.r_[np.zeros(6),1 if control=='grasp' else -1])
                final=rows[-1]
                r=dict(type=spec['type'],control=control,valid_reset=True,success=final['skill_success'],
                       reward=final['episode_reward'],max_hold=max(x['hold_steps'] for x in rows),
                       max_height=max(x['clearance_m'] for x in rows),
                       verified_grasp_steps=sum(x['grasped'] for x in rows),
                       collision_cost=final['cumulative_collision_cost'],collision_seconds=final['cumulative_collision_seconds'],
                       peak_force_n=max(x['gripper_collision_peak_force_n'] for x in rows),
                       intended_peak_force_n=max(x['gripper_intended_contact_peak_force_n'] for x in rows),
                       substeps=sorted(set(x['collision_physics_substeps'] for x in rows)),
                       initial=e.initial_metrics,bounds=[[a.tolist(),b.tolist()] for a,b in bounds],
                       gripper_geoms=sorted(e.collision_monitor.gripper_ids),
                       collision_pairs=[p for x in rows for p in x['gripper_collision_pairs']][:40])
            except Exception:r=dict(type=spec['type'],control=control,valid_reset=False,error=traceback.format_exc())
            finally:
                if e is not None:e.close()
            results.append(r)
            summary=dict(protocol='scripted CPU physics controls, not policy evaluation',results=results,
                         all_resets_valid=len(results)==len(inventory())*3 and all(x['valid_reset'] for x in results),
                         collision_controls_passed=len(results)==len(inventory())*3 and
                            all(x.get('collision_cost',0)>0 for x in results if x['control']=='table_collision') and
                            all(not x.get('success') and x.get('verified_grasp_steps')==0 for x in results if x['control']=='open_lift'))
            save(out/'summary.json',summary)
            print(json.dumps({k:v for k,v in r.items() if k not in ('initial','bounds','collision_pairs','gripper_geoms')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
