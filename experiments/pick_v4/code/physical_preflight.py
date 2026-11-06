"""CPU physical positive/negative controls, separate from policy evaluation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import json
from pathlib import Path
import numpy as np
from pick_env import PickEnv, inventory
from pick_reward import RewardLedger
from runtime import CFG
from liberopro.liberopro.envs.env_wrapper import ControlEnv
import liberopro.liberopro.envs as env_module


def no_render(**kwargs):
    kwargs.update(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    return ControlEnv(**kwargs)


env_module.OffScreenRenderEnv = no_render


class Probe(PickEnv):
    def observation(self):
        return {}


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed, dx, z, control in [(90000000, 0., .052, 'positive'), (90000000, .012, .052, 'positive'),
                                  (90000001, 0., .052, 'positive'), (90000002, 0., .052, 'positive'),
                                  (90000000, 0., .052, 'table_only'), (90000000, 0., .052, 'open_lift'),
                                  (90000000, 0., .052, 'recover'), (90000000, 0., .052, 'severe'),
                                  (90000000, 0., .052, 'repeat')]:
        name = f'{control}-{seed}-dx{dx}'
        env = None
        try:
            env = Probe(dict(inventory()[0], skill='pick'), seed, out/name, 0., True)
            rows = []
            def step(action, phase):
                if env.done:
                    return
                _, _, _, _, info = env.step(action)
                rows.append(dict(info, scripted_phase=phase, action=np.asarray(action).tolist()))
            def move(target, grip, phase, limit=80):
                for _ in range(limit):
                    if env.done:
                        break
                    delta = np.asarray(target)-np.asarray(env.obs['robot0_eef_pos'])
                    if np.linalg.norm(delta)<.003:
                        break
                    step(np.r_[np.clip(delta/.05,-.5,.5),0.,0.,0.,grip],phase)
            pos = env.sim.data.body_xpos[env.body].copy()
            move(pos+[dx,0,z],-1.,'descend')
            for _ in range(25):step(np.r_[np.zeros(6),1. if control!='open_lift' else -1.],'close')
            if control!='table_only':
                move(np.asarray(env.obs['robot0_eef_pos'])+[0.,0.,.10],1. if control!='open_lift' else -1.,'lift')
            if control in ('recover','severe'):
                original_eef=np.asarray(env.obs['robot0_eef_pos']).copy()
                move(original_eef+[.07 if control=='recover' else .23,0.,0.],1.,'excursion')
                if control=='recover':move(original_eef,1.,'return')
            for _ in range(200):step(np.r_[np.zeros(6),1. if control!='open_lift' else -1.],'hold')
            final=rows[-1]
            result=dict(name=name,control=control,seed=seed,steps=len(rows),success=final['skill_success'],
                        reason=final['termination_reason'],episode_reward=final['episode_reward'],
                        max_clearance_m=max(r['clearance_m'] for r in rows),
                        max_hold=max(r['hold_steps'] for r in rows),
                        max_xy_offset_m=max(r['xy_offset_m'] for r in rows),
                        survived_soft_boundary=any(r['xy_offset_m']>.05 and not r['terminated'] for r in rows),
                        recovered_after_soft=any(rows[i]['xy_offset_m']<=.03 and any(r['xy_offset_m']>.05 for r in rows[:i]) for i in range(len(rows))),
                        airborne_pad_misses=sum(r['grasped'] and not r['pad_only_grasp'] for r in rows),
                        verified_grasp_steps=sum(r['grasped'] for r in rows),
                        initial_sim_sha256=json.loads((out/name/'reset.json').read_text())['initial_sim_sha256'],
                        final=final)
            (out/name/'scripted-trace.json').write_text(json.dumps(rows,indent=2))
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='final'}),flush=True)
        except Exception:
            import traceback
            results.append(dict(name=name,error=traceback.format_exc()))
            print(results[-1]['error'],flush=True)
        finally:
            if env is not None:env.close()
    first=next((r for r in results if r['name']=='positive-90000000-dx0.0'),{})
    repeat=next((r for r in results if r['control']=='repeat'),{})
    summary=dict(protocol='CPU physical scripted controls; no policy or learning',results=results,
                 repeat_initial_state_match=first.get('initial_sim_sha256')==repeat.get('initial_sim_sha256'),
                 repeat_reward_match=first.get('episode_reward')==repeat.get('episode_reward'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
