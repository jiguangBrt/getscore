"""CPU-only contact experiment. No model, training, GPU context, or physics edits."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import sys, json, time, collections
BASE = Path('/data/wangyihan/getscore-rl-skills')
RUN = BASE / 'runs/pi05-skills-v3-overnight-20261104-to-20261105-0900'
os.environ['RL_SKILLS_CONFIG'] = str(RUN / 'effective-config.json')
sys.path.insert(0, str(RUN / 'source'))
from v3_skill_env import V3SkillEnv
import numpy as np
import liberopro.liberopro.envs as env_module
from liberopro.liberopro.envs.env_wrapper import ControlEnv

def no_render_env(**kwargs):
    kwargs.update(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    return ControlEnv(**kwargs)
env_module.OffScreenRenderEnv = no_render_env

class Probe(V3SkillEnv):
    def observation(self):
        return {}  # No images or policy are needed to inspect physics contacts.

original_path = RUN / 'eval-environments-final-actor0/env00/episode011897'
reset = json.loads((original_path / 'reset.json').read_text())
original = [json.loads(x) for x in (original_path / 'native-trace.jsonl').read_text().splitlines()]
out = BASE / 'records' / ('grasp-contact-diagnosis-' + time.strftime('%Y%m%d-%H%M%S'))
out.mkdir(parents=True, exist_ok=False)

def measure(e, phase, index):
    m = e.metrics()
    obj = set(e.obj.contact_geoms)
    groups = e.gripper.important_geoms
    pairs = []
    for i, c in enumerate(e.sim.data.contact[:e.sim.data.ncon]):
        g1 = e.sim.model.geom_id2name(c.geom1) or ''
        g2 = e.sim.model.geom_id2name(c.geom2) or ''
        if g1 not in obj and g2 not in obj:
            continue
        other = g2 if g1 in obj else g1
        if 'gripper' not in other:
            continue
        force = np.zeros(6)
        import mujoco
        mujoco.mj_contactForce(e.sim.model._model, e.sim.data._data, i, force)
        pairs.append(dict(object_geom=g1 if g1 in obj else g2, gripper_geom=other,
                          distance=float(c.dist), position=np.asarray(c.pos).tolist(),
                          normal=np.asarray(c.frame[:3]).tolist(), normal_force=float(force[0])))
    touching = {p['gripper_geom'] for p in pairs}
    return dict(phase=phase, index=index, metrics=m, contacts=pairs,
                pad_left=bool(touching & set(groups['left_fingerpad'])),
                pad_right=bool(touching & set(groups['right_fingerpad'])),
                finger_left=bool(touching & set(groups['left_finger'])),
                finger_right=bool(touching & set(groups['right_finger'])))

results = []
variants = [('recorded_eef_path_fixed_orientation', None),
            ('wall_dx12_z52', (.012, 0., .052)),
            ('wall_dx0_z35', (0., 0., .035)),
            ('wall_dx0_z52', (0., 0., .052)),
            ('wall_dx0_z70', (0., 0., .070))]
for name, offset in variants:
    e = None
    try:
        e = Probe(reset['spec'], reset['seed'], out / name, 1., True)
        rows = []
        initial_match = max(abs(np.asarray(e.initial_metrics['object_pos']) - np.asarray(reset['metrics']['object_pos'])))
        def step(action, phase):
            e.obs, _, _, _ = e.wrapper.step(np.asarray(action, dtype=np.float32))
            rows.append(measure(e, phase, len(rows)+1))
        def move(target, gripper, phase, n=100):
            for _ in range(n):
                delta = np.asarray(target) - np.asarray(e.obs['robot0_eef_pos'])
                if np.linalg.norm(delta) < .003:
                    break
                step(np.r_[np.clip(delta/.05, -.5, .5), 0., 0., 0., gripper], phase)
        if offset is None:
            # Approximate kinematic-path replay, NOT exact action replay: historical
            # logs do not contain end-effector orientation or commanded actions.
            for old in original:
                delta = np.asarray(old['eef_pos']) - np.asarray(e.obs['robot0_eef_pos'])
                grip = -1. if old['native_step'] < 17 else 1.
                step(np.r_[np.clip(delta/.05, -1., 1.), 0., 0., 0., grip], 'recorded_path')
        else:
            p = e.sim.data.body_xpos[e.body].copy()
            move(p + np.asarray(offset), -1., 'descend')
            for _ in range(25):
                step(np.r_[np.zeros(6), 1.], 'close')
            move(np.asarray(e.obs['robot0_eef_pos']) + [0., 0., .12], 1., 'lift')
        for _ in range(40):
            step(np.r_[np.zeros(6), 1.], 'hold')
        (out / name / 'contact-trace.json').write_text(json.dumps(rows, indent=2))
        lifted = [r for r in rows if r['metrics']['clearance_m'] >= .05]
        result = dict(name=name, initial_object_max_abs_error_m=float(initial_match),
                      groups=e.gripper.important_geoms, native_steps=len(rows),
                      max_clearance_m=max(r['metrics']['clearance_m'] for r in rows),
                      final_clearance_m=rows[-1]['metrics']['clearance_m'],
                      raised_5cm_steps=len(lifted),
                      raised_5cm_but_grasp_false=sum(not r['metrics']['grasped'] for r in lifted),
                      raised_5cm_bilateral_anyfinger=sum(r['finger_left'] and r['finger_right'] for r in lifted),
                      raised_5cm_bilateral_pads=sum(r['pad_left'] and r['pad_right'] for r in lifted),
                      airborne_contact_geoms=dict(collections.Counter(p['gripper_geom'] for r in lifted for p in r['contacts'])),
                      airborne_contact_examples=[r for r in lifted if not r['metrics']['grasped']][:2])
        results.append(result)
        print('VARIANT ' + json.dumps({k:v for k,v in result.items() if k!='airborne_contact_examples'}), flush=True)
    except Exception:
        import traceback
        results.append(dict(name=name, error=traceback.format_exc()))
        print(results[-1]['error'], flush=True)
    finally:
        if e is not None:
            e.close()
summary = dict(directory=str(out), protocol='CPU-only MuJoCo; rendering disabled; original physics/config/base scene; no actor or critic loaded',
               original_video_case='pick-00', original_seed=reset['seed'], results=results)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print('RESULT_JSON ' + json.dumps(summary), flush=True)
