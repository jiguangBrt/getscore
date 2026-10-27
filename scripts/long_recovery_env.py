"""Restore archived native actions only when replay observations match exactly."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent'),str(ROOT/'data/rlinf-runtime')]
from robots.libero import env_server
from rpent.utils.serialization import to_numpy_tree


class RecoveryFacade(env_server.LiberoEnvFacade):
    def __init__(self,*args,**kwargs):
        self._recovery=None
        self._reset_result=None
        super().__init__(*args,**kwargs)
        self._rpc['env.recovery_info']=lambda:dict(self._recovery or {})
        self._readonly_methods.add('env.recovery_info')

    def reset(self):
        # RPent connects once, then initializes primitives; both resets precede motion.
        if self._reset_result is not None:
            return self._reset_result
        obs,info=super().reset()
        source=Path(os.environ['LONG_RECOVERY_SOURCE'])
        report=dict(source_run=str(source),mode='restart',resumed_steps=0,
                    validation='exact camera images, <=1um per-step EEF error, native termination agreement')
        if os.environ.get('LONG_RECOVERY_DISABLED')=='1':
            report['reason']='native_session_restore_failed_or_explicit_restart'
        else:
            obs,report=self._try_replay(source,obs,report)
        self._recovery=report
        destination=Path(os.environ['LONG_RECOVERY_OUTPUT'])/'recovery.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2)+'\n')
        self._reset_result=(obs,{**info,'recovery':report})
        return self._reset_result

    def _try_replay(self,source,initial_obs,report):
        import numpy as np
        from rlinf.envs.sim.libero.utils import quat2axisangle
        config=json.loads((source/'configuration.json').read_text())
        for kind in ('bddl_files','init_files'):
            entry=config[kind]
            if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Recovery source asset changed: '+entry['path'])
        episode=next(source.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*'))
        with np.load(episode/'transitions.npz') as data:
            actions=data['actions'];states=data['states'];terms=data['terminated']
            images={k:data[k][[0,-1]] for k in ('main_images','wrist_images')}
        report['source_actions']=len(actions)
        report['initial_images_exact']=all(np.array_equal(initial_obs[k],images[k][0]) for k in images)
        if not report['initial_images_exact']:
            report['reason']='initial_observation_differs_from_archive'
            return initial_obs,report
        max_error=0.0
        failed=None
        self._env.env.set_camera_rendering(False)
        try:
            for index,action in enumerate(actions,1):
                if index==len(actions):self._env.env.set_camera_rendering(True)
                _,_,term,trunc,_=self._env.step(action[None],_skip_obs_wrap=True)
                raw=self._env.current_raw_obs[0]
                actual=np.concatenate([raw['robot0_eef_pos'],quat2axisangle(raw['robot0_eef_quat']),raw['robot0_gripper_qpos']])
                error=float(np.max(np.abs(actual[:3]-states[index,:3])))
                max_error=max(max_error,error)
                if error>1e-6 or bool(term[0])!=bool(terms[index-1]) or bool(trunc[0]):
                    failed=index;break
        finally:self._env.env.set_camera_rendering(True)
        report.update(max_eef_component_error_m=max_error,first_replay_mismatch_step=failed)
        if failed is None:
            obs=self._strip_obs(to_numpy_tree(self._env._wrap_obs(self._env.current_raw_obs)))
            report['final_images_exact']=all(np.array_equal(obs[k],images[k][-1]) for k in images)
            saved=json.loads((source/'states.json').read_text())['steps'][-1]['state']
            raw=self._env.current_raw_obs[0]
            report['final_raw_errors']={k:float(np.max(np.abs(raw[k]-saved[k]))) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
            if report['final_images_exact'] and max(report['final_raw_errors'].values())<=1e-8:
                if config['planner']=='kimi':
                    report.update(mode='resume',resumed_steps=len(actions),reason='replay_observations_match_archive')
                    return obs,report
                report['reason']='native_GPT_context_restore_not_available_in_this_entrypoint'
            else:report['reason']='final_observation_differs_from_archive'
        else:report['reason']='native_replay_diverged'
        obs,_=super().reset()
        return obs,report


if __name__=='__main__':
    env_server.LiberoEnvFacade=RecoveryFacade
    env_server.main()
