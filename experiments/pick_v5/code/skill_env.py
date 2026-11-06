"""Shared camera/state and native reset helpers; task/reward live in PickEnv."""
from runtime import CFG
import numpy as np
from scipy.spatial.transform import Rotation


class SkillEnv:
    def _refresh(self):
        self.sim.forward();self.env._update_observables(force=True)
        self.obs=self.env._get_observations(force_update=True)
    def _init_step(self,a):
        self.obs,_,_,_=self.wrapper.step(np.asarray(a,dtype=np.float32))
        self.initialization_actions.append(np.asarray(a).tolist())
    def _move(self,xyz,g=-1,steps=100):
        for _ in range(steps):
            e=np.asarray(self.obs['robot0_eef_pos']);delta=np.asarray(xyz)-e
            if np.linalg.norm(delta)<0.004:break
            self._init_step(np.r_[np.clip(delta/0.05,-0.5,0.5),[0.,0.,0.,g]])
    def observation(self):
        quat=self.obs['robot0_eef_quat'];aa=Rotation.from_quat(quat).as_rotvec()
        state=np.r_[self.obs['robot0_eef_pos'],aa,self.obs['robot0_gripper_qpos']].astype(np.float32)
        return {'main_images':self.obs['agentview_image'][::-1,::-1].copy()[None], 'wrist_images':self.obs['robot0_eye_in_hand_image'][::-1,::-1].copy()[None], 'states':state[None], 'task_descriptions':[CFG['policy_prompt']]}
    def frame(self):return self.observation()['main_images'][0]
    def close(self):self.wrapper.close()
