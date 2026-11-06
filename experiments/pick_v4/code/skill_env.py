from runtime import BASE,CFG,save
import numpy as np
import os
from scipy.spatial.transform import Rotation

class SkillEnv:
    def __init__(self,skill,out):
        import torch
        from liberopro.liberopro import benchmark,get_libero_path
        from liberopro.liberopro.envs import OffScreenRenderEnv
        self.skill=skill; self.out=out
        suite=benchmark.get_benchmark('libero_10')(task_order_index=0)
        task=suite.get_task(6)
        self.task=task
        bddl=BASE/'src/LIBERO-PRO/liberopro/liberopro/bddl_files'/task.problem_folder/task.bddl_file
        self.wrapper=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=CFG['camera_size'],camera_widths=CFG['camera_size'],control_freq=CFG['control_hz'],ignore_done=True,render_gpu_device_id=int(os.environ["MUJOCO_EGL_DEVICE_ID"]))
        self.env=self.wrapper.env;self.sim=self.wrapper.sim
        self.wrapper.seed(CFG['seed'])
        self.obs=self.wrapper.reset()
        states=suite.get_task_init_states(6)
        self.obs=self.wrapper.set_init_state(states[0])
        # hard_reset replaces the simulator; bind only after the final reset.
        self.env=self.wrapper.env;self.sim=self.wrapper.sim
        self.object_name='porcelain_mug_1'
        self.obj=self.env.objects_dict[self.object_name]
        self.body=self.env.obj_body_id[self.object_name]
        self.plate=self.env.obj_body_id['plate_1']
        self.robot=self.env.robots[0]
        gripper=self.robot.gripper
        self.gripper=gripper['right'] if isinstance(gripper,dict) else gripper
        self.initialization_actions=[]
        for _ in range(15):self._init_step(np.array([0,0,0,0,0,0,-1.]))
        self.table_z=float(self.env.workspace_offset[2])
        self.goal_xy=self.sim.data.body_xpos[self.plate,:2].copy()
        self.start_pos=self.sim.data.body_xpos[self.body].copy()
        self.base_bottom_z=self.start_pos[2]+float(self.obj.bottom_offset[2])
        self.training_step=0;self.stable_count=0;self.pick_count=0;self.closed=False;self.done=False
        if skill=='upright':
            qpos=self.sim.data.get_joint_qpos(self.obj.joints[0]).copy()
            qpos[:3]=self.start_pos+np.array([0,0,0.065])
            # A native physics reset, not an executed policy action.
            q=Rotation.from_euler('y',90,degrees=True).as_quat()
            qpos[3:]=q[[3,0,1,2]]
            self.sim.data.set_joint_qpos(self.obj.joints[0],qpos)
            self.sim.data.set_joint_qvel(self.obj.joints[0],np.zeros(6))
            self.sim.forward()
            for _ in range(40):self._init_step(np.array([0,0,0,0,0,0,-1.]))
        if skill=='place':self._prepare_held_state()
        self.start_pos=self.sim.data.body_xpos[self.body].copy()
        self.initial_held=self.is_grasped()
        self.initial_metrics=self.metrics()
        if skill=='place':
            assert self.initial_held and self.initial_metrics['lift_m']>0.05,('Invalid held reset',self.initial_metrics)
        if skill=='upright':assert self.initial_metrics['tilt_degrees']>60,self.initial_metrics
        self.prev_potential=self.potential(self.initial_metrics)
        self._refresh()
        self.out.mkdir(parents=True,exist_ok=True)
        np.save(self.out/'initial_sim_state.npy',self.wrapper.get_sim_state())
        (self.out/'scene.xml').write_text(self.sim.model.get_xml())
        save(self.out/'reset.json',{'skill':skill,'task':task.name,'prompt':CFG['policy_prompt'],'initialization_actions':len(self.initialization_actions),'initial_metrics':self.initial_metrics,'initialization':'scripted physical grasp for place only; excluded from policy trajectories' if skill=='place' else 'native reset / orientation reset; excluded from policy trajectories'})
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
    def _prepare_held_state(self):
        p=self.sim.data.body_xpos[self.body].copy()
        self._move(p+[0.01,0,0.20])
        self._move(p+[0.012,0,0.052],steps=100)
        for _ in range(30):self._init_step(np.array([0,0,0,0,0,0,1.]))
        self._move(p+[0.012,0,0.20],g=1,steps=100)
        for _ in range(10):self._init_step(np.array([0,0,0,0,0,0,1.]))
    def is_grasped(self):
        return bool(self.env._check_grasp(self.gripper,self.obj.contact_geoms))
    def metrics(self):
        pos=self.sim.data.body_xpos[self.body].copy();eef=np.asarray(self.obs['robot0_eef_pos'])
        R=self.sim.data.body_xmat[self.body].reshape(3,3)
        tilt=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
        qvel=self.sim.data.get_joint_qvel(self.obj.joints[0])
        contacts=[];object_geoms=set(self.obj.contact_geoms)
        support=False;target_support=False
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            n1=self.sim.model.geom_id2name(c.geom1) or '';n2=self.sim.model.geom_id2name(c.geom2) or ''
            if n1 in object_geoms or n2 in object_geoms:
                other_id=c.geom2 if n1 in object_geoms else c.geom1
                other=(n2 if n1 in object_geoms else n1) or self.sim.model.body_id2name(self.sim.model.geom_bodyid[other_id]) or ''
                if 'table' in other or 'plate' in other:support=True
                if 'plate_1' in other:target_support=True
        return dict(object_pos=pos.tolist(),eef_pos=eef.tolist(),tilt_degrees=tilt,lift_m=float(pos[2]+float(self.obj.bottom_offset[2])-self.base_bottom_z),grasped=self.is_grasped(),supported=support,target_supported=target_support,target_xy_distance=float(np.linalg.norm(pos[:2]-self.goal_xy)),eef_distance=float(np.linalg.norm(pos+np.array([0,0,0.05])-eef)),linear_speed=float(np.linalg.norm(qvel[:3])),angular_speed=float(np.linalg.norm(qvel[3:])),gripper_opening=float(np.abs(self.obs['robot0_gripper_qpos']).sum()))
    def potential(self,m):
        if self.skill=='pick':return -m['eef_distance']+min(max(m['lift_m'],0),0.1)+(0.1 if m['grasped'] else 0.)
        return -m['target_xy_distance']-0.05*m['tilt_degrees']/90.-max(0.,m['lift_m'])
    def observation(self):
        quat=self.obs['robot0_eef_quat'];aa=Rotation.from_quat(quat).as_rotvec()
        state=np.r_[self.obs['robot0_eef_pos'],aa,self.obs['robot0_gripper_qpos']].astype(np.float32)
        return {'main_images':self.obs['agentview_image'][::-1,::-1].copy()[None], 'wrist_images':self.obs['robot0_eye_in_hand_image'][::-1,::-1].copy()[None], 'states':state[None], 'task_descriptions':[CFG['policy_prompt']]}
    def frame(self):return self.observation()['main_images'][0]
    def step(self,action):
        assert not self.done
        self.obs,_,_,_=self.wrapper.step(np.asarray(action,dtype=np.float32))
        self.training_step+=1;m=self.metrics();threshold=CFG['success']
        picked=m['grasped'] and m['lift_m']>=threshold['lift_m']
        self.pick_count=self.pick_count+1 if picked else 0
        stable=(not m['grasped'] and m['target_supported'] and m['tilt_degrees']<=threshold['upright_degrees'] and m['target_xy_distance']<=threshold['target_radius_m'] and m['linear_speed']<=threshold['linear_speed_m_s'] and m['angular_speed']<=threshold['angular_speed_rad_s'] and m['eef_distance']>=threshold['eef_clearance_m'] and m['gripper_opening']>0.06)
        self.stable_count=self.stable_count+1 if stable else 0
        success=self.pick_count>=threshold['pick_hold_steps'] if self.skill=='pick' else self.stable_count>=threshold['stable_steps']
        failure=m['object_pos'][2]<self.table_z-0.10
        terminated=bool(success or failure);truncated=self.training_step>=CFG['max_native_steps_per_skill'] and not terminated
        phi=self.potential(m);r=CFG['reward']['potential_scale']*(CFG['gamma_per_native_step']*phi-self.prev_potential)-CFG['reward']['step_penalty']
        r+=CFG['reward']['terminal_bonus']*float(success)
        if failure:r-=1.
        self.prev_potential=phi;self.done=terminated or truncated
        return self.observation(),float(r),terminated,bool(truncated),dict(m,skill_success=bool(success),failure=bool(failure),stable_count=self.stable_count,pick_count=self.pick_count)
    def close(self):self.wrapper.close()

if __name__=='__main__':
    import imageio.v2 as imageio
    for skill in CFG['skills']:
        e=SkillEnv(skill,BASE/'outputs/reset_check'/skill)
        imageio.imwrite(e.out/'initial.png',e.frame())
        print(skill,e.initial_metrics,flush=True);e.close()
