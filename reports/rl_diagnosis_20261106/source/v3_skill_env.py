"""RPent-like skill initializations over original LIBERO-PRO physics."""
from train_runtime import BASE,CFG,save,stack_obs
from skill_env import SkillEnv
from v3_reward import RewardLedger, VERSION
assert CFG["task_version"]==VERSION
from pathlib import Path
import numpy as np,os,re,json,traceback,time
from scipy.spatial.transform import Rotation

FLAT_TYPES={'black_book','plate','chefmate_8_frypan','cream_cheese','butter','chocolate_pudding'}
UPRIGHT_EXCLUDED=FLAT_TYPES|{'basket'}
LABELS={'porcelain_mug':'white mug','white_yellow_mug':'yellow and white mug','red_coffee_mug':'red mug','moka_pot':'moka pot','akita_black_bowl':'black bowl','black_book':'book','alphabet_soup':'alphabet soup can','cream_cheese':'cream cheese box','tomato_sauce':'tomato sauce can','butter':'butter box','chocolate_pudding':'chocolate pudding','ketchup':'ketchup bottle','orange_juice':'orange juice carton','milk':'milk carton','wine_bottle':'wine bottle','chefmate_8_frypan':'frying pan','basket':'basket','plate':'plate'}

def inventory():
    from liberopro.liberopro import benchmark
    suite=benchmark.get_benchmark('libero_10')(task_order_index=0);result=[]
    # Prefer accessible table scenes and original object names; retain distractors too.
    for ti in [6,5,0,1,2,3,4,7,8,9]:
        t=suite.get_task(ti);p=BASE/'src/LIBERO-PRO/liberopro/liberopro/bddl_files'/t.problem_folder/t.bddl_file
        s=p.read_text();section=s.split('(:objects',1)[1].split(')',1)[0]
        for names,typ in re.findall(r'([a-z0-9_\s]+)\s+-\s+([a-z0-9_]+)',section):
            for name in names.split():
                result.append(dict(type=typ,object=name,label=LABELS.get(typ,typ.replace('_',' ')),upright_eligible=typ not in UPRIGHT_EXCLUDED,placement_orientation='flat' if typ in FLAT_TYPES else 'upright',task_index=ti,task=t.name))
    return result

class V3SkillEnv(SkillEnv):
    def __init__(self,spec,seed,out,progress=0.,evaluation=False):
        from liberopro.liberopro import benchmark
        from liberopro.liberopro.envs import OffScreenRenderEnv
        self.spec=spec;self.skill=spec['skill'];self.seed=seed;self.rng=np.random.default_rng(seed);self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=True)
        suite=benchmark.get_benchmark('libero_10')(task_order_index=0);ti=spec['task_index'];task=suite.get_task(ti);self.task=task
        bddl=BASE/'src/LIBERO-PRO/liberopro/liberopro/bddl_files'/task.problem_folder/task.bddl_file
        self.wrapper=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=CFG['camera_size'],camera_widths=CFG['camera_size'],control_freq=CFG['control_hz'],ignore_done=True,render_gpu_device_id=int(os.environ['MUJOCO_EGL_DEVICE_ID']),hard_reset=False)
        self.wrapper.seed(seed);self.obs=self.wrapper.reset()
        states=suite.get_task_init_states(ti);self.init_index=seed%len(states);self.obs=self.wrapper.set_init_state(states[self.init_index])
        self.env=self.wrapper.env;self.sim=self.wrapper.sim
        self.object_name=spec['object'];self.obj=self.env.objects_dict[self.object_name];self.body=self.env.obj_body_id[self.object_name]
        self.robot=self.env.robots[0];g=self.robot.gripper;self.gripper=g['right'] if isinstance(g,dict) else g
        self.initialization_actions=[];self.table_z=float(self.env.workspace_offset[2])
        for _ in range(15):self._init_step(np.r_[np.zeros(6),-1.])
        # Original material/friction/mass/controller settings are unchanged.
        joint=self.obj.joints[0];q=self.sim.data.get_joint_qpos(joint).copy()
        q[:2]+=self.rng.uniform(-.012,.012,2)
        yaw=self.rng.uniform(-20,20)
        q[3:]=(Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_quat(q[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        self.sim.data.set_joint_qpos(joint,q);self.sim.data.set_joint_qvel(joint,np.zeros(6));self.sim.forward()
        for _ in range(20):self._init_step(np.r_[np.zeros(6),-1.])
        canonical=self.sim.data.body_xmat[self.body].reshape(3,3).copy()
        # Object axes are canonicalized from original authored resting pose, including books/plates.
        self.local_up=canonical.T@np.array([0.,0.,1.])
        self.start_pos=self.sim.data.body_xpos[self.body].copy();self.base_bottom_z=self.start_pos[2]+float(self.obj.bottom_offset[2])
        self.goal_xy=self.start_pos[:2].copy();self.plate=None;self.target_kind='table_region'
        self.target_radius=CFG[self.skill].get('target_radius_m',.05)
        self.support_z=self.table_z
        self.collision_ids=[self.sim.model.geom_name2id(n) for n in self.obj.contact_geoms]
        # Original scenes have different authored table tops (not workspace_offset).
        candidates=[]
        for gid in range(self.sim.model.ngeom):
            name=(self.sim.model.geom_id2name(gid) or '')+' '+(self.sim.model.body_id2name(self.sim.model.geom_bodyid[gid]) or '')
            if 'table' not in name:continue
            lower,upper=self.geom_bounds(gid)
            if upper[2]-lower[2]>.15:continue # table legs / vertical sides
            if np.all(self.start_pos[:2]>=lower[:2]) and np.all(self.start_pos[:2]<=upper[:2]):candidates.append((float(upper[2]),gid))
        if not candidates:raise RuntimeError('no valid local table collision support beneath object')
        self.support_z=max(z for z,gid in candidates)
        self.support_ids={gid for z,gid in candidates if abs(z-self.support_z)<.002}
        self.table_z=self.support_z

        if self.skill=='upright':
            # Natural settling can undo mild tilts. Retry until the state actually needs recovery.
            startq=self.sim.data.get_joint_qpos(joint).copy()
            severity=90. if evaluation else min(90.,60.+30.*progress)
            for attempt in range(3):
                q=startq.copy();axis=self.rng.uniform(0,2*np.pi);angle=severity if attempt==0 else 90.
                rot=Rotation.from_rotvec(np.array([np.cos(axis),np.sin(axis),0])*np.deg2rad(angle))*Rotation.from_quat(startq[[4,5,6,3]])
                q[:3]=self.start_pos+[0,0,.065];q[3:]=rot.as_quat()[[3,0,1,2]]
                self.sim.data.set_joint_qpos(joint,q);self.sim.data.set_joint_qvel(joint,np.zeros(6));self.sim.forward()
                for _ in range(50):self._init_step(np.r_[np.zeros(6),-1.])
                self._refresh()
                if self.metrics()['tilt_degrees']>10:break
            if self.metrics()['tilt_degrees']<=10:raise RuntimeError('upright reset self-righted; no training on trivial success')
            self.goal_xy=self.sim.data.body_xpos[self.body,:2].copy()
        self._refresh()
        if self.skill=='place':
            self._prepare_held_state()
            if not self.is_grasped():raise RuntimeError('physical held-state setup failed')
            # Move the REAL held object to 10cm COLLISION clearance, never teleport it.
            for _ in range(100):
                self._refresh();gap=self.collision_bottom()-self.support_z
                error=CFG['place']['initial_object_clearance_m']-gap
                if abs(error)<.002:break
                self._init_step(np.r_[0.,0.,np.clip(error/.05,-.25,.25),0.,0.,0.,1.])
            for _ in range(5):self._init_step(np.r_[np.zeros(6),1.])
            self._refresh();gap=self.collision_bottom()-self.support_z
            if not self.is_grasped() or abs(gap-CFG['place']['initial_object_clearance_m'])>CFG['place']['initial_object_clearance_tolerance_m']:
                raise RuntimeError(f'invalid held clearance {gap}, grasp={self.is_grasped()}')
            self.goal_xy=self.sim.data.body_xpos[self.body,:2].copy()
        else:
            pos=self.sim.data.body_xpos[self.body].copy()
            self._move(pos+np.r_[self.rng.uniform(-.015,.015,2),.17],steps=100)
        self._refresh();self.training_step=0;self.stable_count=0;self.pick_count=0;self.closed=False;self.done=False
        self.initial_held=self.is_grasped();self.initial_metrics=self.metrics()
        if self.skill=='upright' and (self.initial_metrics['tilt_degrees']<=10 or self.initial_metrics['linear_speed']>CFG['success']['linear_speed_m_s'] or self.initial_metrics['angular_speed']>CFG['success']['angular_speed_rad_s']):
            raise RuntimeError('upright initial state no longer tilted and naturally stationary')
        self.policy_version='script'

        self.ledger=RewardLedger(CFG,self.skill,self.initial_metrics)
        self.trace_file=self.out/'native-trace.jsonl';self.trace_file.write_text('')
        label=spec['label'];placement=spec.get('placement_orientation','upright')
        template='place_flat' if self.skill=='place' and placement=='flat' else ('place_upright' if self.skill=='place' else self.skill)
        self.prompt=CFG['prompt_templates'][template].format(object=label)
        save(self.out/'reset.json',dict(spec=spec,prompt=self.prompt,seed=seed,init_index=self.init_index,metrics=self.initial_metrics,initialization_actions=len(self.initialization_actions),target_kind=self.target_kind,target_xy=self.goal_xy.tolist(),target_radius=self.target_radius,task_version=VERSION,policy_native_steps=0,collision_bottom_z=self.collision_bottom(),support_z=self.support_z))
    def _prepare_held_state(self):
        p=self.sim.data.body_xpos[self.body].copy()
        # Native scripted physical grasps; reset failure is logged, never a fabricated held state.
        candidates=[(0,0,.025),(0,0,.07),(.012,0,.052)] if self.spec['type'] in ('moka_pot','yellow_moka_pot') else [(.012,0,.052),(0,0,.035),(-.012,.01,.065)]
        for dx,dy,z in candidates:
            self._move(p+[0,0,.22],g=-1,steps=100);self._move(p+[dx,dy,z],g=-1,steps=100)
            for _ in range(25):self._init_step(np.r_[np.zeros(6),1.])
            self._move(np.array(self.obs['robot0_eef_pos'])+[0,0,.16],g=1,steps=100)
            for _ in range(8):self._init_step(np.r_[np.zeros(6),1.])
            if self.is_grasped():return
            self._move(p+[dx,dy,.10],g=1,steps=50)
            for _ in range(15):self._init_step(np.r_[np.zeros(6),-1.])
            p=self.sim.data.body_xpos[self.body].copy()
        raise RuntimeError('all real-grasp initializations failed')
    def metrics(self):
        pos=self.sim.data.body_xpos[self.body].copy();eef=np.asarray(self.obs['robot0_eef_pos']);R=self.sim.data.body_xmat[self.body].reshape(3,3)
        tilt=float(np.degrees(np.arccos(np.clip((R@self.local_up)[2],-1,1))))
        qv=self.sim.data.get_joint_qvel(self.obj.joints[0]);support=target_support=False;geoms=set(self.obj.contact_geoms)
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            n1=self.sim.model.geom_id2name(c.geom1) or '';n2=self.sim.model.geom_id2name(c.geom2) or ''
            if n1 in geoms or n2 in geoms:
                other_id=c.geom2 if n1 in geoms else c.geom1
                other=self.sim.model.geom_id2name(other_id) or self.sim.model.body_id2name(self.sim.model.geom_bodyid[other_id]) or ''
                if 'table' in other and abs(float(c.frame[2]))>.5:
                    support=True
                    if other_id in self.support_ids:target_support=True
                if self.plate is not None and int(self.sim.model.geom_bodyid[other_id])==self.plate:target_support=True;support=True
        dist=float(np.linalg.norm(pos[:2]-self.goal_xy))
        if self.target_kind=='plate':
            target_support=bool(target_support and self.sim.data.body_xpos[self.plate,2]<=pos[2] and dist<.03)
        else:target_support=bool(target_support and dist<=self.target_radius)
        return dict(clearance_m=self.collision_bottom()-self.support_z,table_z=self.table_z,object_pos=pos.tolist(),eef_pos=eef.tolist(),tilt_degrees=tilt,lift_m=float(pos[2]+float(self.obj.bottom_offset[2])-self.base_bottom_z),grasped=self.is_grasped(),supported=support,target_supported=target_support,target_xy_distance=dist,eef_distance=float(np.linalg.norm(pos+[0,0,.05]-eef)),linear_speed=float(np.linalg.norm(qv[:3])),angular_speed=float(np.linalg.norm(qv[3:])),gripper_opening=float(np.abs(self.obs['robot0_gripper_qpos']).sum()),eef_object_center_distance=float(np.linalg.norm(pos-eef)))
    def observation(self):
        obs=super().observation();obs['task_descriptions']=[self.prompt];return obs
    def geom_bounds(self,gid):
        model=self.sim.model;data=self.sim.data;typ=int(model.geom_type[gid]);size=model.geom_size[gid];R=data.geom_xmat[gid].reshape(3,3);pos=data.geom_xpos[gid]
        if typ==7:
            mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);vertices=model.mesh_vert[start:start+n]@R.T+pos
            return vertices.min(0),vertices.max(0)
        if typ==2:extent=np.repeat(size[0],3)
        elif typ in (3,5):extent=np.abs(R[:,2])*size[1]+size[0]*(np.ones(3) if typ==3 else np.sqrt(np.maximum(0.,1-R[:,2]**2)))
        elif typ==4:extent=np.linalg.norm(R*size,axis=1)
        elif typ==6:extent=np.abs(R)@size
        else:raise RuntimeError(f'unsupported collision shape {typ}')
        return pos-extent,pos+extent
    def collision_bottom(self):
        return min(float(self.geom_bounds(gid)[0][2]) for gid in self.collision_ids)
    def step(self,action):
        assert not self.done
        self.obs,_,_,_=self.wrapper.step(np.asarray(action,dtype=np.float32));self.training_step+=1
        m=self.metrics();r,term,trunc,info=self.ledger.advance(m);self.done=term or trunc
        info.update(skill=self.skill,object=self.spec['label'],prompt=self.prompt,policy_start=True,policy_version=self.policy_version,task_version=VERSION)
        with open(self.trace_file,'a') as f:f.write(json.dumps(info)+'\n')
        return self.observation(),r,term,trunc,info

def worker(pipe,worker_id,out):
    env=None;current_spec=None
    try:
        while True:
            cmd,payload=pipe.recv()
            if cmd=='close':break
            if cmd=='reset':
                if env is not None:env.close();env=None
                spec,seed,progress,evaluation,episode_id=payload
                try:
                    env=V3SkillEnv.__new__(V3SkillEnv)
                    env.__init__(spec,seed,Path(out)/f'env{worker_id:02d}'/f'episode{episode_id:06d}',progress,evaluation)
                    pipe.send(dict(ok=True,obs=env.observation(),metrics=env.initial_metrics,prompt=env.prompt))
                except Exception as exc:
                    if env is not None and hasattr(env,'wrapper'):env.wrapper.close()
                    env=None
                    pipe.send(dict(ok=False,error=str(exc),traceback=traceback.format_exc(),spec=spec))
            elif cmd=='inspect':
                pipe.send(dict(native_step=env.training_step,ledger_step=env.ledger.step,earned=env.ledger.earned,hold=env.ledger.hold,stable=env.ledger.stable,sim_sha256=__import__('hashlib').sha256(env.wrapper.get_sim_state().tobytes()).hexdigest(),seed=env.seed,spec=env.spec,sim_time=float(env.sim.data.time)))
            elif cmd=='step':
                actions,record,version=payload;env.policy_version=version;rewards=[];trace=[];frames=[];term=trunc=False
                for a in actions:
                    obs,r,term,trunc,info=env.step(a);rewards.append(r);info['policy_version']=version;trace.append(info)
                    if record:frames.append(overlay(np.concatenate([obs['main_images'][0],obs['wrist_images'][0]],axis=1),info))
                    if term or trunc:break
                pipe.send(dict(obs=obs,rewards=rewards,trace=trace,frames=frames,terminated=term,truncated=trunc,info=info,native_steps=len(rewards)))
    except EOFError:pass
    except Exception:
        try:pipe.send(dict(fatal=True,traceback=traceback.format_exc()))
        except Exception:pass
    finally:
        if env is not None:env.close()
        pipe.close()

def overlay(frame,info):
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(frame.shape[1],frame.shape[0]+220),'#101820');im.paste(Image.fromarray(frame),(0,220));d=ImageDraw.Draw(im)
    lines=[f"{info['skill']} | {info['object']} | POLICY v{info.get('policy_version','script')} | {info['phase']}",info['prompt'],f"native {info['native_step']}/200 remaining {info['remaining_steps']} hold {info['hold_steps']}/40 stable {info['stable_count']}/80",f"height {info['object_lift_m']:.4f} peak {info['max_object_lift_m']:.4f} XY {info['xy_offset_m']:.4f} path {info['xy_path_m']:.4f}",f"reward {info['reward']:.5f} total {info['episode_reward']:.3f} end {info['termination_reason']}",str({k:round(v,5) for k,v in info['reward_components'].items()})]
    import textwrap
    lines=[part for line in lines for part in (textwrap.wrap(line,width=82) or [''])]
    for i,line in enumerate(lines):d.text((7,4+i*20),line,fill='white')
    return np.asarray(im)
