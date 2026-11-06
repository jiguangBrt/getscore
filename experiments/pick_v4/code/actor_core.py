"""Synchronous on-policy joint-skill PPO, two batched actors and parallel MuJoCo workers."""
from config_contract import guard_runtime
guard_runtime('single-white-mug-pick-v4-20261106')
from train_runtime import *
import argparse,json,time,os,sys,traceback,multiprocessing as mp,gc,subprocess,shutil,signal
from collections import defaultdict
from pick_env import inventory,worker as env_worker

PROFILE=Path(__file__).resolve().parents[1]
CODE=PROFILE/'code'
SKILLS=['pick']
def milestone_due(elapsed,last,interval):return int(elapsed//interval)>int(last//interval)
def event(out,kind,**data):
    row=dict(time=time.time(),kind=kind,**data)
    with open(Path(out)/'events.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
def recv(pipe,timeout=600):
    if not pipe.poll(timeout):raise TimeoutError(f'worker did not respond for {timeout}s')
    x=pipe.recv()
    if x.get('fatal'):raise RuntimeError(x['traceback'])
    return x

def choose_spec(reg,rng,elapsed):
    return dict(reg[0],skill='pick')

def finish_gae(rows,bootstrap):
    gae=0.;nv=bootstrap
    for row in reversed(rows):
        gamma=CFG['gamma_per_native_step']**row['native_steps'];cont=0. if row['terminated'] else 1.
        delta=row['reward']+gamma*cont*nv-row['old_value']
        gae=delta+gamma*CFG['gae_lambda_per_chunk']*cont*gae
        row['advantage']=gae;row['return']=gae+row['old_value'];nv=row['old_value']

class Actor:
    def __init__(self,logical,actor_id,out,n):
        self.logical=logical;self.aid=actor_id;self.out=Path(out);self.n=n;self.reg=inventory();self.episode_id=actor_id*10000000
        os.environ['RENDER_LOGICAL_CUDA']=str(logical)
        torch.cuda.set_device(logical);seed_all(CFG['seed']+actor_id)
        self.model=build_model(f'cuda:{logical}')
        ctx=mp.get_context('spawn');self.workers=[]
        for i in range(n):
            a,b=ctx.Pipe();p=ctx.Process(target=env_worker,args=(b,i,str(self.out/f'environments-actor{actor_id}-pid{os.getpid()}')));p.start();b.close();self.workers.append((a,p))
        self.version=-1
        self.active=[None]*n;self.reset_rng=np.random.default_rng(CFG['seed']+actor_id*1000003)
        self.cumulative=dict(started=0,completed=0,success=0,failure=0,timeout=0)
    def inspect_training(self):
        for pipe,p in self.workers:pipe.send(('inspect',None))
        return [recv(pipe) for pipe,p in self.workers]

    def sync(self,path,version):
        state=torch.load(path,map_location='cpu',weights_only=False);apply_state(self.model,state['trainable_state']);self.version=version
    def reset(self,i,spec,seed,progress,evaluation=False):
        self.episode_id+=1;self.workers[i][0].send(('reset',(spec,int(seed),progress,evaluation,self.episode_id)))
    def reset_many(self,specs,seeds,progress,evaluation):
        for i in range(len(specs)):self.reset(i,specs[i],seeds[i],progress,evaluation)
        return [recv(self.workers[i][0],240) for i in range(len(specs))]
    def bootstrap(self,obs):
        with torch.no_grad():_,aux=self.model.predict_action_batch(stack_obs(obs),mode='train')
        return aux['prev_values'].flatten().cpu().tolist()
    def rollout(self,version,elapsed,chunks,video):
        assert version==self.version
        start=time.time();rng=self.reset_rng;progress=min(1.,elapsed/10800.)
        errors=[];segments=[[] for _ in range(self.n)];records=[];episodes=[];frames=[]
        infer_time=env_time=0.;native_total=0;continued=[];coverage=[]
        def initialize(i):
            for retry in range(13):
                spec=choose_spec(self.reg,rng,elapsed);seed=int(rng.integers(80000000))
                self.reset(i,spec,seed,progress);r=recv(self.workers[i][0],240)
                if r.get('ok'):
                    self.active[i]=dict(obs=r['obs'],spec=spec,seed=seed,steps=0,reward=0.,episode_id=self.episode_id)
                    self.cumulative['started']+=1
                    coverage.append(dict(spec=spec,seed=seed,episode_id=self.episode_id))
                    return
                errors.append(dict(spec=spec,seed=seed,error=r.get('error')))
            raise RuntimeError(f'13 reset failures for environment {i}: {errors[-1]}')
        for i in range(self.n):
            if self.active[i] is None:initialize(i)
            else:continued.append(dict(env=i,episode_id=self.active[i]['episode_id'],native_step=self.active[i]['steps']))
        observations=[a['obs'] for a in self.active];specs=[a['spec'] for a in self.active]
        for t in range(chunks):
            ti=time.time()
            with torch.no_grad():actions,aux=self.model.predict_action_batch(stack_obs(observations),mode='train')
            if not torch.isfinite(actions).all() or not torch.isfinite(aux['prev_logprobs']).all():raise FloatingPointError('nonfinite rollout outputs')
            raw=actions.cpu().numpy();infer_time+=time.time()-ti;te=time.time()
            for i,(pipe,p) in enumerate(self.workers):pipe.send(('step',(raw[i],video and i==0,version)))
            results=[recv(pipe,180) for pipe,p in self.workers];env_time+=time.time()-te
            tail_ids=[]
            for i,r in enumerate(results):
                n=r['native_steps'];native_total+=n;self.active[i]['steps']+=n;self.active[i]['reward']+=sum(r['rewards'])
                row=dict(episode_id=self.active[i]['episode_id'],episode_native_step=self.active[i]['steps'],task_index=specs[i]['task_index'],sampling_group=f'{self.aid}-{version}-{t}',sampling_index=i,skill=specs[i]['skill'],object_type=specs[i]['type'],version=version,forward_inputs=slice_tree(aux['forward_inputs'],i),old_logprobs=slice_tree(aux['prev_logprobs'],i),old_value=float(aux['prev_values'][i].item()),reward=sum(CFG['gamma_per_native_step']**j*v for j,v in enumerate(r['rewards'])),undiscounted_reward=sum(r['rewards']),native_steps=n,terminated=r['terminated'],truncated=r['truncated'],valid_action_mask=np.arange(CFG['execute_horizon'])<n)
                segments[i].append(row);observations[i]=r['obs'];self.active[i]['obs']=r['obs']
                if r['frames']:frames.extend(r['frames'])
                if r['terminated'] or r['truncated']:tail_ids.append(i)
            boot_ids=[i for i in tail_ids if results[i]['truncated']]
            values=dict(zip(boot_ids,self.bootstrap([observations[i] for i in boot_ids]))) if boot_ids else {}
            # GAE never crosses resets, and timeout bootstrap uses the original terminal observation.
            for i in tail_ids:
                r=results[i];finish_gae(segments[i],values.get(i,0.));records.extend(segments[i]);segments[i]=[]
                episodes.append(dict(episode_id=self.active[i]['episode_id'],seed=self.active[i]['seed'],task_index=specs[i]['task_index'],native_steps=self.active[i]['steps'],episode_reward=self.active[i]['reward'],skill=specs[i]['skill'],object_type=specs[i]['type'],success=bool(r['info']['skill_success']),failure=bool(r['info']['failure']),timeout=bool(r['info'].get('timeout',False)),final=r['info']))
                self.cumulative['completed']+=1
                for key in ('success','failure','timeout'):self.cumulative[key]+=int(episodes[-1][key])
                self.active[i]=None
            if t+1<chunks:
                for i in tail_ids:
                    initialize(i);observations[i]=self.active[i]['obs'];specs[i]=self.active[i]['spec']
        ids=[i for i,s in enumerate(segments) if s]
        tails=self.bootstrap([observations[i] for i in ids]) if ids else []
        for i,v in zip(ids,tails):finish_gae(segments[i],v);records.extend(segments[i])
        path=self.out/'scratch'/f'rollout-v{version}-actor{self.aid}.pt';atomic_torch(path,records)
        video_path=None
        if frames:
            import imageio.v2 as imageio
            video_path=self.out/'videos'/f'train-v{version}-actor{self.aid}.mp4';video_path.parent.mkdir(exist_ok=True)
            with imageio.get_writer(video_path,fps=20,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
                for f in frames:writer.append_data(f)
        return dict(continued=continued,active=[None if a is None else {k:v for k,v in a.items() if k!='obs'} for a in self.active],coverage=coverage,cumulative=dict(self.cumulative),path=str(path),version=version,chunks=len(records),native_steps=native_total,episodes=episodes,reset_errors=errors,seconds=time.time()-start,inference_seconds=infer_time,environment_seconds=env_time,physical_gpu=CFG['cuda_physical_order'][self.logical],cuda_peak_gib=torch.cuda.max_memory_allocated(self.logical)/2**30,video=str(video_path) if video_path else None)
    def evaluate(self,cases,tag):
        # Keep every live training simulator untouched. Evaluation uses a separate pool.
        evaluation_rng=rng_state();training=self.workers;before=self.inspect_training() if any(a is not None for a in self.active) else None
        ctx=mp.get_context('spawn');evaluation=[]
        try:
            for i in range(self.n):
                a,b=ctx.Pipe();p=ctx.Process(target=env_worker,args=(b,i,str(self.out/f'eval-environments-{tag}-actor{self.aid}')));p.start();b.close();evaluation.append((a,p))
            self.workers=evaluation
            result=self._evaluate(cases,tag)
        finally:
            for pipe,p in evaluation:
                try:pipe.send(('close',None))
                except (BrokenPipeError,EOFError):pass
            for pipe,p in evaluation:
                p.join(15)
                if p.is_alive():p.terminate();p.join(5)
            self.workers=training
            restore_rng(evaluation_rng)
        after=self.inspect_training() if before is not None else None
        assert before==after,'evaluation altered training environment'
        result['training_state_preserved']=True;result['training_before']=before;result['training_after']=after
        return result
    def _evaluate(self,cases,tag):
        import imageio.v2 as imageio
        results=[];videos=[]
        for offset in range(0,len(cases),self.n):
            batch=cases[offset:offset+self.n];specs=[x['spec'] for x in batch];seeds=[x['seed'] for x in batch]
            rr=self.reset_many(specs,seeds,1.,True);active={};frames={};step_counts={};writers={}
            for i,r in enumerate(rr):
                if not r.get('ok'):
                    results.append(dict(case=batch[i],valid_reset=False,error=r.get('error')));continue
                active[i]=r['obs'];step_counts[i]=0
                # One recorded episode of each skill per actor per checkpoint.
                skill=specs[i]['skill']
                if CFG.get('record_all_eval',False) or not any(v['skill']==skill for v in videos):
                    suffix=f'-{batch[i]["id"]}' if CFG.get('record_all_eval',False) else ''
                    p=self.out/'videos'/f'eval-{tag}-actor{self.aid}-{skill}{suffix}.mp4';p.parent.mkdir(exist_ok=True)
                    writers[i]=imageio.get_writer(p,fps=20,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-threads','2']);videos.append(dict(skill=skill,path=str(p),case=batch[i]))
            while active:
                ids=list(active);obs=stack_obs([active[i] for i in ids])
                # Fixed case-specific noise makes evaluations independent of actor batching/order.
                generators=[torch.Generator(device=self.logical).manual_seed(batch[i]['seed']+step_counts[i]*7919) for i in ids]
                noise=torch.cat([torch.randn(1,CFG['action_horizon'],32,device=self.logical,generator=g) for g in generators],0)
                with torch.no_grad():actions,_=self.model.predict_action_batch(obs,mode='eval',noise=noise)
                raw=actions.cpu().numpy()
                for j,i in enumerate(ids):self.workers[i][0].send(('step',(raw[j],i in writers,self.version)))
                for i in ids:
                    r=recv(self.workers[i][0],180);step_counts[i]+=r['native_steps']
                    if i in writers:
                        for f in r['frames']:writers[i].append_data(f)
                    if r['terminated'] or r['truncated']:
                        results.append(dict(case=batch[i],valid_reset=True,success=bool(r['info']['skill_success']),native_steps=step_counts[i],final=r['info']))
                        if i in writers:writers.pop(i).close()
                        del active[i]
                    else:active[i]=r['obs']
        return dict(results=results,videos=videos,version=self.version,tag=tag)
    def close(self):
        for pipe,p in self.workers:
            try:pipe.send(('close',None))
            except Exception:pass
        for pipe,p in self.workers:
            p.join(15)
            if p.is_alive():p.terminate();p.join(5)

def actor_worker(pipe,logical,aid,out,n):
    actor=None
    try:
        actor=Actor(logical,aid,out,n);pipe.send(dict(ready=True))
        while True:
            cmd,data=pipe.recv()
            if cmd=='close':break
            if cmd=='sync':actor.sync(data['path'],data['version']);pipe.send(dict(synced=actor.version))
            elif cmd=='rollout':pipe.send(actor.rollout(**data))
            elif cmd=='eval':pipe.send(actor.evaluate(**data))
            elif cmd=='state':
                pipe.send(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(logical),reset_rng=actor.reset_rng.bit_generator.state,episode_id=actor.episode_id,cumulative=actor.cumulative))
            elif cmd=='restore':
                random.setstate(data['python']);np.random.set_state(data['numpy']);torch.set_rng_state(data['torch']);torch.cuda.set_rng_state(data['cuda'],logical)
                actor.reset_rng.bit_generator.state=data['reset_rng'];actor.episode_id=data['episode_id'];actor.cumulative=data['cumulative'];pipe.send(dict(restored=True))
            elif cmd=='gradient':
                from ppo_update import remote_gradient
                pipe.send(remote_gradient(actor.model,device=f'cuda:{logical}',**data))
    except Exception:
        try:pipe.send(dict(fatal=True,traceback=traceback.format_exc()))
        except Exception:pass
    finally:
        if actor is not None:actor.close()
        pipe.close()

def evaluation_cases(reg):
    return [dict(id=f'pick-{i:02d}',spec=dict(reg[0],skill='pick'),seed=90000000+i)
            for i in range(CFG['evaluation_episodes_per_skill'])]

def metric_eval(results):
    d={}
    for skill in SKILLS:
        rows=[r for r in results if r['case']['spec']['skill']==skill];valid=[r for r in rows if r['valid_reset']]
        d[f'eval/{skill}_success']=sum(r['success'] for r in valid)/len(valid) if valid else 0.
        d[f'eval/{skill}_valid']=len(valid);d[f'eval/{skill}_invalid_reset']=len(rows)-len(valid)
    d['eval/macro_success']=sum(d[f'eval/{s}_success'] for s in SKILLS)/len(SKILLS)
    return d

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

