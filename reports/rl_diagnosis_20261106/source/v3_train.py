"""Synchronous on-policy joint-skill PPO, two batched actors and parallel MuJoCo workers."""
from config_contract import guard_runtime
guard_runtime('rpent-skills-v3-20261104')
from train_runtime import *
import argparse,json,time,os,sys,traceback,multiprocessing as mp,gc,subprocess,shutil,signal
from collections import defaultdict
from v3_skill_env import inventory,worker as env_worker

PROFILE=Path(__file__).resolve().parents[1]
CODE=PROFILE/'code'
SKILLS=['pick','place','upright']
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
    types=['porcelain_mug','white_yellow_mug']
    if elapsed>=3600:types+=['moka_pot']
    allowed=[r for r in reg if r['type'] in types]
    # At 3h all original movable types join; retain 60% core objects.
    if elapsed>=10800 and rng.random()>=.60:allowed=reg
    skill=str(rng.choice(SKILLS,p=CFG['task_weights']))
    if skill=='upright':allowed=[r for r in allowed if r.get('upright_eligible',True)]
    base=dict(allowed[int(rng.integers(len(allowed)))]);base['skill']=skill
    return base

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
                episodes.append(dict(episode_id=self.active[i]['episode_id'],seed=self.active[i]['seed'],task_index=specs[i]['task_index'],native_steps=self.active[i]['steps'],episode_reward=self.active[i]['reward'],skill=specs[i]['skill'],object_type=specs[i]['type'],success=bool(r['info']['skill_success']),failure=bool(r['info']['failure']),timeout=bool(r['truncated']),final=r['info']))
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
                if not any(v['skill']==skill for v in videos):
                    p=self.out/'videos'/f'eval-{tag}-actor{self.aid}-{skill}.mp4';p.parent.mkdir(exist_ok=True)
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
    # Core mugs/pot plus other long objects; fixed held-out seeds, no training data reuse.
    core=[r for typ in ['porcelain_mug','white_yellow_mug','moka_pot'] for r in reg if r['type']==typ]
    cases=[];n=CFG['evaluation_episodes_per_skill']
    for skill in SKILLS:
        eligible=[r for r in reg if skill!='upright' or r.get('upright_eligible',True)]
        for i in range(n):
            base=core[i%len(core)] if i<max(18,n*3//4) else eligible[(i-max(18,n*3//4))%len(eligible)]
            cases.append(dict(id=f'{skill}-{i:02d}',spec=dict(base,skill=skill),seed=90000000+SKILLS.index(skill)*1000+i))
    return cases

def metric_eval(results):
    d={}
    for skill in SKILLS:
        rows=[r for r in results if r['case']['spec']['skill']==skill];valid=[r for r in rows if r['valid_reset']]
        d[f'eval/{skill}_success']=sum(r['success'] for r in valid)/len(valid) if valid else 0.
        d[f'eval/{skill}_valid']=len(valid);d[f'eval/{skill}_invalid_reset']=len(rows)-len(valid)
    d['eval/macro_success']=sum(d[f'eval/{s}_success'] for s in SKILLS)/3
    return d

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def train(out,args):
    import wandb
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'scratch').mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    cfg_path=out/'effective-config.json'
    if cfg_path.exists():assert json.loads(cfg_path.read_text())==CFG,'immutable run config changed'
    else:save(cfg_path,CFG)
    reg=inventory();save(out/'object-inventory.json',reg);cases=evaluation_cases(reg)
    if args.preflight:
        cases=[dict(id=f'{skill}-{i}',spec=dict(next(r for r in reg if r['type']=='porcelain_mug'),skill=skill),seed=90000000+si*1000+i) for si,skill in enumerate(SKILLS) for i in range(1)]
    save(out/'evaluation-cases.json',cases);save(out/'launch-arguments.json',vars(args))
    meta_path=out/'run-meta.json'
    if meta_path.exists():meta=json.loads(meta_path.read_text())
    else:meta=dict(started_at=time.time(),deadline=time.time()+args.hours*3600,run_name=out.name);save(meta_path,meta)
    import hashlib
    source_manifest={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')}
    source_manifest['git_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True).stdout.strip() or 'not-yet-versioned'
    save(out/'source-manifest.json',source_manifest)
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for source in CODE.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    shutil.copy2(PROFILE/'run.sh',snapshot/'run.sh')
    basehash=hashlib.sha256()
    with open(CFG['base_checkpoint'],'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):basehash.update(block)
    assert basehash.hexdigest()==CFG['base_sha256'],'base checkpoint hash mismatch'
    if not args.preflight:
        assert CFG['runtime_implementation_status']=='validated_v3','v3 behavioral preflight gate not passed'
        from overnight_control import validate_launch
        validate_launch(CFG,meta,out,args)

    seed_all(CFG['seed']);torch.cuda.set_device(0);model=build_model('cuda:0')
    actor=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')];critic=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')]
    opt=torch.optim.AdamW([dict(params=actor,lr=CFG['actor_lr']),dict(params=critic,lr=CFG['critic_lr'])],betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'],weight_decay=CFG['weight_decay'])
    update=0;total_steps=0;last_eval=0;last_save=0;last_eval_update=-1;last_eval_tag=None;best=-1.;micro=CFG['micro_batch_size']
    resume_actor_states=None
    latest=out/'checkpoints/latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False);assert ck['config']==CFG and ck['run_meta']==meta,'foreign checkpoint';event(out,'restart_discards_unfinished',active=ck.get('active_episodes',[]),reason='simulators not serialized; no bit-exact resume');apply_state(model,ck['trainable_state']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);update=ck['update'];total_steps=ck['native_steps'];last_eval=ck['last_eval'];last_save=ck['last_milestone'];best=ck['best_eval'];resume_actor_states=ck.get('actor_states');micro=ck.get('microbatch',micro);last_eval_update=ck.get('last_eval_update',-1);last_eval_tag=ck.get('last_eval_tag')
    save(out/'model.json',dict(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),lora_targets=model.lora_targets,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]))
    run=wandb.init(project=CFG['wandb_project'],name=out.name,group=out.name,job_type='train',mode=CFG['wandb_mode'],dir=str(out),config=CFG,settings=wandb.Settings(init_timeout=30,x_stats_gpu_device_ids=[0,1,2,3]))
    run.define_metric('update');run.define_metric('*',step_metric='update')
    save(out/'wandb.json',dict(mode=CFG['wandb_mode'],run_id=run.id,dir=run.dir,url=run.url,project=run.project,entity=run.entity,sync='automatic online sync; local run files retained'))
    ctx=mp.get_context('spawn');actors=[]
    for aid,logical in enumerate([1,2,3]):
        a,b=ctx.Pipe();p=ctx.Process(target=actor_worker,args=(b,logical,aid,str(out),args.envs//3));p.start();b.close();actors.append((a,p))
    for a,p in actors:assert recv(a,300)['ready']
    if resume_actor_states:
        for (a,p),state in zip(actors,resume_actor_states):a.send(('restore',state))
        for a,p in actors[:len(resume_actor_states)]:assert recv(a)['restored']
    event(out,'ready',update=update,actors=len(actors),envs=args.envs,pids=[p.pid for a,p in actors],deadline=meta['deadline'],wandb_mode=CFG['wandb_mode'])
    active_episodes=[]
    def checkpoint(name):
        for a,p in actors:a.send(('state',None))
        actor_states=[recv(a) for a,p in actors]
        ck=dict(resource_migration=json.loads((out/'MIGRATION.json').read_text()) if (out/'MIGRATION.json').exists() else None,actor_states=actor_states,active_episodes=active_episodes,environment_resume='unfinished episodes discarded on process restart',trainable_state=trainable_state(model),optimizer=cpu_tree(opt.state_dict()),rng=rng_state(),config=CFG,source_manifest=source_manifest,base_checkpoint=str(BASE/'checkpoints/pi05_libero_torch/model.safetensors'),base_sha256='f841dc041275ae9df425eedfb7738d84d86e5197e8e47fbfea754eff67ca1a59',update=update,native_steps=total_steps,last_eval=last_eval,last_eval_update=last_eval_update,last_eval_tag=last_eval_tag,last_milestone=last_save,best_eval=best,microbatch=micro,run_meta=meta)
        atomic_torch(out/'checkpoints'/name,ck);event(out,'checkpoint',path=name,update=update,native_steps=total_steps)
    def sync():
        path=out/'scratch/current-policy.pt';atomic_torch(path,dict(trainable_state=trainable_state(model),version=update))
        for a,p in actors:a.send(('sync',dict(path=str(path),version=update)))
        for a,p in actors:assert recv(a,180)['synced']==update
    def evaluate(tag):
        nonlocal best,last_eval,last_eval_update,last_eval_tag
        event(out,'evaluation_start',tag=tag,update=update)
        for aid,(a,p) in enumerate(actors):a.send(('eval',dict(cases=cases[aid::len(actors)],tag=tag)))
        rr=[recv(a,1800) for a,p in actors];rows=sum((r['results'] for r in rr),[]);metrics=metric_eval(rows)
        save(out/f'evaluation-{tag}.json',dict(tag=tag,update=update,metrics=metrics,results=rows,videos=sum((r['videos'] for r in rr),[]),training_state_preserved=all(r['training_state_preserved'] for r in rr),environment_checks=[{k:v for k,v in r.items() if k in ('training_before','training_after')} for r in rr]))
        run.log(dict(metrics,update=update,native_steps=total_steps),step=update)
        for record in sum((r['videos'] for r in rr),[]):
            run.log({f'eval_video/{record["skill"]}':wandb.Video(record['path'],format='mp4'), 'update':update},step=update)
        if metrics['eval/macro_success']>best:best=metrics['eval/macro_success'];checkpoint('best.pt')
        last_eval=time.time()-meta['started_at'];last_eval_update=update;last_eval_tag=tag;event(out,'evaluation_end',tag=tag,update=update,**metrics)
    try:
        sync()
        if not latest.exists():checkpoint('fresh-start.pt')
        if not (out/'evaluation-baseline.json').exists():evaluate('baseline');last_eval=0.;checkpoint('initial.pt');checkpoint('latest.pt')
        if (out/'CONTINUATION.json').exists():
            event(out,'continuation_source',**json.loads((out/'CONTINUATION.json').read_text()))
        while time.time()<meta['deadline'] and update<min(CFG['max_optimizer_steps'],args.max_updates):
            elapsed=time.time()-meta['started_at'];start=time.time()
            event(out,'rollout_start',update=update,elapsed=elapsed)
            for a,p in actors:a.send(('rollout',dict(version=update,elapsed=elapsed+CFG['curriculum_offset_seconds'],chunks=args.chunks,video=update%CFG['video_training_every_updates']==0)))
            rr=[recv(a,1200) for a,p in actors];data=[]
            for r in rr:data.extend(torch.load(r['path'],map_location='cpu',weights_only=False))
            total_steps+=sum(r['native_steps'] for r in rr)
            active_episodes=[dict(actor=aid,**a) for aid,r in enumerate(rr) for a in r['active'] if a is not None]
            if args.preflight and update==0:atomic_torch(out/'verification-batch.pt',data)
            for retry in range(4):
                try:
                    from ppo_update import parallel_update
                    lm=parallel_update(model,opt,data,update,micro,actors,out);break
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    raise RuntimeError('fixed batch8 OOM; cannot change sample grouping')
            update+=1;sync()
            for r in rr:
                with open(out/'episodes.jsonl','a') as f:
                    for ep in r['episodes']:f.write(json.dumps(dict(ep,update=update,trainer_pid=os.getpid()),ensure_ascii=False)+'\n')
            metrics={f'train/{k}':v for k,v in lm.items() if isinstance(v,(float,int,bool))}
            episodes=sum((r['episodes'] for r in rr),[])
            for skill in SKILLS:
                er=[e for e in episodes if e['skill']==skill]
                metrics[f'train/{skill}_completed']=len(er)
                for outcome in ('failure','timeout'):metrics[f'train/{skill}_{outcome}_count']=sum(e[outcome] for e in er)
                metrics[f'train/{skill}_success']=sum(e['success'] for e in er)/len(er) if er else 0.
                if er:metrics[f'train/{skill}_episode_reward']=float(np.mean([e['episode_reward'] for e in er]));metrics[f'train/{skill}_episode_steps']=float(np.mean([e['native_steps'] for e in er]))
                sr=[r for r in data if r['skill']==skill]
                if sr:metrics[f'train/{skill}_chunk_reward']=float(np.mean([r['undiscounted_reward'] for r in sr]));metrics[f'train/{skill}_chunks']=len(sr)
            metrics.update({'train/native_steps':total_steps,'train/update':update,'train/iteration_seconds':time.time()-start,'train/native_steps_per_second':sum(r['native_steps'] for r in rr)/(time.time()-start),'train/reset_errors':sum(len(r['reset_errors']) for r in rr),'system/elapsed_hours':(time.time()-meta['started_at'])/3600})
            metrics['update']=update
            for scene in sorted({r['task_index'] for r in data}):metrics[f'coverage/scene{scene}_chunks']=sum(r['task_index']==scene for r in data)
            for typ in sorted({r['object_type'] for r in data}):metrics[f'coverage/{typ}_chunks']=sum(r['object_type']==typ for r in data)
            run.log(metrics,step=update)
            for r in rr:
                if r.get('video'):run.log({f'train_video/gpu{r["physical_gpu"]}':wandb.Video(r['video'],format='mp4'),'update':update},step=update)
            summary=dict(trainer_pid=os.getpid(),update=update,elapsed_seconds=time.time()-meta['started_at'],native_steps=total_steps,learner=lm,rollouts=[{k:v for k,v in r.items() if k!='episodes'} for r in rr],metrics=metrics)
            with open(out/'metrics.jsonl','a') as f:f.write(json.dumps(summary)+'\n')
            atomic_json(out/'status.json',dict(state='training',**summary,deadline=meta['deadline'],wandb_mode=CFG['wandb_mode']))
            event(out,'update_complete',update=update,native_steps=total_steps,iteration_seconds=time.time()-start,actor_loss=lm['actor_loss'],critic_loss=lm['critic_loss'])
            for r in rr:Path(r['path']).unlink(missing_ok=True)
            # Reset artifacts contain only compact JSON; periodic checkpoint resume is at a clean rollout boundary.
            elapsed=time.time()-meta['started_at']
            if update%CFG['latest_every_updates']==0 or args.preflight:checkpoint('latest.pt')
            if args.preflight and update==1:evaluate('continuity-check');checkpoint('latest.pt')
            if milestone_due(elapsed,last_save,CFG['checkpoint_interval_seconds']):
                last_save=elapsed;checkpoint(f'milestone-{elapsed/3600:.1f}h.pt');checkpoint('latest.pt')
            if milestone_due(elapsed,last_eval,CFG['evaluation_interval_seconds']):
                evaluate(f'{elapsed/3600:.1f}h');checkpoint('latest.pt')
            del data;gc.collect()
        event(out,'run_budget_truncated',active=active_episodes,count=len(active_episodes))
        sync();checkpoint('final.pt')
        # Final policy is evaluated at the end even if the last periodic boundary falls slightly early.
        if last_eval_update!=update:evaluate('final')
        else:
            final_report=json.loads((out/f'evaluation-{last_eval_tag}.json').read_text());final_report['source_tag']=last_eval_tag;final_report['tag']='final';save(out/'evaluation-final.json',final_report)
        checkpoint('final.pt');checkpoint('latest.pt')
        result=dict(state='completed',update=update,native_steps=total_steps,elapsed_hours=(time.time()-meta['started_at'])/3600,best_eval=best,finished_at=time.time(),final_checkpoint=str(out/'checkpoints/final.pt'))
        atomic_json(out/'status.json',result);event(out,'finished',**result);run.summary.update(result)
    except Exception:
        event(out,'fatal',traceback=traceback.format_exc(),update=update)
        # Preserve the last clean optimizer boundary; incomplete gradients are not checkpointed.
        raise
    finally:
        for a,p in actors:
            try:a.send(('close',None))
            except Exception:pass
        for a,p in actors:
            p.join(45)
            if p.is_alive():p.terminate();p.join(10)
        run.finish()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--hours',type=float,default=6.);parser.add_argument('--envs',type=int,default=24);parser.add_argument('--chunks',type=int,default=32);parser.add_argument('--max-updates',type=int,default=100000);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    train(args.out,args)
