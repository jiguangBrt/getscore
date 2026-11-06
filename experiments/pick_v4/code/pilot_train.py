"""Bounded single-GPU pilot. Stop after requested complete episodes, always evaluate."""
import argparse
import gc
import hashlib
import json
import shutil
import signal
import time
from collections import defaultdict
from pathlib import Path
from actor_core import Actor, evaluation_cases, event, rng_state, restore_rng
from train_runtime import *
from ppo_update import compute_gradients
from value_context import CONTEXT_FIELDS


def append(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')


def explained_variance(values, returns):
    variance = np.var(returns)
    return float(1.-np.var(np.asarray(returns)-values)/variance) if variance>1e-12 else None


def stage(context):
    return 'hold' if context[1]>0 else ('lift' if context[9]>0 else ('contact' if context[8]>0 else 'approach'))


@torch.no_grad()
def diagnostics(model, data):
    kls, ratios, predictions = [], [], []
    for off in range(0,len(data),8):
        rows=data[off:off+8]
        result=model(forward_inputs=batch_tree([r['forward_inputs'] for r in rows],'cuda:0'))
        lp=result['logprobs'];old=torch.cat([r['old_logprobs'] for r in rows],0).to(lp.device)
        mask=torch.as_tensor(np.stack([r['valid_action_mask'] for r in rows]),device=lp.device).unsqueeze(-1).expand_as(lp)
        logratio=((lp-old)*mask).sum((1,2));ratio=logratio.exp();kl=ratio-1-logratio
        assert torch.isfinite(kl).all() and torch.isfinite(result['values']).all()
        kls.extend(kl.cpu().tolist());ratios.extend(ratio.cpu().tolist())
        predictions.extend(result['values'].reshape(-1).cpu().tolist())
    returns=np.array([r['return'] for r in data]);before=np.array([r['old_value'] for r in data])
    return dict(approx_kl=float(np.mean(kls)),max_sample_kl=float(np.max(kls)),
                ratio_mean=float(np.mean(ratios)),clip_fraction=float(np.mean(np.abs(np.asarray(ratios)-1)>.2)),
                diagnostic_chunks=len(data),value_explained_variance_before=explained_variance(before,returns),
                value_explained_variance_after=explained_variance(predictions,returns),
                value_mse_before=float(np.mean((returns-before)**2)),
                value_mse_after=float(np.mean((returns-np.array(predictions))**2))),predictions


def update_policy(model,opt,data,update,actor_params,critic_params):
    groups={}
    for r in data:groups.setdefault(r['sampling_group'],[]).append(r)
    ordered=[]
    for key in np.random.permutation(list(groups)):
        group=sorted(groups[str(key)],key=lambda r:r['sampling_index'])
        assert [r['sampling_index'] for r in group]==list(range(8))
        ordered.extend(group)
    adv=np.asarray([r['advantage'] for r in ordered])
    factor=0. if update<CFG['critic_warmup_updates'] else min(1.,(update-CFG['critic_warmup_updates']+1)/CFG['actor_warmup_updates'])
    opt.param_groups[0]['lr']=CFG['actor_lr']*factor
    metrics=compute_gradients(model,ordered,update,8,len(ordered),float(adv.mean()),float(adv.std()),'cuda:0')
    # Prevent unrelated critic gradients from suppressing the actor step.
    actor_norm=torch.nn.utils.clip_grad_norm_(actor_params,CFG['max_grad_norm'])
    critic_norm=torch.nn.utils.clip_grad_norm_(critic_params,CFG['max_grad_norm'])
    if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):raise FloatingPointError('nonfinite gradient')
    # A rejected candidate is rolled back before we stop the run.
    previous=trainable_state(model)
    opt.step()
    post,predictions=diagnostics(model,ordered)
    if post['approx_kl']>CFG['kl_stop']:
        apply_state(model,previous)
        raise FloatingPointError(f'candidate rejected and restored: KL {post["approx_kl"]}')
    metrics.update(post,actor_gradient_norm=float(actor_norm),critic_gradient_norm=float(critic_norm),
                   actor_lr=opt.param_groups[0]['lr'],critic_lr=opt.param_groups[1]['lr'],
                   advantage_mean=float(adv.mean()),advantage_std=float(adv.std()),critic_warmup=factor==0.,
                   peak_gib=torch.cuda.max_memory_allocated(0)/2**30)
    samples=[]
    for r,p in zip(ordered,predictions):
        ctx=r['forward_inputs']['critic_context'].reshape(-1).tolist()
        samples.append(dict(episode_id=r['episode_id'],episode_native_step=r['episode_native_step'],native_steps=r['native_steps'],
                            sampling_version=update,phase=stage(ctx),critic_context=ctx,
                            old_value=r['old_value'],new_value=p,gae_return=r['return'],advantage=r['advantage'],
                            discounted_chunk_reward=r['reward'],terminated=r['terminated']))
    del previous
    model.zero_grad(set_to_none=True)
    return metrics,samples


def eval_metrics(result):
    valid=[r for r in result['results'] if r['valid_reset']]
    finals=[r['final'] for r in valid]
    return dict(valid=len(valid),invalid_reset=len(result['results'])-len(valid),
                successes=sum(r['success'] for r in valid),success_rate=float(np.mean([r['success'] for r in valid])) if valid else 0.,
                mean_return=float(np.mean([r['episode_reward'] for r in finals])) if finals else None,
                reasons=dict(__import__('collections').Counter(r['termination_reason'] for r in finals)))


def train(args):
    stop_requested={'signal':None}
    def request_stop(signum,frame):stop_requested['signal']=signum
    previous_handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if not args.smoke:
        assert CFG['runtime_implementation_status']=='validated_v4_pilot', 'physical and optimizer smoke gates required'
        physical=json.loads(Path(CFG['preflight_physical_report']).read_text())
        assert physical['repeat_initial_state_match'] and physical['repeat_reward_match']
        proof=physical['results']
        assert sum(r.get('success',False) for r in proof if r.get('control')=='positive')>=3
        assert any(r.get('success') and r.get('airborne_pad_misses',0)>0 for r in proof)
        assert all(not r.get('success') and not r.get('verified_grasp_steps') for r in proof if r.get('control') in ('table_only','open_lift'))
        assert any(r.get('control')=='recover' and r.get('success') and r.get('recovered_after_soft') for r in proof)
        assert any(r.get('control')=='severe' and r.get('reason')=='severe_xy_boundary' for r in proof)
        smoke=json.loads(Path(CFG['optimizer_smoke_report']).read_text())
        assert smoke['state']=='completed' and smoke['update']>=2
        smoke_metrics=[json.loads(x)['learner'] for x in Path(CFG['optimizer_smoke_report']).with_name('metrics.jsonl').read_text().splitlines()]
        assert any(m['actor_gradient_norm']>0 for m in smoke_metrics), 'smoke must exercise actor update'
        assert max(m['max_logprob_recompute_error'] for m in smoke_metrics)<=CFG['recompute_max_error']
        if CFG.get('sampling_preflight_report'):
            sampling=json.loads(Path(CFG['sampling_preflight_report']).read_text())['results']
            selected=next(r for r in sampling if r['noise_level']==CFG['noise_level'])
            assert selected['successes']>0 and any(e['success'] and e['episode_reward']>0 for e in selected['episodes'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for sub in ['scratch','checkpoints','value-samples','source']: (out/sub).mkdir()
    save(out/'effective-config.json',CFG)
    save(out/'arguments.json',vars(args))
    manifest={}
    for f in Path(__file__).parent.iterdir():
        if f.suffix in ('.py','.bddl'):
            shutil.copy2(f,out/'source'/f.name);manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    launcher=Path(__file__).parents[1]/'run.sh'
    if launcher.exists():
        shutil.copy2(launcher,out/'source'/'run.sh');manifest['run.sh']=hashlib.sha256(launcher.read_bytes()).hexdigest()
    save(out/'source-manifest.json',manifest)
    save(out/'critic-context-fields.json',CONTEXT_FIELDS)
    hasher=hashlib.sha256()
    with open(CFG['base_checkpoint'],'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):hasher.update(block)
    assert hasher.hexdigest()==CFG['base_sha256']
    seed_all(CFG['seed']);torch.cuda.set_device(0)
    actor=None;update=steps=completed=0;start=time.time();best=-1.;best_return=-float('inf');pending=defaultdict(list)
    resumed_completed=resumed_steps=resumed_update=0
    try:
        actor=Actor(0,0,out,8);actor.version=0;model=actor.model
        ap=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')]
        vp=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')]
        opt=torch.optim.AdamW([dict(params=ap,lr=CFG['actor_lr']),dict(params=vp,lr=CFG['critic_lr'])],
                             betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'],weight_decay=CFG['weight_decay'])
        if args.initial:
            ck=torch.load(args.initial,map_location='cpu',weights_only=False)
            assert ck['config']['task_version']==CFG['task_version']
            assert ck['base_sha256']==CFG['base_sha256']
            apply_state(model,ck['trainable_state'])
            save(out/'initial-source.json',dict(path=args.initial,weight_only=True,optimizer_reset=True,parent_update=ck['update'],
                 parent_initialization=ck.get('initialization','RL checkpoint'),data_provenance=ck.get('data_provenance')))
        if getattr(args,'resume',None):
            assert not args.initial,'choose --initial or --resume, not both'
            ck=torch.load(args.resume,map_location='cpu',weights_only=False)
            contract=['task_version','base_sha256','action_horizon','execute_horizon','flow_steps','noise_method',
                      'noise_level','joint_logprob','lora_rank','lora_alpha','actor_lr','critic_lr','adam_betas',
                      'adam_eps','weight_decay','max_grad_norm','gamma_per_native_step','gae_lambda_per_chunk',
                      'ppo_clip','value_clip','huber_delta','critic_loss_coefficient','reward','pick','success',
                      'prompt_templates','ppo_epochs','ppo_minibatch_chunks','critic_warmup_updates',
                      'actor_warmup_updates','control_hz','max_native_steps_per_skill']
            assert all(ck['config'].get(k)==CFG.get(k) for k in contract),'resume changes training contract; use explicit weight-only --initial'
            assert ck['source_manifest']['single_mug.bddl']==manifest['single_mug.bddl'],'resume scene changed'
            apply_state(model,ck['trainable_state']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
            actor.reset_rng.bit_generator.state=ck['actor_reset_rng'];actor.episode_id=ck['actor_episode_id']
            actor.cumulative=ck['actor_cumulative']
            discarded=sum(a is not None for a in ck['active_episodes'])
            actor.cumulative['discarded_on_resume']=actor.cumulative.get('discarded_on_resume',0)+discarded
            update=resumed_update=ck['update'];steps=resumed_steps=ck['native_steps'];completed=resumed_completed=ck['completed_episodes']
            actor.version=update
            state=trainable_state(model)
            assert all(torch.equal(v,ck['trainable_state'][k]) for k,v in state.items())
            save(out/'resume-source.json',dict(path=args.resume,optimizer_and_rng_restored=True,parameter_max_error=0.,
                 parent_update=update,parent_native_steps=steps,parent_completed_episodes=completed,
                 optimizer_state_steps=sorted({float(s['step'].item()) for s in opt.state.values() if 'step' in s}),
                 discarded_unfinished_environments=ck['active_episodes'],discarded_count=discarded,
                 simulator_resume='fresh resets; simulator states were not serialized'))
            event(out,'resume_discards_unfinished',count=discarded,parent_update=update)
        save(out/'model.json',dict(total_parameters=sum(p.numel() for p in model.parameters()),actor_parameters=sum(p.numel() for p in ap),
                                 critic_parameters=sum(p.numel() for p in vp),lora_targets=model.lora_targets,
                                 trainable_names=[n for n,p in model.named_parameters() if p.requires_grad]))
        def checkpoint(name):
            atomic_torch(out/'checkpoints'/name,dict(trainable_state=trainable_state(model),optimizer=cpu_tree(opt.state_dict()),
                         rng=rng_state(),update=update,native_steps=steps,completed_episodes=completed,config=CFG,
                         actor_reset_rng=actor.reset_rng.bit_generator.state,actor_episode_id=actor.episode_id,actor_cumulative=actor.cumulative,
                         source_manifest=manifest,base_sha256=CFG['base_sha256'],base_checkpoint=CFG['base_checkpoint'],
                         active_episodes=[None if a is None else {k:v for k,v in a.items() if k!='obs'} for a in actor.active],
                         environment_resume='unfinished environments not serialized; any restart must log discarded episodes'))
        cases=evaluation_cases(actor.reg);save(out/'evaluation-cases.json',cases)
        def evaluate(tag):
            nonlocal best,best_return
            event(out,'evaluation_start',tag=tag,update=update)
            result=actor.evaluate(cases,tag);metrics=eval_metrics(result)
            save(out/f'evaluation-{tag}.json',dict(result,metrics=metrics,update=update))
            event(out,'evaluation_end',tag=tag,update=update,**metrics)
            mean_return=metrics['mean_return'] if metrics['mean_return'] is not None else -float('inf')
            if metrics['success_rate']>best or (metrics['success_rate']==best and mean_return>best_return):
                best=metrics['success_rate'];best_return=mean_return;checkpoint('best.pt')
            checkpoint(f'evaluation-{tag}.pt')
            return metrics
        checkpoint('fresh-start.pt');baseline=evaluate('baseline')
        while completed<resumed_completed+args.episodes and update<resumed_update+args.max_updates and time.time()-start<args.hours*3600 and not stop_requested['signal']:
            tick=time.time()
            event(out,'rollout_start',update=update,completed_episodes=completed)
            rr=actor.rollout(update,time.time()-start,args.chunks,update%CFG['video_training_every_updates']==0)
            data=torch.load(rr['path'],map_location='cpu',weights_only=False)
            steps+=rr['native_steps'];completed+=len(rr['episodes'])
            if update==0:atomic_torch(out/'verification-batch.pt',data)
            lm,samples=update_policy(model,opt,data,update,ap,vp)
            update+=1;actor.version=update
            save(out/'value-samples'/f'update-{update:04d}.json',samples)
            # Empirical complete-episode return, separate from bootstrapped GAE targets.
            for sample in samples:pending[sample['episode_id']].append(sample)
            for ep in rr['episodes']:
                append(out/'episodes.jsonl',dict(ep,update=update))
                trajectory=sorted(pending.pop(ep['episode_id'],[]),key=lambda r:r['episode_native_step'])
                future=0.
                for sample in reversed(trajectory):
                    future=sample['discounted_chunk_reward']+CFG['gamma_per_native_step']**sample['native_steps']*future
                    sample['empirical_return']=future
                append(out/'completed-value-trajectories.jsonl',dict(episode_id=ep['episode_id'],success=ep['success'],
                       versions=sorted({s['sampling_version'] for s in trajectory}),samples=trajectory))
            er=rr['episodes'];finals=[e['final'] for e in er]
            metrics=dict(update=update,native_steps=steps,completed_episodes=completed,
                         elapsed_seconds=time.time()-start,iteration_seconds=time.time()-tick,learner=lm,
                         rollout_seconds=rr['seconds'],reset_errors=rr['reset_errors'],continued=rr['continued'],
                         iteration_episodes=len(er),iteration_successes=sum(e['success'] for e in er),
                         mean_episode_reward=float(np.mean([e['episode_reward'] for e in er])) if er else None,
                         reasons=dict(__import__('collections').Counter(f['termination_reason'] for f in finals)),
                         cumulative=rr['cumulative'],train_video=rr['video'])
            append(out/'metrics.jsonl',metrics);atomic_json(out/'status.json',dict(state='training',**metrics))
            event(out,'update_complete',update=update,native_steps=steps,completed_episodes=completed,
                  successes=sum(e['success'] for e in er),seconds=time.time()-tick,kl=lm['approx_kl'],value_mse=lm['value_mse_after'])
            Path(rr['path']).unlink()
            if update%CFG['checkpoint_every_updates']==0:checkpoint('latest.pt')
            if update%CFG['evaluate_every_updates']==0:evaluate(f'update-{update:04d}')
            del data,samples;gc.collect()
        checkpoint('final.pt');final=evaluate('final');checkpoint('latest.pt')
        result=dict(state='completed',update=update,native_steps=steps,completed_episodes=completed,
                    elapsed_seconds=time.time()-start,baseline=baseline,final=final,best_success_rate=best,
                    requested_episodes=args.episodes,new_completed_episodes=completed-resumed_completed,
                    new_native_steps=steps-resumed_steps,new_updates=update-resumed_update,
                    resumed_completed_episodes=resumed_completed,
                    stop_reason='operator_stop' if stop_requested['signal'] else ('episode_budget' if completed>=resumed_completed+args.episodes else 'time_or_update_budget'),
                    budget_truncated_active=[None if a is None else {k:v for k,v in a.items() if k!='obs'} for a in actor.active])
        atomic_json(out/'status.json',result);event(out,'finished',**result)
    except Exception:
        import traceback
        failure=dict(state='failed',update=update,native_steps=steps,completed_episodes=completed,traceback=traceback.format_exc())
        atomic_json(out/'status.json',failure);event(out,'fatal',**failure)
        raise
    finally:
        if actor is not None:actor.close()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--episodes',type=int,default=1000);parser.add_argument('--max-updates',type=int,default=300)
    parser.add_argument('--hours',type=float,default=6.);parser.add_argument('--chunks',type=int,default=32)
    parser.add_argument('--initial',help='Optional same-v4 weight-only initialization, explicitly recorded')
    parser.add_argument('--resume',help='Restore optimizer/RNG/counters; explicitly discard unfinished simulator episodes')
    parser.add_argument('--smoke',action='store_true',help='Explicit preflight run before the pilot launch gate')
    train(parser.parse_args())
