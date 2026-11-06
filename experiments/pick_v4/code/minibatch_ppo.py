"""Alternative pilot updater: standard clipped PPO over multiple minibatches.

Not used by pilot A. Original sampling groups stay intact, and exact on-policy
recompute is checked before the first step, not after the policy has changed.
"""
import argparse
import copy
import time
import pilot_train as runner
from train_runtime import *


@torch.no_grad()
def check_initial_recompute(model, data):
    lp_error=value_error=0.
    for off in range(0,len(data),8):
        rows=data[off:off+8]
        r=model(forward_inputs=batch_tree([x['forward_inputs'] for x in rows],'cuda:0'))
        old=torch.cat([x['old_logprobs'] for x in rows],0).to(r['logprobs'].device)
        lp_error=max(lp_error,float((r['logprobs']-old).abs().max()))
        ov=torch.tensor([x['old_value'] for x in rows],device='cuda:0')
        value_error=max(value_error,float((r['values'].reshape(-1)-ov).abs().max()))
    if lp_error>CFG['recompute_max_error'] or value_error>1e-4:
        raise RuntimeError(f'initial on-policy mismatch: logprob={lp_error}, value={value_error}')
    return lp_error,value_error


def gradient(model,rows,adv_mean,adv_std,warmup):
    from rlinf.algorithms.losses import compute_ppo_actor_loss,compute_ppo_critic_loss
    model.zero_grad(set_to_none=True)
    losses=[]
    for off in range(0,len(rows),8):
        group=rows[off:off+8]
        result=model(forward_inputs=batch_tree([r['forward_inputs'] for r in group],'cuda:0'))
        lp=result['logprobs'];old=torch.cat([r['old_logprobs'] for r in group],0).to(lp.device)
        mask=torch.as_tensor(np.stack([r['valid_action_mask'] for r in group]),device=lp.device).unsqueeze(-1).expand_as(lp)
        adv=torch.tensor([(r['advantage']-adv_mean)/(adv_std+1e-8) for r in group],device=lp.device)
        al,_=compute_ppo_actor_loss(logprobs=(lp*mask).sum((1,2)),old_logprobs=(old*mask).sum((1,2)),
                                    clip_ratio_low=CFG['ppo_clip'],clip_ratio_high=CFG['ppo_clip'],advantages=adv)
        vl,_=compute_ppo_critic_loss(values=result['values'].reshape(-1),
                                   returns=torch.tensor([r['return'] for r in group],device=lp.device),
                                   prev_values=torch.tensor([r['old_value'] for r in group],device=lp.device),
                                   value_clip=CFG['value_clip'],huber_delta=CFG['huber_delta'])
        loss=((0. if warmup else 1.)*al+CFG['critic_loss_coefficient']*vl)*len(group)/len(rows)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite minibatch loss')
        loss.backward();losses.append((float(al.detach()),float(vl.detach()),len(group)))
    return dict(actor_loss=sum(a*n for a,v,n in losses)/len(rows),critic_loss=sum(v*n for a,v,n in losses)/len(rows),chunks=len(rows))


def minibatch_update(model,opt,data,update,actor_params,critic_params):
    start=time.time();groups={}
    for r in data:
        assert r['version']==update
        groups.setdefault(r['sampling_group'],[]).append(r)
    groups=[sorted(g,key=lambda r:r['sampling_index']) for g in groups.values()]
    assert all([r['sampling_index'] for r in g]==list(range(8)) for g in groups)
    ordered=[r for g in groups for r in g]
    lp_error,value_error=check_initial_recompute(model,ordered)
    adv=np.asarray([r['advantage'] for r in ordered]);mean=float(adv.mean());std=float(adv.std())
    warmup=update<CFG['critic_warmup_updates']
    factor=0. if warmup else min(1.,(update-CFG['critic_warmup_updates']+1)/CFG['actor_warmup_updates'])
    opt.param_groups[0]['lr']=CFG['actor_lr']*factor
    all_metrics=[];actor_norms=[];critic_norms=[];accepted_steps=0;accepted_epochs=0;rejected=None
    minibatch=CFG['ppo_minibatch_chunks'];assert minibatch%8==0
    for epoch in range(CFG['ppo_epochs']):
        previous=trainable_state(model);previous_opt=copy.deepcopy(cpu_tree(opt.state_dict()))
        epoch_rows=[r for i in np.random.permutation(len(groups)) for r in groups[i]]
        epoch_metrics=[];an=[];vn=[]
        for off in range(0,len(epoch_rows),minibatch):
            rows=epoch_rows[off:off+minibatch]
            m=gradient(model,rows,mean,std,warmup)
            a=torch.nn.utils.clip_grad_norm_(actor_params,CFG['max_grad_norm'])
            v=torch.nn.utils.clip_grad_norm_(critic_params,CFG['max_grad_norm'])
            if not torch.isfinite(a) or not torch.isfinite(v):raise FloatingPointError('nonfinite minibatch gradient')
            opt.step();epoch_metrics.append(m);an.append(float(a));vn.append(float(v))
        post,predictions=runner.diagnostics(model,ordered)
        if post['approx_kl']>CFG['kl_stop']:
            apply_state(model,previous);opt.load_state_dict(previous_opt)
            rejected=dict(epoch=epoch,approx_kl=post['approx_kl'])
            post,predictions=runner.diagnostics(model,ordered)
            break
        accepted_steps+=len(epoch_metrics);accepted_epochs+=1
        all_metrics.extend(epoch_metrics);actor_norms.extend(an);critic_norms.extend(vn)
        del previous,previous_opt
        if post['approx_kl']>=CFG['kl_target']:break
    weight=sum(m['chunks'] for m in all_metrics)
    metrics=dict(actor_loss=sum(m['actor_loss']*m['chunks'] for m in all_metrics)/weight if weight else 0.,
                 critic_loss=sum(m['critic_loss']*m['chunks'] for m in all_metrics)/weight if weight else 0.,
                 max_logprob_recompute_error=lp_error,max_value_recompute_error=value_error,
                 actor_gradient_norm=float(np.mean(actor_norms)) if actor_norms else 0.,
                 critic_gradient_norm=float(np.mean(critic_norms)) if critic_norms else 0.,
                 actor_lr=opt.param_groups[0]['lr'],critic_lr=opt.param_groups[1]['lr'],critic_warmup=warmup,
                 advantage_mean=mean,advantage_std=std,chunks=len(ordered),microbatch=8,
                 optimizer_steps=accepted_steps,accepted_epochs=accepted_epochs,rejected_epoch=rejected,
                 seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated(0)/2**30,**post)
    samples=[]
    for r,p in zip(ordered,predictions):
        ctx=r['forward_inputs']['critic_context'].reshape(-1).tolist()
        samples.append(dict(episode_id=r['episode_id'],episode_native_step=r['episode_native_step'],native_steps=r['native_steps'],
                            sampling_version=update,phase=runner.stage(ctx),critic_context=ctx,
                            old_value=r['old_value'],new_value=p,gae_return=r['return'],advantage=r['advantage'],
                            discounted_chunk_reward=r['reward'],terminated=r['terminated']))
    model.zero_grad(set_to_none=True)
    return metrics,samples


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--max-updates',type=int,default=300);p.add_argument('--hours',type=float,default=6.)
    p.add_argument('--chunks',type=int,default=32);p.add_argument('--initial');p.add_argument('--resume')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not args.smoke:
        import json
        proof=json.loads(Path(CFG['optimizer_probe_report']).read_text())['metrics']
        assert proof['accepted_epochs']==2 and proof['optimizer_steps']==8
        assert proof['max_logprob_recompute_error']<=CFG['recompute_max_error'] and proof['rejected_epoch'] is None
    runner.update_policy=minibatch_update
    runner.train(args)
