"""Parallel gradients with one central optimizer; globally weighted exact batch reduction."""
from train_runtime import *
import time,gc

def compute_gradients(model,data,update,microbatch,global_count,adv_mean,adv_std,device):
    from rlinf.algorithms.losses import compute_ppo_actor_loss,compute_ppo_critic_loss
    assert {r['version'] for r in data}=={update}
    assert microbatch==CFG['envs_per_actor'] and len(data)%microbatch==0, 'BF16 recompute must preserve original actor batch'
    warmup=update<CFG['critic_warmup_updates'];model.zero_grad(set_to_none=True);metrics=[];start=time.time()
    for offset in range(0,len(data),microbatch):
        rows=data[offset:offset+microbatch];fi=batch_tree([r['forward_inputs'] for r in rows],device)
        result=model(forward_inputs=fi);lp=result['logprobs'];old=torch.cat([r['old_logprobs'] for r in rows],0).to(lp.device)
        difference=float((lp-old).abs().max().detach())
        if difference>CFG['recompute_max_error']:raise RuntimeError(f'on-policy logprob mismatch {difference}')
        mask=torch.tensor(np.stack([r['valid_action_mask'] for r in rows]),device=lp.device).unsqueeze(-1).expand_as(lp)
        lpn=(lp*mask).sum((1,2));oldn=(old*mask).sum((1,2))
        advantages=torch.tensor([(r['advantage']-adv_mean)/(adv_std+1e-8) for r in rows],device=lp.device,dtype=torch.float32)
        al,am=compute_ppo_actor_loss(logprobs=lpn,old_logprobs=oldn,clip_ratio_low=CFG['ppo_clip'],clip_ratio_high=CFG['ppo_clip'],advantages=advantages)
        vl,vm=compute_ppo_critic_loss(values=result['values'].reshape(-1),returns=torch.tensor([r['return'] for r in rows],device=lp.device,dtype=torch.float32),prev_values=torch.tensor([r['old_value'] for r in rows],device=lp.device,dtype=torch.float32),value_clip=CFG['value_clip'],huber_delta=CFG['huber_delta'])
        loss=((0. if warmup else 1.)*al+CFG['critic_loss_coefficient']*vl)*len(rows)/global_count
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
        loss.backward();metrics.append(dict(actor_loss=float(al.detach()),critic_loss=float(vl.detach()),max_logprob_recompute_error=difference,count=len(rows)))
    torch.cuda.synchronize(device)
    return dict(actor_loss=sum(x['actor_loss']*x['count'] for x in metrics)/len(data),critic_loss=sum(x['critic_loss']*x['count'] for x in metrics)/len(data),max_logprob_recompute_error=max(x['max_logprob_recompute_error'] for x in metrics),chunks=len(data),seconds=time.time()-start,microbatch=microbatch)

def remote_gradient(model,path,update,microbatch,global_count,adv_mean,adv_std,device,outpath):
    data=torch.load(path,map_location='cpu',weights_only=False)
    for _ in range(4):
        try:
            metrics=compute_gradients(model,data,update,microbatch,global_count,adv_mean,adv_std,device);break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            raise RuntimeError('OOM in fixed sampling batch: stop rather than change policy numerics')
    gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
    atomic_torch(outpath,dict(gradients=gradients,metrics=metrics));model.zero_grad(set_to_none=True)
    return dict(path=str(outpath),metrics=metrics)

@torch.no_grad()
def post_update_diagnostics(model,rows,device):
    kls=[];ratios=[]
    for off in range(0,len(rows),8):
        rr=rows[off:off+8];r=model(forward_inputs=batch_tree([x['forward_inputs'] for x in rr],device));lp=r['logprobs'];old=torch.cat([x['old_logprobs'] for x in rr],0).to(device)
        mask=torch.tensor(np.stack([x['valid_action_mask'] for x in rr]),device=device).unsqueeze(-1).expand_as(lp)
        logratio=((lp-old)*mask).sum((1,2));ratio=logratio.exp();kl=ratio-1-logratio
        if not torch.isfinite(kl).all():raise FloatingPointError('nonfinite post-update KL')
        kls.extend(kl.cpu().tolist());ratios.extend(ratio.cpu().tolist())
    return dict(approx_kl=float(np.mean(kls)),ratio_mean=float(np.mean(ratios)),clip_fraction=float(np.mean(np.abs(np.asarray(ratios)-1)>.2)))

def parallel_update(model,opt,data,update,microbatch,actors,out):
    start=time.time();global_count=len(data);adv=np.asarray([r['advantage'] for r in data]);mean=float(adv.mean());std=float(adv.std())
    groups={}
    for row in data:groups.setdefault(row['sampling_group'],[]).append(row)
    ordered=[]
    for key in np.random.permutation(list(groups)):
        group=sorted(groups[str(key)],key=lambda r:r['sampling_index'])
        assert len(group)==CFG['envs_per_actor'] and [r['sampling_index'] for r in group]==list(range(CFG['envs_per_actor']))
        ordered.append(group)
    data=[r for group in ordered for r in group]
    pieces=[[r for group in ordered[i::(len(actors)+1)] for r in group] for i in range(len(actors)+1)];paths=[]
    for aid,(pipe,proc) in enumerate(actors):
        path=Path(out)/'scratch'/f'gradient-input-v{update}-rank{aid+1}.pt';outpath=path.with_name(f'gradient-output-v{update}-rank{aid+1}.pt');paths.extend([path,outpath]);atomic_torch(path,pieces[aid+1])
        pipe.send(('gradient',dict(path=str(path),update=update,microbatch=microbatch,global_count=global_count,adv_mean=mean,adv_std=std,outpath=str(outpath))))
    warmup=update<CFG['critic_warmup_updates'];factor=0. if warmup else min(1.,(update-CFG['critic_warmup_updates']+1)/CFG['actor_warmup_updates'])
    opt.param_groups[0]['lr']=CFG['actor_lr']*factor;opt.param_groups[1]['lr']=CFG['critic_lr'];opt.zero_grad(set_to_none=True)
    for _ in range(4):
        try:local=compute_gradients(model,pieces[0],update,microbatch,global_count,mean,std,'cuda:0');break
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            raise RuntimeError('OOM in fixed sampling batch: stop rather than change policy numerics')
    all_metrics=[local];params=dict(model.named_parameters())
    for pipe,proc in actors:
        if not pipe.poll(600):raise TimeoutError('gradient worker timeout')
        r=pipe.recv()
        if r.get('fatal'):raise RuntimeError(r['traceback'])
        payload=torch.load(r['path'],map_location='cpu',weights_only=False);all_metrics.append(payload['metrics'])
        for n,g in payload['gradients'].items():
            p=params[n];g=g.to(p.device)
            if p.grad is None:p.grad=g
            else:p.grad.add_(g)
        del payload
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CFG['max_grad_norm'])
    if not torch.isfinite(grad):raise FloatingPointError('nonfinite merged gradient')
    opt.step();diagnostics=post_update_diagnostics(model,data[:32],'cuda:0')
    if diagnostics['approx_kl']>CFG['kl_stop']:raise FloatingPointError(f'PPO trust-region guard: {diagnostics}')
    for p in paths:p.unlink(missing_ok=True)
    return dict(actor_loss=sum(m['actor_loss']*m['chunks'] for m in all_metrics)/global_count,critic_loss=sum(m['critic_loss']*m['chunks'] for m in all_metrics)/global_count,max_logprob_recompute_error=max(m['max_logprob_recompute_error'] for m in all_metrics),gradient_norm=float(grad),actor_lr=opt.param_groups[0]['lr'],critic_warmup=warmup,advantage_mean=mean,advantage_std=std,seconds=time.time()-start,microbatch=microbatch,chunks=global_count,parallel_gradient_gpus=CFG['cuda_physical_order'],per_rank=all_metrics,peak_gib=torch.cuda.max_memory_allocated(0)/2**30,**diagnostics)
