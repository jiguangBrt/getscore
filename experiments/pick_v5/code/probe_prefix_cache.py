"""One paired real PPO update: frozen-prefix caching versus original math."""
import argparse
import time
from pathlib import Path
from train_runtime import *
from minibatch_ppo import minibatch_update,check_initial_recompute


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    seed_all(42);torch.cuda.set_device(0);model=build_model('cuda:0')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    initial=ck['trainable_state'];apply_state(model,initial)
    data=torch.load(a.batch,map_location='cpu',weights_only=False)
    for row in data:row['version']=4
    ap=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')]
    vp=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')]
    # Warm ordinary kernels without an optimizer update; no RNG or physics data changes.
    check_initial_recompute(model,sorted(data,key=lambda r:(r['sampling_group'],r['sampling_index']))[:8])
    runs=[];reference=None;reference_opt=None;reference_samples=None
    for enabled in (False,True):
        apply_state(model,initial);seed_all(42421);CFG['cache_frozen_prefix']=enabled
        opt=torch.optim.AdamW([dict(params=ap,lr=CFG['actor_lr']),dict(params=vp,lr=CFG['critic_lr'])],
            betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'],weight_decay=CFG['weight_decay'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.time()
        metrics,samples=minibatch_update(model,opt,data,4,ap,vp)
        torch.cuda.synchronize();seconds=time.time()-start
        state=trainable_state(model);optimizer=cpu_tree(opt.state_dict())
        record=dict(enabled=enabled,seconds=seconds,metrics=metrics)
        if not enabled:reference=state;reference_opt=optimizer;reference_samples=samples
        else:
            delta=max(float((state[k]-reference[k]).abs().max()) for k in state)
            opt_delta=max(float((s[k]-reference_opt['state'][pid][k]).abs().max())
                          for pid,s in optimizer['state'].items() for k in s)
            prediction_delta=max(abs(x['new_value']-y['new_value']) for x,y in zip(samples,reference_samples))
            assert delta<=1e-7 and opt_delta<=1e-7 and prediction_delta<=1e-7,(delta,opt_delta,prediction_delta)
            assert metrics['max_logprob_recompute_error']<=CFG['recompute_max_error']
            assert metrics['actor_gradient_norm']>0 and metrics['accepted_epochs']==2 and metrics['optimizer_steps']==8
            assert model._frozen_prefix_cache is None and all('frozen_prefix_key' not in r['forward_inputs'] for r in data)
            record.update(max_parameter_difference=delta,max_optimizer_difference=opt_delta,max_value_difference=prediction_delta)
        runs.append(record);save(a.out/'result.json',dict(passed=len(runs)==2,runs=runs,
            speedup=runs[0]['seconds']/seconds if enabled else None,role='paired update on recorded on-policy batch; not additional training episodes'))
        print(dict(enabled=enabled,seconds=seconds,actor_gradient_norm=metrics['actor_gradient_norm']),flush=True)
