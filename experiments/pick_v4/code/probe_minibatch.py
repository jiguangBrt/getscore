"""Offline optimizer checks on an actually collected, original-policy batch."""
import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
from train_runtime import *
from minibatch_ppo import minibatch_update


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    run=Path(a.run);seed_all(CFG['seed']);torch.cuda.set_device(0)
    model=build_model('cuda:0');fresh=torch.load(run/'checkpoints/fresh-start.pt',map_location='cpu',weights_only=False)
    apply_state(model,fresh['trainable_state']);data=torch.load(run/'verification-batch.pt',map_location='cpu',weights_only=False)
    ap=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')]
    vp=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')]
    opt=torch.optim.AdamW([dict(params=ap,lr=CFG['actor_lr']),dict(params=vp,lr=CFG['critic_lr'])],
                         betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'],weight_decay=CFG['weight_decay'])
    before=trainable_state(model);start=time.time()
    metrics,samples=minibatch_update(model,opt,data,0,ap,vp)
    actor_sq=critic_sq=0.
    for n,v in trainable_state(model).items():
        sq=float(((v-before[n])**2).sum())
        if n.startswith('value_head.'):critic_sq+=sq
        else:actor_sq+=sq
    result=dict(protocol='offline gradient/optimizer check; not a policy success evaluation',
                config=CFG,batch_source=str(run/'verification-batch.pt'),metrics=metrics,
                actor_delta_l2=actor_sq**.5,critic_delta_l2=critic_sq**.5,seconds=time.time()-start)
    save(out/'result.json',result);save(out/'samples.json',samples)
    print(json.dumps(result,indent=2),flush=True)
