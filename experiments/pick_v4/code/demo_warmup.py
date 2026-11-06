"""Optional physically verified demo warmup, followed by independent policy eval.

This is supervised flow matching, not PPO. Its use must be reported explicitly;
policy evaluation always starts from an open gripper and mug resting on the table.
"""
import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from actor_core import Actor,evaluation_cases,event
from train_runtime import *
from pilot_train import eval_metrics
from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation


def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'scratch').mkdir();(out/'checkpoints').mkdir()
    save(out/'effective-config.json',CFG);save(out/'arguments.json',vars(args))
    (out/'source').mkdir();manifest={}
    for source in Path(__file__).parent.iterdir():
        if source.suffix in ('.py','.bddl'):
            shutil.copy2(source,out/'source'/source.name);manifest[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(out/'source-manifest.json',manifest)
    summary=json.loads((Path(args.demos)/'summary.json').read_text())
    valid=[r for r in summary['results'] if r['success']]
    assert len(valid)>=16 and all(r['reward']>0 for r in valid)
    data=[dict(np.load(r['data'])) for r in valid]
    windows=[(i,t) for i,d in enumerate(data) for t in range(0,len(d['actions'])-CFG['action_horizon']+1,2)]
    save(out/'data-provenance.json',dict(demonstrations=len(data),windows=len(windows),summary=str(Path(args.demos)/'summary.json'),
                                      method='supervised flow matching warmup; not RL',heldout_seeds='90000000+; never used as demonstrations',
                                      development_seeds='93000000+; checkpoint selection only',
                                      file_sha256={r['data']:hashlib.sha256(Path(r['data']).read_bytes()).hexdigest() for r in valid}))
    seed_all(CFG['seed']);torch.cuda.set_device(0);actor=None
    try:
        actor=Actor(0,0,out,8);actor.version=0;model=actor.model
        params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')]
        opt=torch.optim.AdamW(params,lr=args.lr,betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'],weight_decay=CFG['weight_decay'])
        cases=evaluation_cases(actor.reg)
        for case in cases:
            case['seed']+=3000000
            case['id']='dev-'+case['id']
        save(out/'evaluation-cases.json',cases)
        def checkpoint(name,step):
            atomic_torch(out/'checkpoints'/name,dict(trainable_state=trainable_state(model),optimizer=cpu_tree(opt.state_dict()),
                         update=step,config=CFG,base_sha256=CFG['base_sha256'],base_checkpoint=CFG['base_checkpoint'],
                         source_manifest=manifest,
                         initialization='physically validated scripted-demo supervised warmup; not pure RL',data_provenance=str(out/'data-provenance.json')))
        def evaluate(tag,step):
            actor.version=step;r=actor.evaluate(cases,tag);m=eval_metrics(r)
            save(out/f'evaluation-{tag}.json',dict(r,metrics=m,update=step));event(out,'evaluation_end',tag=tag,update=step,**m)
            return m
        checkpoint('fresh-start.pt',0);evaluate('baseline',0)
        rng=np.random.default_rng(CFG['seed']);start=time.time();best=-1.
        for step in range(1,args.steps+1):
            chosen=[windows[i] for i in rng.integers(len(windows),size=8)]
            env_obs=dict(main_images=np.stack([data[i]['main_images'][t] for i,t in chosen]),
                         wrist_images=np.stack([data[i]['wrist_images'][t] for i,t in chosen]),
                         states=np.stack([data[i]['states'][t] for i,t in chosen]),task_descriptions=[CFG['prompt_templates']['pick'].format(object='white mug')]*8)
            raw_actions=np.stack([data[i]['actions'][t:t+CFG['action_horizon']] for i,t in chosen])
            repacked=repack_env_obs('pi05_libero',env_obs,select_state=model._select_configured_state)
            repacked['actions']=raw_actions
            transformed=model.input_transform(repacked,transpose=False)
            actions=torch.as_tensor(transformed.pop('actions'),device='cuda:0',dtype=torch.float32)
            obs=preprocess_observation(model._observation_dict_to_device(transformed),train=False)
            with torch.no_grad():_,mask,cache=model.build_prefix_cache(obs)
            noise=torch.randn_like(actions)
            times=torch.distributions.Beta(1.5,1.).sample((8,)).to('cuda:0')*.999+.001
            xt=times[:,None,None]*noise+(1-times[:,None,None])*actions
            suffix=model.run_suffix(obs,xt,times,cache,mask)
            velocity=model.velocity_from_suffix(suffix).float()
            loss=((velocity-(noise-actions))[:,:CFG['execute_horizon'],:7]**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite demo loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite demo gradient')
            opt.step()
            if step%10==0:
                row=dict(step=step,flow_matching_loss=float(loss.detach()),gradient_norm=float(norm),elapsed_seconds=time.time()-start)
                with open(out/'metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(out/'status.json',dict(state='warmup',**row));print(json.dumps(row),flush=True)
            if step%args.evaluate_every==0 or step==args.steps:
                checkpoint('latest.pt',step);m=evaluate(f'step-{step:04d}',step)
                if m['success_rate']>best:best=m['success_rate'];checkpoint('best.pt',step)
                if m['successes']>=max(2,len(cases)//4):
                    checkpoint('final.pt',step)
                    atomic_json(out/'status.json',dict(state='completed',method='demo_warmup',step=step,final=m));break
            del cache,suffix,velocity,loss
            if step%50==0:gc.collect()
        else:
            checkpoint('final.pt',args.steps);atomic_json(out/'status.json',dict(state='completed',method='demo_warmup',step=args.steps,final=m))
    finally:
        if actor is not None:actor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--demos',required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--evaluate-every',type=int,default=100)
    train(p.parse_args())
