"""Check real stochastic-policy successes before committing to the PPO pilot."""
import argparse
import dataclasses
import json
from pathlib import Path
from actor_core import Actor
from train_runtime import *
from minibatch_ppo import check_initial_recompute


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'scratch').mkdir()
    seed_all(42);torch.cuda.set_device(0);actor=None
    try:
        actor=Actor(0,0,a.out,8);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        apply_state(actor.model,ck['trainable_state']);actor.version=0;results=[]
        for level in (.5,.25):
            seed_all(431);actor.reset_rng=np.random.default_rng(4400);actor.active=[None]*8
            actor.model.rl_cfg=dataclasses.replace(actor.model.rl_cfg,noise_level=level)
            result=actor.rollout(0,0.,25,True)
            data=torch.load(result['path'],map_location='cpu',weights_only=False)
            groups={}
            for r in data:groups.setdefault(r['sampling_group'],[]).append(r)
            ordered=[r for g in groups.values() for r in sorted(g,key=lambda r:r['sampling_index'])]
            lp_error,value_error=check_initial_recompute(actor.model,ordered)
            eps=result['episodes']
            record=dict(noise_level=level,completed=len(eps),successes=sum(e['success'] for e in eps),
                        mean_reward=float(np.mean([e['episode_reward'] for e in eps])) if eps else None,
                        max_hold=max((int(e['final']['earned'].get('hold',0)*40) for e in eps),default=0),
                        logprob_recompute_error=lp_error,value_recompute_error=value_error,
                        episodes=eps,unfinished_discarded=result['active'])
            results.append(record);save(a.out/f'noise-{level}.json',record)
            save(a.out/'summary.json',dict(protocol='actual stochastic policy sampling; no optimizer or scripted commands',checkpoint=a.checkpoint,results=results))
            print(json.dumps({k:v for k,v in record.items() if k not in ('episodes','unfinished_discarded')}),flush=True)
            Path(result['path']).unlink()
    finally:
        if actor is not None:actor.close()
