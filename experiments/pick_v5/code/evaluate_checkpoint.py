"""Fresh full-task policy evaluation and actor input-contract verification."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from actor_core import Actor,evaluation_cases
from pilot_train import eval_metrics
from train_runtime import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-base',type=int,default=94000000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'scratch').mkdir();(a.out/'source').mkdir()
    for f in Path(__file__).parent.iterdir():
        if f.suffix in ('.py','.bddl'):shutil.copy2(f,a.out/'source'/f.name)
    save(a.out/'effective-config.json',CFG)
    seed_all(CFG['seed']);torch.cuda.set_device(0);actor=None
    try:
        actor=Actor(0,0,a.out,8);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        assert ck['config']['task_version'] in CFG['transfer_allowed_task_versions'] and ck['base_sha256']==CFG['base_sha256']
        apply_state(actor.model,ck['trainable_state']);actor.version=ck['update']
        save(a.out/'checkpoint-source.json',dict(path=a.checkpoint,sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),update=ck['update'],training_config=ck['config']))
        cases=evaluation_cases(actor.reg)
        for i,case in enumerate(cases):case['seed']=a.seed_base+i;case['id']=f'fresh-pick-{i:02d}'
        save(a.out/'evaluation-cases.json',cases)
        # Critic-only features must have zero effect on evaluation actor outputs.
        rr=actor.reset_many([c['spec'] for c in cases[:8]],[c['seed'] for c in cases[:8]],1.,True)
        assert all(r['ok'] for r in rr)
        obs=stack_obs([r['obs'] for r in rr]);clean={k:v for k,v in obs.items() if k!='critic_context'}
        gen=torch.Generator(device=0).manual_seed(98123)
        noise=torch.randn(8,CFG['action_horizon'],32,device='cuda:0',generator=gen)
        with torch.no_grad():
            first,_=actor.model.predict_action_batch(obs,mode='eval',noise=noise.clone())
            second,_=actor.model.predict_action_batch(clean,mode='eval',noise=noise.clone())
        difference=float((first-second).abs().max())
        assert difference==0.,'privileged critic context affected actor'
        save(a.out/'actor-input-contract.json',dict(max_action_difference_with_without_critic_context=difference,rows=8,noise_seed=98123))
        result=actor.evaluate(cases,'fresh');metrics=eval_metrics(result)
        save(a.out/'evaluation-fresh.json',dict(result,metrics=metrics,update=actor.version))
        save(a.out/'status.json',dict(state='completed',metrics=metrics,checkpoint=a.checkpoint,update=actor.version,seed_base=a.seed_base))
        print(json.dumps(metrics,indent=2),flush=True)
    finally:
        if actor is not None:actor.close()
