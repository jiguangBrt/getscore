"""Optional initialization data, explicitly separate from policy RL successes."""
import argparse
import json
from pathlib import Path
import numpy as np
from pick_env import PickEnv, inventory
from runtime import CFG, save


def collect(out, count):
    out.mkdir(parents=True,exist_ok=False);results=[];accepted=0
    for seed in range(1200000,1200000+count*4):
        e=None;observations=[];actions=[]
        try:
            e=PickEnv(dict(inventory()[0],skill='pick'),seed,out/f'seed-{seed}',0.,False)
            def step(action):
                if e.done:return
                observations.append(e.observation())
                actions.append(np.asarray(action,dtype=np.float32))
                e.step(action)
            def move(target,grip):
                for _ in range(80):
                    if e.done:break
                    delta=np.asarray(target)-np.asarray(e.obs['robot0_eef_pos'])
                    if np.linalg.norm(delta)<.003:break
                    step(np.r_[np.clip(delta/.05,-.5,.5),0.,0.,0.,grip])
            pos=e.sim.data.body_xpos[e.body].copy();move(pos+[0,0,.052],-1.)
            for _ in range(25):step(np.r_[np.zeros(6),1.])
            move(np.asarray(e.obs['robot0_eef_pos'])+[0,0,.10],1.)
            for _ in range(200):step(np.r_[np.zeros(6),1.])
            final=json.loads(e.trace_file.read_text().splitlines()[-1])
            result=dict(seed=seed,success=final['skill_success'],reward=final['episode_reward'],steps=len(actions),
                        role='scripted demonstration; never count as policy success',prompt=e.prompt)
            if final['skill_success']:
                assert final['hold_steps']>=40 and final['clearance_m']>=.05 and final['grasped']
                path=out/f'seed-{seed}'/'demo.npz'
                np.savez_compressed(path,
                                    main_images=np.stack([o['main_images'][0] for o in observations]),
                                    wrist_images=np.stack([o['wrist_images'][0] for o in observations]),
                                    states=np.concatenate([o['states'] for o in observations]),actions=np.stack(actions))
                accepted+=1;result['data']=str(path)
            results.append(result);save(out/'summary.json',dict(config=CFG,accepted=accepted,attempts=len(results),results=results))
            print(json.dumps(result),flush=True)
            if accepted>=count:break
        finally:
            if e is not None:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=32)
    a=p.parse_args();collect(a.out,a.count)
