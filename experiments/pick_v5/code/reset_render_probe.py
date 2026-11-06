"""Prove skipped setup-camera renders preserve physics, policy pixels and reward."""
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
from pick_env import PickEnv,inventory
from runtime import CFG,save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);results=[]
    for typ in ('porcelain_mug','butter','moka_pot'):
        spec=next(x for x in inventory() if x['type']==typ)
        variants=[]
        for skip in (False,True):
            CFG['skip_reset_camera_render']=skip
            e=None
            try:
                start=time.time();e=PickEnv(dict(spec,skill='pick'),1230000,a.out/f'{typ}-skip{skip}',0.,True)
                seconds=time.time()-start;obs=e.observation()
                pixels={k:hashlib.sha256(obs[k].tobytes()).hexdigest() for k in ('main_images','wrist_images','states')}
                simhash=hashlib.sha256(e.wrapper.get_sim_state().tobytes()).hexdigest()
                rewards=[e.step(np.r_[np.zeros(6),-1.])[1] for _ in range(10)]
                variants.append(dict(skip=skip,reset_seconds=seconds,obs=pixels,simhash=simhash,rewards=rewards))
            finally:
                if e is not None:e.close()
        match=all(variants[0][k]==variants[1][k] for k in ('obs','simhash','rewards'))
        results.append(dict(type=typ,exact_match=match,variants=variants))
        save(a.out/'summary.json',dict(results=results,passed=all(r['exact_match'] for r in results)))
        assert match,typ
