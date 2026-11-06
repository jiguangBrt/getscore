"""Repeated asset switches: deterministic reset pixels/state and prompt cleanup."""
import argparse
import hashlib
import time
import weakref
from pathlib import Path
from pick_env import PickEnv,inventory
from runtime import save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);expected={};results=[]
    for repeat in range(2):
        for spec in inventory():
            start=time.time();e=PickEnv(dict(spec,skill='pick'),1260000,a.out/f"{spec['type']}-{repeat}",0.,True)
            obs=e.observation()
            hashes={k:hashlib.sha256(obs[k].tobytes()).hexdigest() for k in ('main_images','wrist_images','states')}
            hashes['sim']=hashlib.sha256(e.wrapper.get_sim_state().tobytes()).hexdigest()
            if repeat:assert hashes==expected[spec['type']],spec['type']
            else:expected[spec['type']]=hashes
            ref=weakref.ref(e);e.close();del e
            assert ref() is None,'PickEnv retained after close (monitor reference cycle)'
            results.append(dict(type=spec['type'],repeat=repeat,seconds=time.time()-start,hashes=hashes,immediately_released=True))
            save(a.out/'summary.json',dict(passed=len(results)==24,results=results))
