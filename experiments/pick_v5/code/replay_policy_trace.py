"""Render saved policy commands and prove physical-state/reward replay matches.

This is a visualization of an existing policy evaluation, not a new evaluation
or scripted demonstration. It never contributes to reported policy success counts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from pick_env import PickEnv,overlay
from runtime import save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    reset=json.loads((a.source/'reset.json').read_text());rows=[json.loads(x) for x in (a.source/'native-trace.jsonl').read_text().splitlines()]
    e=PickEnv(reset['spec'],reset['seed'],a.out/'replay',0.,True);writer=None
    try:
        current=json.loads((a.out/'replay/reset.json').read_text())
        assert current['initial_sim_sha256']==reset['initial_sim_sha256'],'different physical initial state'
        writer=imageio.get_writer(a.out/'recorded-policy-success.mp4',fps=20,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-threads','2'])
        max_position=max_reward=0.
        for old in rows:
            e.policy_version=old['policy_version']
            obs,r,term,trunc,info=e.step(old['action_command'])
            max_position=max(max_position,float(np.abs(np.array(info['object_pos'])-old['object_pos']).max()))
            max_reward=max(max_reward,abs(r-old['reward']))
            frame=overlay(np.concatenate([obs['main_images'][0],obs['wrist_images'][0]],axis=1),info)
            writer.append_data(frame)
        assert max_position<1e-8 and max_reward<1e-8,(max_position,max_reward)
        assert info['skill_success']==rows[-1]['skill_success']
        save(a.out/'verification.json',dict(source=str(a.source),role='rendered replay of recorded policy commands; not a new evaluation',
                                          source_policy_version=rows[0]['policy_version'],seed=reset['seed'],native_steps=len(rows),
                                          max_position_error_m=max_position,max_reward_error=max_reward,
                                          success=info['skill_success'],reward=info['episode_reward']))
    finally:
        if writer is not None:writer.close()
        e.close()
