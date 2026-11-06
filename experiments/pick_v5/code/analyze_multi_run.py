"""Audit learned full episodes, stage coverage and collision costs by asset."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np


def mean(values):return float(np.mean(values)) if values else None


def audit(run,out):
    out.mkdir(parents=True,exist_ok=True)
    episodes=[json.loads(x) for x in (run/'episodes.jsonl').read_text().splitlines()]
    ids={e['episode_id'] for e in episodes}
    assert len(ids)==len(episodes),'duplicate complete episode IDs'
    traces={}
    for f in run.glob('environments-actor*/env*/episode*/native-trace.jsonl'):
        eid=int(f.parent.name.removeprefix('episode'))
        if eid in ids:
            assert eid not in traces,'duplicate training trace'
            traces[eid]=f
    rows=[]
    for e in episodes:
        f=traces[e['episode_id']]
        trajectory=[json.loads(x) for x in f.read_text().splitlines()]
        assert len(trajectory)==e['native_steps']
        assert trajectory[-1]['terminated']
        component_error=max(abs(sum(x['reward_components'].values())-x['reward']) for x in trajectory)
        return_error=abs(sum(x['reward'] for x in trajectory)-e['episode_reward'])
        assert component_error<1e-9 and return_error<1e-8
        if e['success']:
            last=trajectory[-40:]
            assert len(last)==40 and all(x['grasped'] and x['clearance_m']>=.05 and x['xy_offset_m']<.1 and x['linear_speed']<=.01 and x['angular_speed']<=.1 for x in last)
            assert max(x['clearance_m'] for x in last)-min(x['clearance_m'] for x in last)<=.005
            assert e['episode_reward']>0
        parts=defaultdict(float)
        for x in trajectory:
            for k,v in x['reward_components'].items():parts[k]+=v
        rows.append(dict(type=e['object_type'],episode_id=e['episode_id'],update=e['update'],seed=e['seed'],
                         success=e['success'],reward=e['episode_reward'],native_steps=e['native_steps'],
                         contact=any(x['bilateral_contact'] for x in trajectory),
                         lift=any(x['grasped'] for x in trajectory),max_hold=max(x['hold_steps'] for x in trajectory),
                         lift_5cm=any(x['grasped'] and x['clearance_m']>=.05 for x in trajectory),
                         max_height=max(x['clearance_m'] for x in trajectory),
                         collision_cost=-parts['gripper_collision'],
                         collision_seconds=trajectory[-1]['cumulative_collision_seconds'],
                         collision_peak_force_n=max(x['gripper_collision_peak_force_n'] for x in trajectory),
                         reward_components=dict(parts),reason=trajectory[-1]['termination_reason'],trace=str(f)))
    def aggregate(group):
        return dict(episodes=len(group),successes=sum(r['success'] for r in group),
                    success_rate=mean([r['success'] for r in group]),mean_return=mean([r['reward'] for r in group]),
                    positive_return=sum(r['reward']>0 for r in group),
                    contact_rate=mean([r['contact'] for r in group]),lift_rate=mean([r['lift'] for r in group]),
                    lift_5cm_rate=mean([r['lift_5cm'] for r in group]),
                    any_hold_rate=mean([r['max_hold']>0 for r in group]),
                    mean_max_hold=mean([r['max_hold'] for r in group]),
                    mean_collision_cost=mean([r['collision_cost'] for r in group]),
                    mean_collision_seconds=mean([r['collision_seconds'] for r in group]),
                    reasons=dict(Counter(r['reason'] for r in group)))
    by={t:aggregate([r for r in rows if r['type']==t]) for t in sorted({r['type'] for r in rows})}
    metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    summary=dict(run=str(run),overall=aggregate(rows),by_object=by,
                 verified_positive_successes=all(r['reward']>0 for r in rows if r['success']),
                 max_logprob_recompute_error=max(r['learner']['max_logprob_recompute_error'] for r in metrics),
                 max_kl=max(r['learner']['approx_kl'] for r in metrics),
                 rejected_epochs=sum(r['learner']['rejected_epoch'] is not None for r in metrics),
                 mean_timings={k:mean([r[k] for r in metrics]) for k in ('iteration_seconds','rollout_seconds','inference_seconds','environment_seconds','reset_seconds')},
                 rows=rows)
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.run,a.out)
