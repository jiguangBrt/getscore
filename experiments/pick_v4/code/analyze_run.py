"""Compact stage, reward, and empirical critic diagnostics from native traces."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def avg(xs):return statistics.fmean(xs) if xs else None


def describe_traces(paths,allowed_episodes=None):
    episodes=[]
    for path in sorted(paths):
        episode_id=int(path.parent.name.removeprefix('episode'))
        if allowed_episodes is not None and episode_id not in allowed_episodes:continue
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:continue
        end=rows[-1]
        if not end['terminated']:continue
        parts=collections.Counter()
        for r in rows:parts.update(r['reward_components'])
        assert math.isclose(sum(parts.values()),end['episode_reward'],abs_tol=1e-6)
        reset=json.loads((path.parent/'reset.json').read_text())
        episodes.append(dict(seed=reset['seed'],episode_id=int(path.parent.name.removeprefix('episode')),
                             update=end.get('policy_version',0)+1,success=end['skill_success'],reason=end['termination_reason'],
                             steps=len(rows),reward=end['episode_reward'],parts=dict(parts),
                             ever_contact=any(r['bilateral_contact_steps']>=3 for r in rows),
                             ever_verified=any(r['grasped'] for r in rows),
                             ever_lift_5cm=any(r['grasped'] and r['clearance_m']>=.05 for r in rows),
                             max_hold=max(r['hold_steps'] for r in rows),peak_clearance=max(r['clearance_m'] for r in rows),
                             soft_excursion=any(r['xy_offset_m']>.03 for r in rows),
                             recovered_after_soft=any(rows[i]['xy_offset_m']<=.03 and any(r['xy_offset_m']>.03 for r in rows[:i]) for i in range(len(rows))),
                             pad_miss_verified_steps=sum(r['grasped'] and not r['pad_only_grasp'] for r in rows),
                             path=str(path)))
    episodes.sort(key=lambda e:(e['update'],e['episode_id']))
    n=len(episodes)
    reward_parts={k:avg([e['parts'].get(k,0.) for e in episodes]) for k in set(k for e in episodes for k in e['parts'])}
    return dict(episodes=n,successes=sum(e['success'] for e in episodes),success_rate=avg([int(e['success']) for e in episodes]),
                reasons=dict(collections.Counter(e['reason'] for e in episodes)),
                mean_reward=avg([e['reward'] for e in episodes]),mean_components=reward_parts,
                ever_contact=sum(e['ever_contact'] for e in episodes),ever_verified=sum(e['ever_verified'] for e in episodes),
                ever_lift_5cm=sum(e['ever_lift_5cm'] for e in episodes),any_hold=sum(e['max_hold']>0 for e in episodes),
                hold_at_least_10=sum(e['max_hold']>=10 for e in episodes),
                max_hold=max((e['max_hold'] for e in episodes),default=0),
                recovered_after_soft=sum(e['recovered_after_soft'] for e in episodes),
                pad_miss_verified_steps=sum(e['pad_miss_verified_steps'] for e in episodes),records=episodes)


def critic_summary(samples):
    returns=[s['empirical_return'] for s in samples];values=[s['old_value'] for s in samples]
    err=[r-v for r,v in zip(returns,values)]
    variance=statistics.pvariance(returns) if returns else 0.
    return dict(samples=len(samples),mse=avg([e*e for e in err]),bias=avg(err),mean_return=avg(returns),mean_value=avg(values),
                explained_variance=1-statistics.pvariance(err)/variance if variance>1e-12 else None)


def analyze(run):
    result={'run':str(run),'evaluations':{}}
    for ev in sorted(run.glob('evaluation-*.json')):
        if ev.name=='evaluation-cases.json':continue
        metadata=json.loads(ev.read_text())
        if not isinstance(metadata,dict) or 'results' not in metadata:continue
        tag=ev.stem.removeprefix('evaluation-')
        result['evaluations'][tag]=describe_traces(run.glob(f'eval-environments-{tag}-actor*/env*/episode*/native-trace.jsonl'))
    epfile=run/'episodes.jsonl'
    learned_rows=[json.loads(x) for x in epfile.read_text().splitlines()] if epfile.exists() else []
    learned_episodes={r['episode_id'] for r in learned_rows}
    result['training']=describe_traces(run.glob('environments-actor*/env*/episode*/native-trace.jsonl'),learned_episodes)
    order={r['episode_id']:i for i,r in enumerate(learned_rows)}
    result['training']['records'].sort(key=lambda r:order[r['episode_id']])
    p=run/'metrics.jsonl'
    result['updates']=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    p=run/'completed-value-trajectories.jsonl'
    trajectories=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    samples=[s for t in trajectories for s in t['samples']]
    result['empirical_critic']=critic_summary(samples)
    result['empirical_critic_by_phase']={phase:critic_summary([s for s in samples if s['phase']==phase]) for phase in ['approach','contact','lift','hold']}
    result['empirical_critic_late']=critic_summary(samples[-max(1,len(samples)//5):])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();result=analyze(a.run)
    if a.out:a.out.write_text(json.dumps(result,indent=2))
    compact={k:v for k,v in result.items() if k not in ('updates','evaluations','training')}
    compact['evaluations']={k:{x:y for x,y in v.items() if x!='records'} for k,v in result['evaluations'].items()}
    compact['training']={k:v for k,v in result['training'].items() if k!='records'}
    print(json.dumps(compact,indent=2))
