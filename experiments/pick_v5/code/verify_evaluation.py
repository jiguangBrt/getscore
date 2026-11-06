"""Independent reconciliation of reset validity, policy actions and success traces."""
import argparse
import json
from pathlib import Path
from collections import Counter


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=json.loads((a.run/'evaluation-fresh.json').read_text())
    cases=result['results'];assert len({(r['case']['spec']['type'],r['case']['seed']) for r in cases})==len(cases)
    traces={}
    for f in a.run.glob('eval-environments-fresh-actor0/env*/episode*/native-trace.jsonl'):
        reset=json.loads((f.parent/'reset.json').read_text());key=(reset['spec']['type'],reset['seed'])
        assert key not in traces
        traces[key]=(f,reset)
    rows=[];versions=Counter()
    for case in cases:
        assert case['valid_reset'],'invalid resets cannot be silently dropped'
        spec=case['case']['spec'];seed=case['case']['seed'];f,reset=traces[(spec['type'],seed)]
        initial=reset['metrics']
        assert not initial['grasped'] and initial['supported'] and abs(initial['clearance_m'])<.006
        assert initial['gripper_opening']>.065
        trajectory=[json.loads(x) for x in f.read_text().splitlines()];last=trajectory[-1]
        assert len(trajectory)==case['native_steps'] and last['terminated'] and not last['truncated']
        assert abs(sum(r['reward'] for r in trajectory)-last['episode_reward'])<1e-8
        assert abs(last['episode_reward']-case['final']['episode_reward'])<1e-8
        assert all(r['policy_version']==result['update'] for r in trajectory)
        versions.update(r['policy_version'] for r in trajectory)
        assert all(abs(sum(r['reward_components'].values())-r['reward'])<1e-9 for r in trajectory)
        assert all(0<=r['gripper_collision_cost']<=.010000001 and r['collision_physics_substeps']==25 for r in trajectory)
        assert all(abs(r['reward_components']['gripper_collision']+r['gripper_collision_cost'])<1e-12 for r in trajectory)
        if case['success']:
            tail=trajectory[-40:]
            assert len(tail)==40 and last['hold_steps']==40 and last['episode_reward']>0
            assert all(r['grasped'] and r['clearance_m']>=.05 and r['xy_offset_m']<.1
                       and r['linear_speed']<=.01 and r['angular_speed']<=.1 for r in tail)
            assert max(r['clearance_m'] for r in tail)-min(r['clearance_m'] for r in tail)<=.005
        rows.append(dict(type=spec['type'],seed=seed,success=case['success'],steps=len(trajectory),
                         reward=last['episode_reward'],collision_cost=last['cumulative_collision_cost'],
                         max_hold=max(r['hold_steps'] for r in trajectory),
                         max_lift=max(r['clearance_m'] for r in trajectory),
                         contact=any(r['bilateral_contact'] for r in trajectory),
                         verified_lift=any(r['grasped'] for r in trajectory),
                         lift_5cm=any(r['grasped'] and r['clearance_m']>=.05 for r in trajectory),
                         first_lift_5cm_step=next((r['native_step'] for r in trajectory if r['grasped'] and r['clearance_m']>=.05),None),
                         first_hold_step=next((r['native_step'] for r in trajectory if r['hold_steps']>0),None),
                         hold_breaks=sum(prev['hold_steps']>cur['hold_steps'] for prev,cur in zip(trajectory,trajectory[1:])),
                         finger_contacts_exempt_frames=sum(r['gripper_intended_contact_peak_force_n']>.5 and r['gripper_collision_cost']==0 for r in trajectory),
                         trace=str(f)))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(passed=True,episodes=len(rows),successes=sum(r['success'] for r in rows),
         per_type_counts=dict(Counter(r['type'] for r in rows)),policy_native_steps_by_version=dict(versions),rows=rows),indent=2)+'\n')
