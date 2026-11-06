"""Read-only audit of the completed run; stdout contains aggregate evidence."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import statistics as st

ROOT = Path('/data/wangyihan/getscore-rl-skills/runs/pi05-skills-v3-overnight-20261104-to-20261105-0900')
cfg = json.loads((ROOT / 'effective-config.json').read_text())

def mean(xs):
    return st.mean(xs) if xs else None

def longest(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best

def positive_components(final, skill):
    budgets = {
        'pick': {'height': .3, 'hold': .2},
        'place': {'lowering': .2, 'stability': .2},
        'upright': {'approach': .05, 'orientation': .3, 'stability': .2},
    }
    events = {'contact': .1, 'grasp': .05, 'controlled_lift': .05,
              'release': .1, 'retreat': .05}
    out = {k: final.get('earned', {}).get(k, 0) * v for k, v in budgets[skill].items()}
    out.update({k: v for k, v in events.items() if k in final.get('events', [])})
    if final.get('skill_success'):
        out['success'] = cfg['reward']['terminal_bonus']
    return out

training = defaultdict(list)
with (ROOT / 'episodes.jsonl').open() as f:
    for line in f:
        row = json.loads(line)
        training[row['skill']].append(row)
report = {'run': str(ROOT), 'training': {}, 'evaluation_traces': {}}
for skill, rows in training.items():
    sums = Counter()
    events = Counter()
    earned = Counter()
    for row in rows:
        final = row['final']
        parts = positive_components(final, skill)
        parts['horizontal'] = -cfg[skill]['horizontal_penalty_per_m'] * final['xy_path_m']
        parts['step'] = -row['native_steps'] * cfg['reward']['step_penalty']
        parts['failure'] = -int(row['failure']) * cfg['reward']['failure_penalty']
        parts['timeout'] = -int(row['timeout']) * cfg['reward']['timeout_penalty']
        if skill == 'pick':
            parts['descent'] = -cfg['pick']['descent_penalty_per_m'] * max(0, final['cumulative_descent_m'] - cfg['pick']['descent_deadband_m'])
        assert abs(sum(parts.values()) - row['episode_reward']) < 1e-5, (skill, row['episode_id'])
        for key, value in parts.items():
            sums[key] += value
        events.update(final.get('events', []))
        earned.update(k for k, v in final.get('earned', {}).items() if v > 0)
    report['training'][skill] = {
        'n': len(rows), 'success': sum(r['success'] for r in rows),
        'termination_reasons': dict(Counter(r['final']['termination_reason'] for r in rows)),
        'mean_episode_reward': mean([r['episode_reward'] for r in rows]),
        'mean_native_steps': mean([r['native_steps'] for r in rows]),
        'mean_reward_components_reconstructed_and_verified': {k: v / len(rows) for k, v in sums.items()},
        'ever_earned_positive_progress_count': dict(earned), 'ever_event_count': dict(events),
    }

for tag in ['baseline', 'final']:
    cases = defaultdict(list)
    for directory in sorted(ROOT.glob(f'eval-environments-{tag}-actor*')):
        for path in sorted(directory.glob('env*/episode*/native-trace.jsonl')):
            trace = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not trace:
                continue
            skill = trace[0]['skill']
            reset = json.loads(path.with_name('reset.json').read_text())
            local_radius = cfg[skill].get('target_radius_m', .05)
            local = [t['target_xy_distance'] <= local_radius for t in trace]
            upright = [t['tilt_degrees'] <= cfg['success']['upright_degrees'] for t in trace]
            supported = [t['target_supported'] for t in trace]
            quiet = [t['linear_speed'] <= cfg['success']['linear_speed_m_s'] and t['angular_speed'] <= cfg['success']['angular_speed_rad_s'] for t in trace]
            released = [not t['grasped'] and t['gripper_opening'] > .06 for t in trace]
            retreat = [t['eef_object_center_distance'] >= cfg['success']['eef_clearance_m'] for t in trace]
            contact = [l and u and s and q for l, u, s, q in zip(local, upright, supported, quiet)]
            stable = [c and r and e for c, r, e in zip(contact, released, retreat)]
            sums = Counter()
            for t in trace:
                sums.update(t['reward_components'])
            assert abs(sum(sums.values()) - trace[-1]['episode_reward']) < 1e-5
            row = {
                'seed': reset['seed'], 'object': reset['spec']['label'],
                'scene': reset['spec']['task_index'], 'path': str(path),
                'steps': len(trace), 'reward_components': dict(sums),
                'ever_grasped': any(t['grasped'] for t in trace),
                'ever_lifted_5cm_while_grasped': any(t['grasped'] and t['lift_m'] >= .05 for t in trace),
                'max_hold_steps': max(t['hold_steps'] for t in trace),
                'max_stable_steps': max(t['stable_count'] for t in trace),
                'ever_lowering_reward': trace[-1]['earned'].get('lowering', 0) > 0,
                'ever_local_supported': any(l and s for l, s in zip(local, supported)),
                'ever_local_upright_supported': any(l and u and s for l, u, s in zip(local, upright, supported)),
                'ever_contact_eligible': any(contact),
                'ever_release_eligible': any(c and r for c, r in zip(contact, released)),
                'ever_complete_stable_predicate': any(stable),
                'longest_local_upright_supported_steps': longest([l and u and s for l, u, s in zip(local, upright, supported)]),
                'max_upright_supported_released_steps_anywhere': longest([u and s and r for u, s, r in zip(upright, [t['supported'] for t in trace], released)]),
                'ever_orientation_reward': trace[-1]['earned'].get('orientation', 0) > 0,
                'min_tilt_degrees': min(t['tilt_degrees'] for t in trace),
                'final_reason': trace[-1]['termination_reason'],
                'events': trace[-1]['events'],
            }
            cases[skill].append(row)
    out = {}
    for skill, rows in cases.items():
        parts = Counter()
        for row in rows:
            parts.update(row['reward_components'])
        out[skill] = {
            'n': len(rows),
            'ever_counts': {k: sum(r[k] for r in rows) for k in rows[0] if k.startswith('ever_')},
            'max_hold_steps_across_cases': max(r['max_hold_steps'] for r in rows),
            'max_stable_steps_across_cases': max(r['max_stable_steps'] for r in rows),
            'mean_reward_components': {k: v / len(rows) for k, v in parts.items()},
            'cases': rows,
        }
    report['evaluation_traces'][tag] = out
print(json.dumps(report, ensure_ascii=False, indent=2))
