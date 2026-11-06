"""Reproduce the final aggregate and per-object task-stage figure."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
data = json.loads((root / 'learning-data.json').read_text())
main = [r for n in ('audit-h.json', 'audit-k.json')
        for r in json.loads((root / n).read_text())['rows']]
proof = json.loads((root / 'final-evaluation-proof.json').read_text())
fresh = proof['rows']
types = data['pick-v5-pilot800-20261106-h']['effective-config']['object_types']

def aggregate(rows):
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                success_rate=float(np.mean([r['success'] for r in rows])),
                positive_returns=sum(r['reward'] > 0 for r in rows),
                mean_return=float(np.mean([r['reward'] for r in rows])),
                mean_collision_cost=float(np.mean([r['collision_cost'] for r in rows])))

def failure_stage(r):
    if r['success']: return 'success'
    if not r['contact']: return 'no_bilateral_contact'
    if not r['verified_lift']: return 'contact_without_verified_lift'
    if not r['lift_5cm']: return 'verified_lift_below_5cm'
    if not r['max_hold']: return 'lift_5cm_without_stable_hold'
    return 'stable_hold_interrupted_or_too_late'

summary = dict(
    total_complete_rl_episodes=sum(len(r.get('episodes', [])) for r in data.values()),
    main_training=aggregate(main), final_evaluation=aggregate(fresh),
    final_by_object={t: aggregate([r for r in fresh if r['type'] == t]) for t in types},
    final_stage_counts=dict(Counter(failure_stage(r) for r in fresh)),
    selected_checkpoint='H update9; selected on fixed development coverage before fresh evaluation',
    ready_for_larger_evaluation=True, long_run_learning_improvement_demonstrated=False)
(root / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')

fields = ['contact', 'verified_lift', 'lift_5cm', 'hold_started', 'success']
matrix = np.array([[sum((r['max_hold'] > 0 if f == 'hold_started' else r[f])
                       for r in fresh if r['type'] == t) for f in fields] for t in types])
short = ['white mug', 'soup can', 'ketchup', 'tomato can', 'butter', 'cheese',
         'black bowl', 'white/yellow mug', 'book', 'chocolate', 'moka pot', 'yellow moka']
fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
im = ax.imshow(matrix / 16, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
ax.set_yticks(range(12), short)
ax.set_xticks(range(5), ['bilateral\ncontact', 'verified\nairborne grasp',
                        'lift >=5cm', 'stable hold\nstarted', '2s hold\nsuccess'])
for i in range(12):
    for j in range(5):
        ax.text(j, i, f'{matrix[i,j]}/16', ha='center', va='center',
                color='white' if matrix[i,j] > 10 else '#183040')
ax.set_title('Fresh held-out evaluation: task stages by object\nH PPO update9; 192 full episodes; no checkpoint selection on these seeds')
fig.colorbar(im, ax=ax, shrink=.8, label='fraction of all 16 cases')
fig.savefig(root / 'task-stages.png', dpi=170)
plt.close(fig)
print(json.dumps(summary, indent=2))
