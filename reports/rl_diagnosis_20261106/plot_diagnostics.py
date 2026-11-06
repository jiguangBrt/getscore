from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (ROOT.parent / 'rl_skills_v3_20261104/overnight/metrics.jsonl').read_text().splitlines()]
audit = json.loads((ROOT / 'audit-data.json').read_text())
x = np.array([r['update'] for r in rows])

def ema(values, alpha=.06):
    out = []
    for v in values:
        out.append(v if not out else alpha * v + (1-alpha) * out[-1])
    return out

colors = {'pick': '#2671a8', 'place': '#299d85', 'upright': '#ed9224'}
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
for skill, color in colors.items():
    y = [r['metrics'][f'train/{skill}_episode_reward'] for r in rows]
    axes[0,0].plot(x, ema(y), color=color, label=skill)
axes[0,0].set(title='Complete-episode reward (EMA of update means)', ylabel='Undiscounted episode return', xlabel='Optimizer update')
axes[0,0].legend()

categories = ['Positive progress', 'XY travel', 'Failure', 'Timeout', 'Other costs']
for i, (skill, color) in enumerate(colors.items()):
    d = audit['training'][skill]['mean_reward_components_reconstructed_and_verified']
    vals = [sum(v for v in d.values() if v > 0), d['horizontal'], d['failure'], d['timeout'], d['step'] + d.get('descent', 0)]
    axes[0,1].bar(np.arange(len(categories)) + (i-1)*.25, vals, .25, color=color, label=skill)
axes[0,1].set(title='Reward components: every completed training episode', ylabel='Mean contribution per episode')
axes[0,1].set_xticks(np.arange(len(categories)), categories, rotation=15)
axes[0,1].axhline(0, color='#333', linewidth=.8)
axes[0,1].legend()

kl = np.array([r['learner']['approx_kl'] for r in rows])
axes[1,0].plot(x, kl, color='#8052b0', alpha=.35, linewidth=.8)
axes[1,0].plot(x, ema(kl), color='#64338e', label='EMA')
axes[1,0].set_yscale('log')
axes[1,0].set(title='Post-update KL: only 32/768 chunks checked', ylabel='Approximate KL (log scale)', xlabel='Optimizer update')
axes[1,0].text(.03, .08, 'All logged clip fractions = 0', transform=axes[1,0].transAxes, fontsize=10)

for skill, color in colors.items():
    y = [r['metrics'][f'train/{skill}_success'] for r in rows]
    axes[1,1].plot(x, y, color=color, label=skill, linewidth=2)
axes[1,1].set(title='Training success: all 22,407 completed episodes', ylabel='Success rate', xlabel='Optimizer update', ylim=(-.02, 1.02))
axes[1,1].text(.5, .55, 'pick: 0 / 9,774\nplace: 0 / 7,848\nupright: 0 / 4,785', ha='center', transform=axes[1,1].transAxes, fontsize=13)
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.16, axis='y')
fig.suptitle('Completed pi0.5 v3 run | reward signal and policy update audit', fontsize=15)
fig.savefig(ROOT / 'diagnostic-curves.png', dpi=165)
plt.close(fig)
