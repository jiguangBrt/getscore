# Single white mug pick pilot (2026-11-06)

This experiment is isolated from previous multi-skill runs. It uses the unchanged
`pi05_libero_torch/model.safetensors` base and fresh action-expert LoRA adapters.
Only `pick` is trained. The authored LIBERO living-room table, robot, white mug,
controller, masses, friction, and 20 Hz physics remain; other movable objects are
removed in a separate BDDL file. Position and yaw vary across episodes.

`code/pick_env.py` verifies target-specific left/right finger contacts including
finger bodies and pads, force on both sides, at least three consecutive contacts,
cup clearance, and relative-pose stability. Ground contact alone is not a verified
airborne grasp. Success additionally requires 5 cm cup-bottom clearance, XY offset
under 10 cm, linear speed <= 1 cm/s, angular speed <= 0.1 rad/s, and 40 continuous
native steps whose height range is <= 5 mm.

`code/pick_reward.py` implements bounded, once-only progress credit: approach 0.1,
bilateral contact 0.1, verified lift 0.5 (saturates at 10 cm), hold progress 0.4,
settling progress 0.2, success 2.0. Per native step cost is 0.0005. Timeout costs 0.3; a severe failure
costs 2.0. After earning lift credit, cumulative cup-bottom descent beyond a
single 1 mm allowance costs 2.0 per meter. Arm descent before grasp is not penalized.

XY offset <= 3 cm has no boundary cost. From 3 cm to 10 cm the occupancy cost
increases quadratically to 0.1 reward/second, then saturates. Returning stops this
cost. XY offset >= 20 cm or falling 10 cm below the support hard-terminates the
episode. The 200-step deadline is part of the task and terminates without value
bootstrap. Sampling-batch boundaries preserve the simulator and bootstrap the
unfinished segment.

The critic receives frozen VLM features plus 19 explicit task context features
(remaining time, hold/progress credit, clearance, offset, contact and motion).
The actor retains the original image/robot-state input format and receives the
short instruction `pick up the white mug and hold it still`. No scripted actions
or demonstrations enter the PPO batch.

The pilots use one explicitly selected GPU UUID each, eight parallel environments, action horizon 10,
execution chunk 8, Flow-SDE noise 0.5, rank-16/alpha-16 LoRA, actor LR 5e-6,
critic LR 1e-4, three critic-only updates followed by ten actor warmup updates,
and separate actor/critic norm clipping
at 1. Full-batch post-update KL, exact sampling/recompute checks, stage metrics,
GAE targets, and empirical complete-episode returns are saved.

Run `physical_preflight.py` first (CPU), then `pilot_train.py --smoke`. A normal
pilot requires those gates and runs approximately 1000 complete episodes with
periodic held-out evaluations. `pilot_train.py` is the one-full-batch-step control;
`minibatch_ppo.py` is the main candidate with minibatch64 and two epochs, up to
eight Adam steps per256-row rollout. Original eight-row sampling groups remain
intact. `--max-updates` counts rollout/update cycles, not individual Adam steps.
`--initial` explicitly starts a new experiment
from a same-v4 weight checkpoint and resets its optimizer; it is not exact resume.
`--resume` restores weights, optimizer, RNG, reset RNG, policy version, and counters;
its episode/update arguments specify additional work. It checks that the learning
contract and scene match. Unfinished simulator episodes are explicitly discarded
and newly reset. Checkpoints save trainable weights, optimizer, RNG, reset RNG, and active-episode
metadata, but not simulator state. Never claim exact environment continuation
after restarting a process. SIGINT/SIGTERM requests a clean stop after the current
update, with checkpoint and final evaluation.

`collect_verified_demos.py` and `demo_warmup.py` are an explicitly separate
supervised initialization candidate, not pure RL. Only physically successful full
trajectories enter warmup. Demonstration seeds start at1200000, development seeds
at93000000, fixed policy tests at90000000, and fresh tests at94000000. No scripted
actions or intervention enter the subsequent PPO rollouts or policy tests.

The deployment actor does not need privileged critic context. In evaluation mode
the context is omitted from actor inputs; a fixed-noise action equality test is
performed by `evaluate_checkpoint.py`. Training still requires true critic context.

Long-run readiness requires physically valid reward ordering and boundary
recovery, reproducible held-out initial states, finite and matching on-policy
probabilities/values, actual actor updates within the KL guard, and evidence that
the policy reaches the desired grasp/hold stages without collapsing. A low critic
loss or positive shaped return alone is insufficient.
