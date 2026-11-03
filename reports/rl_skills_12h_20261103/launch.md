# 12h run launched and checked

Run: /data/wangyihan/getscore-rl-skills/runs/pi05-skills-12h-20261103

Beijing start 2026-11-03 18:37:15; deadline 2026-11-04 06:37:15 plus final save/eval. Detached supervisor PID2615898; trainer PID2615902. Source loaded at b76f608, per-file hashes stored in run source-manifest. Subsequent plot/launch documentation commits do not change the loaded training loop.

Launch verification:6 completed updates,24499 native actions. Latest inspected checkpoint was update5 (116496523 bytes),216 LoRA tensors and8 critic tensors changed from initial. Logprob recompute maximum0; inspected update6 post-update KL1.40e-8. All actual CUDA/EGL render processes were checked on physical0,1,3.

Baseline held-out evaluation: pick14/24 valid,place1/16 valid,upright0/23 valid. Out of24 attempts/skill, place has8 physical-grasp initialization failures and upright1 non-tilted reset. Invalid cases are explicit and never reported as learned policy success. Among place failures are plate,butter,pan,bowl,pudding,cream cheese,basket and one moka case. Successful initialization coverage remains a limitation for broad-object placement.

A CPU-only plot watcher generates learning-curves.png. Results/metrics/videos are mirrored every5min to /home/jh/getscore/outputs/rl-skills-long-20261103/live while the workstation is awake; authoritative server execution and artifacts do not depend on that mirror. No W&B key was supplied or required (offline logging explicitly selected).

Results: run RESULTS.md / RESULTS.json, metrics.jsonl, evaluation-*.json, videos/, checkpoints/, wandb/. Supervisor records bounded restarts/failure and finalizes automatically. RPent integration remains separate.
