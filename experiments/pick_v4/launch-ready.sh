#!/usr/bin/env bash
set -euo pipefail
PICK_BASE=/data/wangyihan/getscore-rl-skills
PICK_PROFILE="$PICK_BASE/experiments/pick-v4-ready-20261106"
export RL_SKILLS_CONFIG="$PICK_PROFILE/configs/ready.json"
PICK_HAS_SOURCE=false
for PICK_ARG in "$@"; do
  case "$PICK_ARG" in --resume|--resume=*|--initial|--initial=*) PICK_HAS_SOURCE=true;; esac
done
if [[ "$PICK_HAS_SOURCE" == false ]]; then
  PICK_SOURCE=$(python3 -c 'import hashlib,json,sys; from pathlib import Path; r=json.load(open(sys.argv[1])); p=Path(r["checkpoint_remote"]); assert hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"], "Validated checkpoint checksum changed"; print(p)' "$PICK_PROFILE/READY.json")
  set -- --resume "$PICK_SOURCE" "$@"
fi
exec bash "$PICK_PROFILE/run.sh" minibatch_ppo.py "$@"
