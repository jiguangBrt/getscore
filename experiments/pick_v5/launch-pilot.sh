#!/usr/bin/env bash
set -euo pipefail
TRAIN_BASE=/data/wangyihan/getscore-rl-skills
TRAIN_PROFILE="$TRAIN_BASE/experiments/pick-v5-validated-20261106"
export RL_SKILLS_CONFIG="$TRAIN_PROFILE/configs/validated.json"
TRAIN_EXPLICIT_SOURCE=0
for TRAIN_ARG in "$@"; do
  if [[ "$TRAIN_ARG" == --resume || "$TRAIN_ARG" == --initial ]]; then TRAIN_EXPLICIT_SOURCE=1; fi
done
if [[ "$TRAIN_EXPLICIT_SOURCE" == 0 ]]; then
  TRAIN_CHECKPOINT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_remote"])' "$TRAIN_PROFILE/VALIDATED.json")
  python3 - "$TRAIN_PROFILE/VALIDATED.json" "$TRAIN_CHECKPOINT" <<'PY'
import hashlib,json,sys
manifest=json.load(open(sys.argv[1]))
digest=hashlib.sha256()
with open(sys.argv[2],'rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
assert digest.hexdigest()==manifest['sha256'],'selected checkpoint SHA256 mismatch'
PY
  set -- --resume "$TRAIN_CHECKPOINT" "$@"
fi
exec "$TRAIN_PROFILE/run.sh" minibatch_ppo.py "$@"
