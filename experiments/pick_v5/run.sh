#!/usr/bin/env bash
set -euo pipefail
TRAIN_BASE=/data/wangyihan/getscore-rl-skills
TRAIN_PROFILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$TRAIN_BASE/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export RL_SKILLS_CONFIG="${RL_SKILLS_CONFIG:-$TRAIN_PROFILE/configs/pilot.json}"
TRAIN_GPU_UUIDS=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["gpu_uuids"]))' "$RL_SKILLS_CONFIG")
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_UUIDS" RENDER_LOGICAL_CUDA=0
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$TRAIN_PROFILE/code:$TRAIN_BASE/src/RLinf:$TRAIN_BASE/src/openpi/src:$TRAIN_BASE/src/openpi/packages/openpi-client/src:$TRAIN_BASE/src/LIBERO-PRO"
export WANDB_MODE=offline WANDB_DIR="$TRAIN_BASE/runs" WANDB_CACHE_DIR="$TRAIN_BASE/cache/wandb" WANDB_CONFIG_DIR="$TRAIN_BASE/configs/wandb" WANDB_DATA_DIR="$TRAIN_BASE/cache/wandb-data"
export TMPDIR="$TRAIN_BASE/cache/tmp"
cd "$TRAIN_BASE"
exec "$TRAIN_BASE/.venv/bin/python" -u "$TRAIN_PROFILE/code/$1" "${@:2}"
