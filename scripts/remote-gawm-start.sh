#!/usr/bin/env bash
set -euo pipefail
GAWM_ROOT=/mnt/reacher-fast/gawm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
exec "$GAWM_ROOT/.venv-container/bin/python" "$GAWM_ROOT/services/gawm_policy_server.py" --root "$GAWM_ROOT"
