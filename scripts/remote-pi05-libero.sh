#!/usr/bin/env bash
set -euo pipefail
OPENPI_PROJECT=/mnt/reacher-fast/openpi_ur_pp_202607
cd "$OPENPI_PROJECT/repo"
export CUDA_VISIBLE_DEVICES=${PI05_GPU:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_DATA_HOME="$OPENPI_PROJECT/checkpoints"
export JAX_COMPILATION_CACHE_DIR="$OPENPI_PROJECT/cache/jax_libero_sync"
test -f "$OPENPI_PROJECT/checkpoints/openpi-assets/checkpoints/pi05_libero/download-manifest.json" || {
  echo "Official checkpoint verification manifest is missing" >&2
  exit 1
}
exec .venv/bin/python scripts/serve_policy.py \
  --env LIBERO --port 8008 \
  policy:checkpoint --policy.config pi05_libero \
  --policy.dir "$OPENPI_PROJECT/checkpoints/openpi-assets/checkpoints/pi05_libero"
