#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=pro
export LIBERO_PRO_ASSET_PATH="$PWD/data/libero-pro/assets"
export LIBERO_ASSET_PATH="$LIBERO_PRO_ASSET_PATH"
export LIBERO_CONFIG_PATH="$PWD/data/libero-pro/config"
export LIBERO_PRO_CONFIG_PATH="$PWD/data/libero-pro/config-pro"
export RPENT_RLINF_ROOT="$PWD/data/rlinf-runtime"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_CONFIG_PATH"
runner=scripts.run_libero_fake
if [[ "${1:-}" == "--planner" ]]; then
  case "${2:-}" in
    fake) runner=scripts.run_libero_fake ;;
    codex) runner=scripts.run_libero_codex ;;
    *) echo "Usage: $0 [--planner fake|codex] [runner options]" >&2; exit 2 ;;
  esac
  shift 2
fi
exec .venv/bin/python -m "$runner" "$@"
