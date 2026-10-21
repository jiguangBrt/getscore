#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
planner=fake
if [[ "${1:-}" == "--planner" ]]; then
  planner="${2:?Expected fake or codex}"
  shift 2
fi
exec ./scripts/run-liberopro.sh --planner "$planner" --config configs/liberopro-gawm.toml "$@"
