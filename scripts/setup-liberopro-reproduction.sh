#!/usr/bin/env bash
# Prepare the local simulator stack; inference weights stay on 49.233.66.228.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: $0"
  echo "Requires uv, Git, Linux x86_64 and an NVIDIA driver with EGL support."
  echo "Downloads pinned runtime sources, Python packages and LIBERO-PRO assets."
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unexpected arguments; use --help." >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1
command -v uv >/dev/null
command -v git >/dev/null
git submodule update --init --recursive RPent

# Existing checkouts are verified, never reset or overwritten.
pinned_source() {
  local directory="$1" repository="$2" revision="$3"
  if [[ ! -e "$directory" ]]; then
    git init "$directory"
    git -C "$directory" remote add origin "$repository"
    git -C "$directory" fetch --depth=1 origin "$revision"
    git -C "$directory" checkout --detach FETCH_HEAD
  fi
  if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
    echo "Revision mismatch in $directory; inspect it before continuing." >&2
    exit 1
  fi
  git -C "$directory" diff --quiet HEAD --
}
mkdir -p data
pinned_source data/rlinf-runtime https://github.com/RLinf/RLinf.git \
  b6c6250587fcbd31dbc1d59ea1907f888a0dd748
pinned_source data/libero-runtime https://github.com/RLinf/LIBERO.git \
  a8323074d93a09e32bd898630a70531b1f51bc77
pinned_source data/liberopro-runtime https://github.com/RLinf/LIBERO-PRO.git \
  d1e11fb181b8544487d27742c0caa3a4d46452ad
if [[ ! -e data/openpi-client ]]; then
  pinned_source data/openpi-runtime https://github.com/RLinf/openpi.git \
    3ab5477b975d5683b2368d7c4cb97e83cb86e36f
  ln -s openpi-runtime/packages/openpi-client data/openpi-client
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
# Complete installed-package snapshot, with portable editable paths. Avoid
# resolving the unrelated training extras or installing model servers locally.
uv pip install --python .venv/bin/python --no-deps \
  -r configs/liberopro-reproduction.requirements.txt

.venv/bin/python - <<'PY'
from pathlib import Path
import json
import os
from huggingface_hub import snapshot_download
import yaml

root = Path.cwd()
assets = root / 'data/libero-pro/assets'
snapshot_download(
    repo_id='RLinf/LIBERO-PRO-assets', repo_type='dataset',
    revision='e4b10c9adc5735c9c8776917527178e693ef10b3',
    local_dir=str(assets), token=False,
)
for relative in ('data/libero-runtime/libero/libero/assets',
                 'data/liberopro-runtime/liberopro/liberopro/assets'):
    target = root / relative
    if target.is_symlink():
        if target.resolve() != assets:
            raise RuntimeError(f'Existing assets link points elsewhere: {target}')
    elif target.exists():
        raise RuntimeError(f'Existing assets directory must be inspected: {target}')
    else:
        target.symlink_to(os.path.relpath(assets, target.parent), target_is_directory=True)

benchmark = root / 'data/liberopro-runtime/liberopro/liberopro'
config = dict(benchmark_root=str(benchmark), bddl_files=str(benchmark / 'bddl_files'),
              init_states=str(benchmark / 'init_files'), assets=str(assets),
              datasets=str(root / 'data/datasets'))
directory = root / 'data/libero-pro/config'
directory.mkdir(parents=True, exist_ok=True)
path = directory / 'config.yaml'
if path.exists() and yaml.safe_load(path.read_text()) != config:
    raise RuntimeError(f'Existing simulator configuration differs; inspect {path}')
path.write_text(yaml.safe_dump(config))
(root / 'data/libero-pro/config-pro').mkdir(parents=True, exist_ok=True)
print(json.dumps(dict(status='prepared', root=str(root), assets=str(assets))))
PY
echo "Local setup complete. Connect the inference tunnel, then follow README.md."
