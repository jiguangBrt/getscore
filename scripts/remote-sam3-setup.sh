#!/usr/bin/env bash
set -euo pipefail
cd /services
export UV_CACHE_DIR=/services/uv-cache
if [[ ! -x .venv-sam3/bin/python ]]; then
  ./uv venv --system-site-packages --python /usr/bin/python3 .venv-sam3
fi
# Use the image's tested Torch/CUDA stack; install only SAM3's missing image
# inference dependencies instead of resolving a new Torch and CUDA toolkit.
UV_HTTP_TIMEOUT=120 ./uv pip install --python .venv-sam3/bin/python --no-deps \
  -e ./sam3 einops 'ftfy==6.1.1' regex iopath wcwidth portalocker tqdm
.venv-sam3/bin/python -c 'import sam3, torch; print(torch.__version__, sam3.__file__)'
