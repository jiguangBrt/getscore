#!/usr/bin/env bash
set -euo pipefail
cd /services/rpent
export PYTHONPATH=/services/rpent
export SAM3_CHECKPOINT_PATH=/models/sam3.pt
# This image's Torch uses CUDA 13; its NVRTC builtins must be discoverable.
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
exec /services/.venv-sam3/bin/python /services/rpent/rpent/robots/components/sam3_server.py \
  --cuda-device 0 --transport http --host 192.168.50.73 --port 8114
