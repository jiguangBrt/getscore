#!/usr/bin/env bash
set -euo pipefail
exec ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18008:127.0.0.1:18008 \
  -L 127.0.0.1:18114:127.0.0.1:18114 \
  -L 127.0.0.1:18115:127.0.0.1:18115 \
  "${INFERENCE_SSH_TARGET:-wangyihan@49.233.66.228}"
