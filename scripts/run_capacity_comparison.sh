#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/home/xjh/miniforge3/envs/ode/bin/python}"
OUTPUT="${OUTPUT:-results/capacity_cuda}"
mkdir -p "$OUTPUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -u -m reproduction.scale_comparison --output "$OUTPUT" \
  --epochs "${EPOCHS:-20}" --jobs "${JOBS:-4}" "$@" 2>&1 | tee -a "$OUTPUT/run.log"
for protocol in native_base native_matched bce_base bce_matched; do
  "$PYTHON" -m reproduction.check_comparison "$OUTPUT/$protocol" | tee "$OUTPUT/$protocol/audit.log"
done
"$PYTHON" -m reproduction.summarize_capacity "$OUTPUT"
