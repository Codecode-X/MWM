#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Activate conda ode before invoking, or set PYTHON to its interpreter.
PYTHON="${PYTHON:-python3}"
OUTPUT="${OUTPUT:-results/comparison_cuda}"
mkdir -p "$OUTPUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -u -m src.comparison --output "$OUTPUT" \
  --jobs "${JOBS:-4}" --epochs "${EPOCHS:-50}" --device "${DEVICE:-cuda}" "$@" \
  2>&1 | tee -a "$OUTPUT/run.log"
"$PYTHON" -m reproduction.check_comparison "$OUTPUT" 2>&1 | tee "$OUTPUT/audit.log"
