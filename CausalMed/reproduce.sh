#!/usr/bin/env bash
set -euo pipefail
CAUSALMED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$CAUSALMED_ROOT")"
REPRO_OUTPUT="${1:-$CAUSALMED_ROOT/outputs/reproduced}"
if [[ -e "$REPRO_OUTPUT" ]]; then
  echo "Output path already exists: $REPRO_OUTPUT" >&2
  exit 1
fi
mkdir -p "$REPRO_OUTPUT"
REPRO_OUTPUT="$(cd -- "$REPRO_OUTPUT" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$CAUSALMED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
conda run --no-capture-output -n ode pytest -c "$CAUSALMED_ROOT/pytest.ini" \
  "$CAUSALMED_ROOT/synthetic_cf_level3/tests" -q | tee "$REPRO_OUTPUT/pytest.log"
conda run --no-capture-output -n ode python -m synthetic_cf_level3.calibration \
  --patients 500 --out-config "$REPRO_OUTPUT/final_config.yaml" \
  --report "$REPRO_OUTPUT/training_cost_calibration.json"
conda run --no-capture-output -n ode python -m synthetic_cf_level3.generate \
  --config "$REPRO_OUTPUT/final_config.yaml" --smoke --out-dir "$REPRO_OUTPUT/smoke_100"
conda run --no-capture-output -n ode python -m synthetic_cf_level3.generate \
  --config "$REPRO_OUTPUT/final_config.yaml" --all-gammas --out-dir "$REPRO_OUTPUT/level3_suite"
