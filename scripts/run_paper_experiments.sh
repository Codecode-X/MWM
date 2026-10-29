#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
EPOCHS="${EPOCHS:-30}"
OUTPUT="${OUTPUT:-results/paper_cuda}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -m unittest discover -s tests -v
pids=()
gpu=0
for task in interpolation extrapolation; do
  for model in latent_ode ode_rnn; do
    directory="$OUTPUT/${model}_${task}"
    mkdir -p "$directory"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PYTHON" -m src.train --model "$model" --task "$task" --epochs "$EPOCHS" \
        --device cuda --output "$directory" "$@" 2>&1 | tee "$directory/train.log"
      "$PYTHON" -m src.evaluate "$directory/best.pt" --device cuda > "$directory/test_reloaded.json"
    ) &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then exit 1; fi
"$PYTHON" -m reproduction.summarize_results "$OUTPUT" --output "$OUTPUT"
"$PYTHON" -m reproduction.check_results "$OUTPUT"
