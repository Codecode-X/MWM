#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/xjh/miniforge3/envs/ode/bin/python -u -m mimic_experiment.init_covariate_ablation_study "$@"
