#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/xjh/miniforge3/envs/ode/bin/python -u -m mimic_experiment.weak_prior_ablation_study "$@"
