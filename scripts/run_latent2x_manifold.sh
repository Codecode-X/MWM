#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
ODE_PYTHON="${PYTHON:-/home/xjh/miniforge3/envs/ode/bin/python}"
"$ODE_PYTHON" -u -m paper_odeae.diagnostics.run_manifold "$@"
exec "$ODE_PYTHON" -u -m paper_odeae.diagnostics.run_manifold_followups "$@"
