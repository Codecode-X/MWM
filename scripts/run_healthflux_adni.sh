#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
HEALTHFLUX_PYTHON="${HEALTHFLUX_PYTHON:-/home/xjh/miniforge3/envs/healthflux/bin/python}"
exec "$HEALTHFLUX_PYTHON" -u -m healthflux.study "$@"
