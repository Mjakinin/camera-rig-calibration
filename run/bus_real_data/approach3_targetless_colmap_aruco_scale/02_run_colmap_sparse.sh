#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec python3 \
  run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse_grouped.py \
  "$@"
