#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

exec python3 run/real_vehicle_data/13_evaluate_real_marker_table3.py "$@"
