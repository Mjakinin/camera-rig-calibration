#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== AP02 Phase 1 legacy wrapper ==="
echo "AP02 no longer performs approach-specific ArUco detection."
echo "Using shared baseline instead."

bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase1_detect_debug.sh
