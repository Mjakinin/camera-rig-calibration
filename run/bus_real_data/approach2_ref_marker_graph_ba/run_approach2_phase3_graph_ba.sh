#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== AP02 Phase 3: Graph BA static_only ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py \
  --mode static_only \
  --max-nfev 80

echo
echo "=== AP02 Phase 3: Graph BA with_moving ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/07_run_ref_marker_graph_ba.py \
  --mode with_moving \
  --max-nfev 80

echo
echo "=== AP02 BA static_only summary ==="
cat results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/static_only/ba_summary.txt

echo
echo "=== AP02 BA with_moving summary ==="
cat results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/with_moving/ba_summary.txt

echo
echo "[OK] AP02 Phase 3 complete."
