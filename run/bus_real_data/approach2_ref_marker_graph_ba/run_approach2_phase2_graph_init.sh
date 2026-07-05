#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== AP02 Phase 2: sanity check AP02 Python independence ==="
if grep -R "01_marker_direct_relay\|AP1_ROOT\|AP01_ROOT\|moving_detections.csv\|01_static_a4_marker_detection" -n \
  run/bus_real_data/approach2_ref_marker_graph_ba \
  --include="*.py"; then
  echo "[ERROR] AP02 Python scripts reference AP01 result internals. Stop."
  exit 1
fi
echo "[OK] AP02 Python scripts do not reference AP01 internals."

echo
echo "=== AP02 Phase 2: single reference marker PnP baseline ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/04_single_ref_marker_pnp_baseline.py

echo
echo "=== AP02 Phase 2: graph initialization static_only ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py --mode static_only

echo
echo "=== AP02 Phase 2: graph initialization with_moving ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/05_initialize_ref_marker_pose_graph_v2.py --mode with_moving

echo
echo "=== AP02 Phase 2: compare graph initialization variants ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/06_compare_graph_initialization.py

echo
echo "=== Single Reference Marker Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/04_single_ref_marker_pnp/single_ref_marker_report.txt

echo
echo "=== Static-only Graph Connectivity ==="
cat results/bus_real_data/02_ref_marker_graph_ba/05_graph_initialization/static_only/graph_connectivity_report.txt

echo
echo "=== With-moving Graph Connectivity ==="
cat results/bus_real_data/02_ref_marker_graph_ba/05_graph_initialization/with_moving/graph_connectivity_report.txt

echo
echo "=== Comparison Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/06_graph_initialization_comparison/graph_initialization_comparison_report.txt

echo
echo "[OK] AP02 Phase 2 complete."
