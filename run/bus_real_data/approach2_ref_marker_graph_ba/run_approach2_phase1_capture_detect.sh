#!/usr/bin/env bash
set -eo pipefail

cd /workspaces/project

echo "=== AP02 Phase 1: capture independent AP02 dataset ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/01_capture_ap02_dataset.py --moving-frames 204 --moving-dt 0.10

echo
echo "=== AP02 Phase 1: detect AP02 ArUco observations ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/02_detect_ap02_aruco_observations.py

echo
echo "=== AP02 Phase 1: make debug artifacts ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/03_make_ap02_debug_artifacts.py

echo
echo "=== AP02 Detection Summary ==="
cat results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_detection_summary.txt

echo
echo "=== AP02 Debug Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/03_debug_artifacts/ap02_debug_artifacts_report.txt

echo
echo "[OK] AP02 Phase 1 complete."
