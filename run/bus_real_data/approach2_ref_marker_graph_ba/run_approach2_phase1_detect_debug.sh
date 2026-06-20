#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/project

SHARED_OBS="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations"
AP02_OBS="results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations"

echo "=== AP02 Phase 1: use shared ArUco baseline ==="
bash run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh

mkdir -p "$AP02_OBS"
cp "$SHARED_OBS/shared_static_aruco_observations.csv" "$AP02_OBS/ap02_static_aruco_observations.csv"
cp "$SHARED_OBS/shared_moving_aruco_observations.csv" "$AP02_OBS/ap02_moving_aruco_observations.csv"
cp "$SHARED_OBS/shared_all_aruco_observations.csv" "$AP02_OBS/ap02_all_aruco_observations.csv"
cp "$SHARED_OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" "$AP02_OBS/ap02_detection_summary.txt"
cp "$SHARED_OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" "$AP02_OBS/AP02_ARUCO_DETECTION_SUMMARY.txt"

echo
echo "=== AP02 Phase 1: make debug artifacts ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/03_make_ap02_debug_artifacts.py

echo
echo "=== AP02 Detection Summary ==="
cat "$AP02_OBS/ap02_detection_summary.txt"

echo
echo "=== AP02 Debug Report ==="
cat results/bus_real_data/02_ref_marker_graph_ba/03_debug_artifacts/ap02_debug_artifacts_report.txt

echo
echo "[OK] AP02 Phase 1 shared-baseline/debug complete."
