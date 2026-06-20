#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/project

RUN_SHARED_BASELINE=1
RUN_GRAPH_INIT=1
RUN_BA=1
RUN_REPORT=1

SHARED_OBS="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations"
AP02_OBS="results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-shared-baseline)
      RUN_SHARED_BASELINE=0
      shift
      ;;
    --skip-graph-init)
      RUN_GRAPH_INIT=0
      shift
      ;;
    --skip-ba)
      RUN_BA=0
      shift
      ;;
    --skip-report)
      RUN_REPORT=0
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

import_shared_observations_for_ap02() {
  echo
  echo "=== Import shared ArUco observations into AP02 expected filenames ==="

  if [[ ! -f "$SHARED_OBS/shared_all_aruco_observations.csv" ]]; then
    echo "[ERROR] Missing shared observations: $SHARED_OBS"
    echo "Run: bash run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh"
    exit 1
  fi

  mkdir -p "$AP02_OBS"

  cp "$SHARED_OBS/shared_static_aruco_observations.csv" "$AP02_OBS/ap02_static_aruco_observations.csv"
  cp "$SHARED_OBS/shared_moving_aruco_observations.csv" "$AP02_OBS/ap02_moving_aruco_observations.csv"
  cp "$SHARED_OBS/shared_all_aruco_observations.csv" "$AP02_OBS/ap02_all_aruco_observations.csv"

  if [[ -f "$SHARED_OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" ]]; then
    cp "$SHARED_OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" "$AP02_OBS/ap02_detection_summary.txt"
    cp "$SHARED_OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" "$AP02_OBS/AP02_ARUCO_DETECTION_SUMMARY.txt"
  fi

  echo "[OK] AP02 now uses shared ArUco baseline."
}

echo "=== AP02: Ref-marker graph + BA pipeline ==="

if [[ "$RUN_SHARED_BASELINE" == "1" ]]; then
  echo
  echo "=== 1/5 Shared baseline preprocessing ==="
  bash run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh
else
  echo
  echo "=== 1/5 Skip shared baseline preprocessing ==="
fi

import_shared_observations_for_ap02

echo
echo "=== 2/5 AP02 debug artifacts from shared observations ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/03_make_ap02_debug_artifacts.py

if [[ "$RUN_GRAPH_INIT" == "1" ]]; then
  echo
  echo "=== 3/5 Graph initialization ==="
  bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase2_graph_init.sh
else
  echo
  echo "=== 3/5 Skip graph initialization ==="
fi

if [[ "$RUN_BA" == "1" ]]; then
  echo
  echo "=== 4/5 Graph bundle adjustment fast ==="
  bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_phase3_graph_ba_fast.sh
else
  echo
  echo "=== 4/5 Skip graph BA ==="
fi

echo
echo "=== Export final AP02 results ==="
python3 run/bus_real_data/approach2_ref_marker_graph_ba/08_export_ap02_final_results.py

if [[ "$RUN_REPORT" == "1" ]]; then
  echo
  echo "=== 5/5 Ref-ArUco comparison/report ==="
  python3 run/bus_real_data/approach_comparison_ref_aruco/01_eval_ap02_ref_aruco_vs_gt.py
  python3 run/bus_real_data/approach_comparison_ref_aruco/02_make_ap02_readable_ref_aruco_report.py
else
  echo
  echo "=== 5/5 Skip report ==="
fi

echo
echo "[OK] AP02 full pipeline complete."
