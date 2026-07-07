#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

RUN_METHODS=0
REUSE_AP03=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-methods)
      RUN_METHODS=1
      shift
      ;;
    --refresh-only)
      RUN_METHODS=0
      shift
      ;;
    --reuse-ap03)
      REUSE_AP03=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo
      echo "Usage:"
      echo "  $0 --refresh-only"
      echo "  $0 --run-methods [--reuse-ap03]"
      exit 2
      ;;
  esac
done

if [[ "$RUN_METHODS" == "1" ]]; then
  echo "================================================================================"
  echo "FULL-RUN PREFLIGHT"
  echo "================================================================================"

  if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not available."
    echo "[ERROR] No pipeline step was started."
    exit 127
  fi

  if ! command -v colmap >/dev/null 2>&1; then
    echo "[ERROR] COLMAP is not available."
    echo "[ERROR] No preprocessing or method pipeline was started."
    exit 127
  fi

  echo "[OK] Python: $(command -v python3)"
  echo "[OK] COLMAP: $(command -v colmap)"
fi

if [[ "$RUN_METHODS" == "1" ]]; then
  echo "================================================================================"
  echo "SHARED BASELINE"
  echo "================================================================================"

  bash \
    run/bus_real_data/_shared/baseline/run_shared_preprocessing.sh

  echo
  echo "================================================================================"
  echo "AP01"
  echo "================================================================================"

  RUN_SHARED_BASELINE=0 \
    bash \
    run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh

  echo
  echo "================================================================================"
  echo "AP02"
  echo "================================================================================"

  bash \
    run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
    --skip-shared-baseline

  echo
  echo "================================================================================"
  echo "AP03"
  echo "================================================================================"

  if [[ "$REUSE_AP03" == "1" ]]; then
    bash \
      run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh \
      --reuse-existing
  else
    bash \
      run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
  fi
fi

echo
echo "================================================================================"
echo "EVALUATION + FINAL REPORT REFRESH"
echo "================================================================================"

bash \
  run/bus_real_data/reporting/run_refresh_final_results.sh \
  --promote

echo
echo "[OK] Method/evaluation/reporting pipeline complete."
