#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

ROOT="results/bus_real_data/ablation/moving_cam/fov"

for variant in \
  fov_40deg \
  fov_69deg_baseline \
  fov_100deg \
  fov_140deg_extreme
do
  echo
  echo "================================================================================"
  echo "[MOVING RES DETECT] $variant"
  echo "================================================================================"

  RAW="$ROOT/$variant/raw_images"
  OBS="$ROOT/$variant/aruco_observations"

  rm -rf "$OBS"

  python3 run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py \
    --dataset "$RAW" \
    --out "$OBS" \
    --dictionary DICT_4X4_50 \
    2>&1 | tee "$ROOT/$variant/aruco_detection.log"

  cat "$OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt"
done
