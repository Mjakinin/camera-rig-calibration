#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

ROOT="results/bus_real_data/ablation/moving_cam/res"

for variant in \
  moving_res_160x90_extreme_pixel \
  moving_res_320x180_low \
  moving_res_1280x720_baseline \
  moving_res_2560x1440_upscaled
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
