#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ROOT="results/bus_real_data/ablation/moving_cam/density"
DETECTOR="run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"
VARIANTS=(
  density_stride_1_100pct
  density_stride_2_50pct
  density_stride_4_25pct
  density_stride_8_12p5pct
)

for variant in "${VARIANTS[@]}"; do
  raw="$ROOT/$variant/raw_images"
  obs="$ROOT/$variant/aruco_observations"
  if [[ ! -d "$raw" ]]; then
    echo "[ERROR] missing density raw_images: $raw"
    exit 1
  fi
  rm -rf "$obs"
  python3 "$DETECTOR" \
    --dataset "$raw" \
    --out "$obs" \
    --dictionary DICT_4X4_50
  cat "$obs/SHARED_ARUCO_DETECTION_SUMMARY.txt"
done
