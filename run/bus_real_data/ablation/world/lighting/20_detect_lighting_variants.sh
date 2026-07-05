#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

ROOT="results/bus_real_data/ablation/world/lighting"

VARIANTS=(
  ceiling_dark_extreme
  ceiling_low
  ceiling_normal
  ceiling_bright
)

for variant in "${VARIANTS[@]}"; do
    RAW="$ROOT/$variant/raw_images"
    OBS="$ROOT/$variant/aruco_observations"

    echo
    echo "================================================================================"
    echo "ARUCO DETECTION: $variant"
    echo "================================================================================"

    if [[ ! -d "$RAW/static" ]]; then
        echo "[ERROR] Missing static images: $RAW/static"
        exit 1
    fi

    if [[ ! -d "$RAW/moving" ]]; then
        echo "[ERROR] Missing moving images: $RAW/moving"
        exit 1
    fi

    STATIC_COUNT="$(
      find "$RAW/static" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
        | wc -l
    )"

    MOVING_COUNT="$(
      find "$RAW/moving" -maxdepth 1 -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
        | wc -l
    )"

    echo "static images: $STATIC_COUNT"
    echo "moving images: $MOVING_COUNT"

    if [[ "$STATIC_COUNT" -lt 4 ]]; then
        echo "[ERROR] Expected at least four static-camera images."
        exit 1
    fi

    if [[ "$MOVING_COUNT" -lt 1 ]]; then
        echo "[ERROR] No moving-camera images found."
        exit 1
    fi

    rm -rf "$OBS"

    python3 \
      run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py \
      --dataset "$RAW" \
      --out "$OBS" \
      --dictionary DICT_4X4_50 \
      2>&1 | tee "$ROOT/$variant/aruco_detection.log"

    if [[ -f "$OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt" ]]; then
        cat "$OBS/SHARED_ARUCO_DETECTION_SUMMARY.txt"
    else
        echo "[ERROR] Detector did not create its summary."
        exit 1
    fi
done
