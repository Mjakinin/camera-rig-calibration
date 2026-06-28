#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

DATASET="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images"
OUT="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations"

RUN_DETECT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-detect)
      RUN_DETECT=0
      shift
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== Shared bus real-data preprocessing ==="
echo "DATASET=$DATASET"
echo "OUT=$OUT"

if [[ ! -d "$DATASET" ]]; then
  echo "[ERROR] missing shared raw dataset: $DATASET"
  exit 1
fi

if [[ "$RUN_DETECT" == "1" ]]; then
  echo
  echo "=== Detect shared ArUco observations ==="
  python3 run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py \
    --dataset "$DATASET" \
    --out "$OUT"
else
  echo
  echo "=== Skip shared ArUco detection ==="
fi

echo
echo "[OK] shared preprocessing complete."
