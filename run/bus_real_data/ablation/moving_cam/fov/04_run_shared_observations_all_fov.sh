#!/usr/bin/env bash
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"

DATASET_ROOT="results/bus_real_data/ablation/moving_cam/fov/00_prepared_datasets"
OUT_ROOT="results/bus_real_data/ablation/moving_cam/fov/01_shared_observations"
LOG_ROOT="$OUT_ROOT/_logs"

DETECT_SCRIPT="run/bus_real_data/_shared/baseline/02_detect_shared_aruco_observations.py"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$DETECT_SCRIPT" ]; then
  echo "[ERROR] detector script not found:"
  echo "  $DETECT_SCRIPT"
  exit 1
fi

echo "[INFO] dataset root: $DATASET_ROOT"
echo "[INFO] output root:  $OUT_ROOT"
echo

for raw_dir in "$DATASET_ROOT"/*/raw_images; do
  [ -d "$raw_dir" ] || continue

  variant="$(basename "$(dirname "$raw_dir")")"
  out_dir="$OUT_ROOT/$variant"
  log_file="$LOG_ROOT/${variant}_shared_observations.log"

  echo
  echo "============================================================"
  echo "[RUN] shared ArUco observations: $variant"
  echo "============================================================"
  echo "[INFO] dataset: $raw_dir"
  echo "[INFO] out:     $out_dir"
  echo "[INFO] log:     $log_file"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  PYTHONPATH=run/bus_real_data python3 "$DETECT_SCRIPT" \
    --dataset "$raw_dir" \
    --out "$out_dir" \
    2>&1 | tee "$log_file"

  echo "[OK] finished: $variant"
done

echo
echo "[OK] all shared observations finished."
echo
echo "---- output files ----"
find "$OUT_ROOT" -maxdepth 3 -type f | sort
