#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

RUN_PREPARE=1
RUN_COLMAP=1
RUN_INSPECT=1
RUN_SCALE=1
MIN_AREA_PX2=100
REPROJ_THRESH_PX=5
RANSAC_ITERS=1000
MIN_INLIERS=4
REUSE_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-prepare)
      RUN_PREPARE=0
      shift
      ;;
    --skip-colmap)
      RUN_COLMAP=0
      shift
      ;;
    --skip-inspect)
      RUN_INSPECT=0
      shift
      ;;
    --skip-scale)
      RUN_SCALE=0
      shift
      ;;
    --reuse-existing)
      RUN_PREPARE=0
      RUN_COLMAP=0
      RUN_INSPECT=0
      RUN_SCALE=0
      REUSE_EXISTING=1
      shift
      ;;
    --min-area-px2)
      MIN_AREA_PX2="$2"
      shift 2
      ;;
    --reproj-thresh-px)
      REPROJ_THRESH_PX="$2"
      shift 2
      ;;
    --ransac-iters)
      RANSAC_ITERS="$2"
      shift 2
      ;;
    --min-inliers)
      MIN_INLIERS="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 2
      ;;
  esac
done

AP3_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
CANONICAL_POSES="$AP3_ROOT/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
CANONICAL_META="$AP3_ROOT/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
CANONICAL_REPORT="$AP3_ROOT/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_REPORT.txt"

if [[ "$REUSE_EXISTING" == "0" ]]; then
  echo "=== Clean AP03 generated outputs ==="
  rm -rf \
    "$AP3_ROOT/01_colmap_dataset" \
    "$AP3_ROOT/02_colmap_sparse" \
    "$AP3_ROOT/03_reconstruction_inspection" \
    "$AP3_ROOT/06_triangulated_ref_aruco_registration" \
    "$AP3_ROOT/07_final_results"
fi

echo "=== AP03: targetless COLMAP + marker-size-only metric scale ==="
echo "MIN_AREA_PX2=$MIN_AREA_PX2"
echo "REPROJ_THRESH_PX=$REPROJ_THRESH_PX"
echo "RANSAC_ITERS=$RANSAC_ITERS"
echo "MIN_INLIERS=$MIN_INLIERS"
echo

if [[ "$RUN_PREPARE" == "1" ]]; then
  echo "=== 1/4 Prepare COLMAP dataset from shared raw images ==="
  python3 \
    run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
    --moving-stride 1 \
    --max-moving 0
else
  echo "=== 1/4 Reuse prepared dataset ==="
fi

if [[ "$RUN_COLMAP" == "1" ]]; then
  echo
  echo "=== 2/4 Run grouped calibrated COLMAP sparse reconstruction ==="
  bash \
    run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh
else
  echo
  echo "=== 2/4 Reuse COLMAP sparse reconstruction ==="
fi

if [[ "$RUN_INSPECT" == "1" ]]; then
  echo
  echo "=== 3/4 Inspect COLMAP reconstruction ==="
  python3 \
    run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py
else
  echo
  echo "=== 3/4 Reuse reconstruction inspection ==="
fi

if [[ "$RUN_SCALE" == "1" ]]; then
  echo
  echo "=== 4/4 Estimate metric scale from marker side lengths ==="
  python3 \
    run/bus_real_data/approach3_targetless_colmap_aruco_scale/10_estimate_scale_from_marker_size_only.py \
    --marker-ids 0-14 \
    --min-area-px2 "$MIN_AREA_PX2" \
    --reproj-thresh-px "$REPROJ_THRESH_PX" \
    --ransac-iters "$RANSAC_ITERS" \
    --min-inliers "$MIN_INLIERS"
else
  echo
  echo "=== 4/4 Reuse marker-size-only metric scale ==="
fi

for required in "$CANONICAL_POSES" "$CANONICAL_META" "$CANONICAL_REPORT"; do
  if [[ ! -s "$required" ]]; then
    echo "[ERROR] Missing canonical AP03 output: $required"
    exit 1
  fi
done

python3 - "$CANONICAL_POSES" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
expected = {"cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"}
found = {row.get("entity_id", "") for row in rows}
missing = sorted(expected - found)
if missing:
    raise SystemExit(f"[ERROR] AP03 canonical output missing cameras: {missing}")
print(f"[OK] AP03 canonical camera coverage: {len(expected)}/{len(expected)}")
PY

echo
echo "[OK] AP03 full pipeline complete."
echo "[OK] Canonical poses: $CANONICAL_POSES"
echo "[OK] Canonical report: $CANONICAL_REPORT"
