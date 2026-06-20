#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/project

export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

RUN_PREPARE=1
RUN_COLMAP=1
RUN_INSPECT=1
RUN_REGISTRATION=1
MIN_AREA_PX2=1000
REPROJ_THRESH_PX=5

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
    --skip-registration)
      RUN_REGISTRATION=0
      shift
      ;;
    --reuse-existing)
      RUN_PREPARE=0
      RUN_COLMAP=0
      RUN_INSPECT=0
      RUN_REGISTRATION=0
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
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

OUT="results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration"

echo "=== AP03: Targetless COLMAP + ArUco scale registration ==="
echo "MIN_AREA_PX2=${MIN_AREA_PX2}"
echo "REPROJ_THRESH_PX=${REPROJ_THRESH_PX}"
echo

if [[ "$RUN_PREPARE" == "1" ]]; then
  echo "=== 1/7 Prepare COLMAP dataset from shared raw images ==="
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
    --moving-stride 1 \
    --max-moving 0
else
  echo "=== 1/7 Skip dataset preparation ==="
fi

if [[ "$RUN_COLMAP" == "1" ]]; then
  echo
  echo "=== 2/7 Run COLMAP sparse reconstruction ==="
  if ! command -v colmap >/dev/null 2>&1; then
    echo "[ERROR] colmap not found."
    exit 1
  fi
  bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh
else
  echo
  echo "=== 2/7 Skip COLMAP reconstruction ==="
fi

if [[ "$RUN_INSPECT" == "1" ]]; then
  echo
  echo "=== 3/7 Inspect COLMAP reconstruction ==="
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py
else
  echo
  echo "=== 3/7 Skip COLMAP inspection ==="
fi

if [[ "$RUN_REGISTRATION" == "1" ]]; then
  echo
  echo "=== 4/7 Detect Ref14 scale observations ==="
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06a_detect_ref14_scale_observations.py \
    --out-root "$OUT" \
    --min-area-px2 "$MIN_AREA_PX2"

  echo
  echo "=== 5/7 Triangulate Ref14 corners and estimate Sim3 ==="
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06b_triangulate_ref14_corners.py \
    --out-root "$OUT" \
    --reproj-thresh-px "$REPROJ_THRESH_PX" \
    --ransac-iters 1000 \
    --min-inliers 4

  echo
  echo "=== 6/7 Apply Ref14 registration to COLMAP model ==="
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06c_apply_ref14_registration_to_colmap.py \
    --out-root "$OUT"
else
  echo
  echo "=== 4-6/7 Reuse existing AP03 registration outputs ==="
fi

echo
echo "=== 7/7 Evaluate and report AP03 ==="
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06d_eval_ap03_static_cameras.py \
  --out-root "$OUT"

python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06e_make_ap03_final_report.py \
  --out-root "$OUT"

echo
echo "[OK] AP03 full pipeline complete."
