#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export AUTO_CONFIRM=1

echo "===== BASELINE PAIRWISE RERUN — NO RES, NO BACKUP ====="

echo
echo "===== HARD CLEAN OLD METHOD OUTPUTS AND OLD FINAL REPORT ====="
rm -rf \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/90_approach_comparison_ref_aruco \
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT

echo
echo "===== INPUT CHECK ====="
test -d results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
test -d results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations
echo "[OK] shared baseline exists"

echo
echo "===== AP01 ====="
bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh

echo
echo "===== AP02 ====="
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh

echo
echo "===== AP03 TARGETLESS COLMAP ====="
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
  --moving-stride 1 \
  --max-moving 0

if ! command -v colmap >/dev/null 2>&1; then
  echo "[FAIL] colmap not found"
  exit 1
fi

bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py

echo
echo "===== AP03 MARKER-SIZE-ONLY SCALE, NO SDF MARKER MAP ====="
set +e
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/10_estimate_scale_from_marker_size_only.py \
  --marker-ids 0-14 \
  --min-area-px2 100 \
  --reproj-thresh-px 5 \
  --ransac-iters 1000 \
  --min-inliers 4
AP03_SCALE_CODE=$?
set -e

if [[ "$AP03_SCALE_CODE" -ne 0 ]]; then
  echo "[WARN] AP03 marker-size scale failed. Final pairwise report will mark AP03 as FAILED."
fi

echo
echo "===== COMMON PAIRWISE FINAL EVALUATION ====="
python3 run/bus_real_data/evaluation/10_eval_pairwise_static_camera_extrinsics.py

echo
echo "===== HARD GUARD: NO RES SOURCE PATHS IN FINAL REPORT ====="
if grep -RIn "results/bus_real_data/ablation/moving_cam/res\|results/bus_real_data/ablation/moving_cam_res" \
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT; then
  echo "[FAIL] forbidden RES source leaked into final report"
  exit 1
fi
echo "[OK] no RES source paths"

echo
echo "===== DONE ====="
echo "[OK] final report:"
echo "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"
