#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

variant="${1:?usage: $0 <variant>}"

ROOT="results/bus_real_data/ablation/moving_cam/res"
VAR_ROOT="$ROOT/$variant"
VAR_FINAL="$VAR_ROOT/FINAL_RESULTS"
SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
RUNLOG="$VAR_ROOT/RUN_FULL_AP01_AP02_AP03.log"

mkdir -p "$VAR_FINAL"

exec > >(tee "$RUNLOG") 2>&1

echo "================================================================================"
echo "[RES VARIANT] $variant"
echo "================================================================================"

if [ ! -d "$VAR_ROOT/raw_images" ]; then
  echo "[ERROR] missing raw_images: $VAR_ROOT/raw_images"
  exit 1
fi

if [ ! -d "$VAR_ROOT/aruco_observations" ]; then
  echo "[ERROR] missing aruco_observations. Run 11_detect_clean_res_variants.sh first."
  exit 1
fi

echo
echo "=== Install variant into shared baseline ==="
rm -rf "$SHARED/raw_images"
rm -rf "$SHARED/aruco_observations"
rm -rf "$SHARED/metadata"

mkdir -p "$SHARED"
cp -a "$VAR_ROOT/raw_images" "$SHARED/raw_images"
cp -a "$VAR_ROOT/aruco_observations" "$SHARED/aruco_observations"

if [ -d "$VAR_ROOT/metadata" ]; then
  cp -a "$VAR_ROOT/metadata" "$SHARED/metadata"
else
  mkdir -p "$SHARED/metadata"
fi

echo "[OK] installed $variant into $SHARED"

echo
echo "=== Clean method outputs ==="
rm -rf results/bus_real_data/01_marker_direct_relay_multimarker_multichain
rm -rf results/bus_real_data/02_ref_marker_graph_ba
rm -rf results/bus_real_data/03_targetless_colmap_aruco_scale
rm -rf results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT

echo
echo "=== AP01 ==="
AP01_STATUS="OK"
RUN_SHARED_BASELINE=0 bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh || AP01_STATUS="FAILED"

echo
echo "=== AP01 direct cam3->cam1 guard ==="
python3 - <<'PY2'
import csv
from pathlib import Path
p = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv")
rows = list(csv.DictReader(p.open()))
ok = False
for r in rows:
    if r.get("category") == "main_no_gt" and r.get("target_camera") == "cam_edge_1":
        method = r.get("method", "")
        pair = r.get("pair", "")
        if "direct_static" in method and pair == "cam3_to_cam1":
            ok = True
        print("[AP01 cam1]", pair, method, r.get("translation_error_cm"), r.get("rotation_error_deg"))
if not ok:
    raise SystemExit("[ERROR] AP01 cam3->cam1 is not direct_static in final_extrinsics_summary.csv")
print("[OK] AP01 cam3->cam1 direct static guard passed")
PY2

echo
echo "=== AP02 ==="
AP02_STATUS="OK"
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh --skip-shared-baseline --skip-report || AP02_STATUS="FAILED"

echo
echo "=== AP03 targetless COLMAP + marker-size-only scale ==="
AP03_STATUS="OK"
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
  --moving-stride 1 \
  --max-moving 0 || AP03_STATUS="FAILED_PREPARE"

if [ "$AP03_STATUS" = "OK" ]; then
  bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh || AP03_STATUS="FAILED_COLMAP"
fi

if [ "$AP03_STATUS" = "OK" ]; then
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py || AP03_STATUS="FAILED_INSPECT"
fi

if [ "$AP03_STATUS" = "OK" ]; then
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/10_estimate_scale_from_marker_size_only.py \
    --marker-ids 0-14 \
    --min-area-px2 100 \
    --reproj-thresh-px 5 \
    --ransac-iters 1000 \
    --min-inliers 4 || AP03_STATUS="FAILED_SCALE"
fi

echo
echo "=== Primary pairwise evaluator ==="
PAIRWISE_STATUS="OK"
python3 run/bus_real_data/evaluation/10_eval_pairwise_static_camera_extrinsics.py || PAIRWISE_STATUS="FAILED"

echo
echo "=== Secondary Ref14/world camera-map evaluator ==="
SECONDARY_STATUS="OK"
python3 run/bus_real_data/evaluation/11_eval_secondary_ref14_world_camera_maps.py || SECONDARY_STATUS="FAILED"

echo
echo "=== Collect FINAL_RESULTS ==="
rm -rf "$VAR_FINAL"
mkdir -p "$VAR_FINAL"

if [ -d results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT ]; then
  cp -a results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/. "$VAR_FINAL/"
fi

cp "$VAR_ROOT/VARIANT_METADATA.json" "$VAR_FINAL/VARIANT_METADATA.json" 2>/dev/null || true
cp "$VAR_ROOT/aruco_observations/SHARED_ARUCO_DETECTION_SUMMARY.txt" "$VAR_FINAL/DIAGNOSTIC_SHARED_ARUCO_DETECTION_SUMMARY.txt" 2>/dev/null || true

cp results/bus_real_data/03_targetless_colmap_aruco_scale/03_reconstruction_inspection/ap03_colmap_inspection_report.txt \
   "$VAR_FINAL/DIAGNOSTIC_AP03_COLMAP_RECONSTRUCTION.txt" 2>/dev/null || true

cp results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_REPORT.txt \
   "$VAR_FINAL/DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.txt" 2>/dev/null || true

cp results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json \
   "$VAR_FINAL/DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.json" 2>/dev/null || true

cp results/bus_real_data/02_ref_marker_graph_ba/08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.txt \
   "$VAR_FINAL/DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.txt" 2>/dev/null || true

cp results/bus_real_data/02_ref_marker_graph_ba/08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv \
   "$VAR_FINAL/DIAGNOSTIC_AP02_GT_ALIGNED_FULL_MARKER_MAP.csv" 2>/dev/null || true

cat > "$VAR_FINAL/RUN_STATUS.txt" <<TXT
variant=$variant
AP01_STATUS=$AP01_STATUS
AP02_STATUS=$AP02_STATUS
AP03_STATUS=$AP03_STATUS
PAIRWISE_STATUS=$PAIRWISE_STATUS
SECONDARY_STATUS=$SECONDARY_STATUS
TXT

echo
echo "=== RUN STATUS ==="
cat "$VAR_FINAL/RUN_STATUS.txt"

echo
echo "[OK] variant final results written to $VAR_FINAL"
