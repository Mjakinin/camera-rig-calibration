#!/usr/bin/env bash
set -u -o pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "[ERROR] current directory is not inside a git repository"
  exit 2
fi
cd "$REPO_ROOT"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found in PATH=$PATH"
  exit 127
fi

ABL_ROOT="${1:?usage: $0 <ablation_root> <label> <variant>}"
ABL_LABEL="${2:?usage: $0 <ablation_root> <label> <variant>}"
variant="${3:?usage: $0 <ablation_root> <label> <variant>}"

VAR_ROOT="$ABL_ROOT/$variant"
VAR_FINAL="$VAR_ROOT/FINAL_RESULTS"
SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
RUNLOG="$VAR_ROOT/RUN_FULL_AP01_AP02_AP03.log"
SIM_MARKER_IDS="${SIM_MARKER_IDS:-0-14,16-20}"

for required in \
  "$VAR_ROOT/raw_images" \
  "$VAR_ROOT/aruco_observations/shared_all_aruco_observations.csv"
do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] missing variant input: $required"
    exit 1
  fi
done

mkdir -p "$VAR_ROOT"
exec > >(tee "$RUNLOG") 2>&1

echo "================================================================================"
echo "$ABL_LABEL $variant"
echo "SIM_MARKER_IDS=$SIM_MARKER_IDS"
echo "================================================================================"

echo
echo "=== Install immutable variant input as temporary shared baseline ==="
rm -rf "$SHARED/raw_images" "$SHARED/aruco_observations" "$SHARED/metadata"
mkdir -p "$SHARED"
cp -a "$VAR_ROOT/raw_images" "$SHARED/raw_images"
cp -a "$VAR_ROOT/aruco_observations" "$SHARED/aruco_observations"
if [[ -d "$VAR_ROOT/metadata" ]]; then
  cp -a "$VAR_ROOT/metadata" "$SHARED/metadata"
else
  mkdir -p "$SHARED/metadata"
fi

echo
echo "=== Clean method outputs ==="
rm -rf \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale

now_seconds() {
  python3 - <<'PY'
import time
print(time.time())
PY
}

elapsed_seconds() {
  python3 - "$1" <<'PY'
import sys
import time
print(f"{time.time() - float(sys.argv[1]):.6f}")
PY
}

AP01_STATUS="OK"
AP02_STATUS="OK"
AP03_STATUS="OK"
AP01_STARTED="$(now_seconds)"
echo
echo "=== AP01 final baseline ==="
RUN_SHARED_BASELINE=0 \
  bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh \
  || AP01_STATUS="FAILED"
AP01_RUNTIME_SECONDS="$(elapsed_seconds "$AP01_STARTED")"

if [[ "$AP01_STATUS" == "OK" ]]; then
  python3 - <<'PY' || AP01_STATUS="FAILED_DIRECT_GUARD"
import csv
from pathlib import Path
path = Path(
    "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/"
    "07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv"
)
rows = list(csv.DictReader(path.open()))
valid = any(
    row.get("category") == "main_no_gt"
    and row.get("target_camera") == "cam_edge_1"
    and row.get("pair") == "cam3_to_cam1"
    and "direct_static" in row.get("method", "")
    for row in rows
)
if not valid:
    raise SystemExit("[ERROR] AP01 cam3->cam1 final solution is not direct_static")
print("[OK] AP01 cam3->cam1 direct-static guard")
PY
fi

AP02_STARTED="$(now_seconds)"
echo
echo "=== AP02 final distortion-aware soft-L1 graph BA ==="
unset AP02_ROOT AP02_REF_MARKER_ID AP02_OBS SHARED_OBS || true
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh \
  --skip-shared-baseline --skip-report \
  || AP02_STATUS="FAILED"
AP02_RUNTIME_SECONDS="$(elapsed_seconds "$AP02_STARTED")"

AP02_GT_FULL_MAP_STATUS="NOT_AVAILABLE"
AP02_GT_STATUS_FILE="results/bus_real_data/02_ref_marker_graph_ba/08_final_results/AP02_GT_ALIGNED_FULL_MAP_STATUS.txt"
if [[ -f "$AP02_GT_STATUS_FILE" ]]; then
  AP02_GT_FULL_MAP_STATUS="$(
    awk -F= '$1 == "status" {print $2}' "$AP02_GT_STATUS_FILE" | tail -n 1
  )"
  AP02_GT_FULL_MAP_STATUS="${AP02_GT_FULL_MAP_STATUS:-NOT_AVAILABLE}"
fi

AP03_STARTED="$(now_seconds)"
echo
echo "=== AP03 targetless grouped COLMAP + marker-size-only scale ==="
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
  --moving-stride 1 --max-moving 0 \
  || AP03_STATUS="FAILED_PREPARE"

if [[ "$AP03_STATUS" == "OK" ]]; then
  bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh \
    || AP03_STATUS="FAILED_COLMAP"
fi
if [[ "$AP03_STATUS" == "OK" ]]; then
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py \
    || AP03_STATUS="FAILED_INSPECT"
fi
if [[ "$AP03_STATUS" == "OK" ]]; then
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/10_estimate_scale_from_marker_size_only.py \
    --marker-ids "$SIM_MARKER_IDS" \
    --marker-length-m 0.17 \
    --min-area-px2 100 \
    --reproj-thresh-px 5 \
    --ransac-iters 1000 \
    --min-inliers 4 \
    || AP03_STATUS="FAILED_SCALE"
fi
AP03_RUNTIME_SECONDS="$(elapsed_seconds "$AP03_STARTED")"

echo
echo "=== Partial-aware simulation evaluation ==="
rm -rf "$VAR_FINAL"
mkdir -p "$VAR_FINAL"
python3 run/bus_real_data/evaluation/12_eval_partial_static_camera_results.py \
  --final-root "$VAR_FINAL"

cp "$VAR_ROOT/VARIANT_METADATA.json" "$VAR_FINAL/VARIANT_METADATA.json" 2>/dev/null || true
cp "$VAR_ROOT/aruco_observations/SHARED_ARUCO_DETECTION_SUMMARY.txt" \
  "$VAR_FINAL/DIAGNOSTIC_SHARED_ARUCO_DETECTION_SUMMARY.txt" 2>/dev/null || true
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
AP01_RUNTIME_SECONDS=$AP01_RUNTIME_SECONDS
AP02_STATUS=$AP02_STATUS
AP02_RUNTIME_SECONDS=$AP02_RUNTIME_SECONDS
AP02_GT_FULL_MAP_STATUS=$AP02_GT_FULL_MAP_STATUS
AP03_STATUS=$AP03_STATUS
AP03_RUNTIME_SECONDS=$AP03_RUNTIME_SECONDS
SIM_MARKER_IDS=$SIM_MARKER_IDS
TXT

cat "$VAR_FINAL/RUN_STATUS.txt"

if [[ "${REFRESH_CANONICAL_FINAL:-1}" == "1" ]]; then
  bash run/bus_real_data/reporting/run_refresh_final_results.sh \
    --reuse-baseline --promote
else
  echo "[INFO] canonical final-result refresh disabled"
fi

echo "[OK] variant results: $VAR_FINAL"
