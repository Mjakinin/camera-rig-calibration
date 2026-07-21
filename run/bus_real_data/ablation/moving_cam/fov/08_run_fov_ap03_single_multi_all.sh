#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export AUTO_CONFIRM=1

VARIANTS=(
  "fov_40deg"
  "fov_50deg"
  "fov_60deg"
  "fov_69deg_baseline"
  "fov_80deg"
  "fov_90deg"
  "fov_100deg"
  "fov_110deg"
  "fov_120deg"
  "fov_140deg_extreme"
)

BASE_SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
BASE_RAW="$BASE_SHARED/raw_images"
BASE_OBS="$BASE_SHARED/aruco_observations"
BASE_META="$BASE_SHARED/metadata"

ABL_ROOT="results/bus_real_data/ablation/moving_cam/fov"
DATASET_ROOT="$ABL_ROOT/00_prepared_datasets"
OBS_ROOT="$ABL_ROOT/01_shared_observations"
CAPTURE_ROOT="$ABL_ROOT/00_captures"

AP03_RESULT="results/bus_real_data/03_targetless_colmap_aruco_scale"
AP03_OUT="$ABL_ROOT/04_ap03_results"
LOG_ROOT="$ABL_ROOT/_pipeline_logs/ap03_single_multi"
BACKUP_ROOT="$ABL_ROOT/_runtime_backup_ap03_single_multi"

mkdir -p "$AP03_OUT" "$LOG_ROOT" "$ABL_ROOT/99_summary"

echo "===== BACKUP SHARED BASELINE + GLOBAL AP03 RESULT ====="
rm -rf "$BACKUP_ROOT"
mkdir -p "$BACKUP_ROOT"

[ -d "$BASE_RAW" ] && cp -a "$BASE_RAW" "$BACKUP_ROOT/raw_images"
[ -d "$BASE_OBS" ] && cp -a "$BASE_OBS" "$BACKUP_ROOT/aruco_observations"
[ -d "$BASE_META" ] && cp -a "$BASE_META" "$BACKUP_ROOT/metadata"
[ -d "$AP03_RESULT" ] && cp -a "$AP03_RESULT" "$BACKUP_ROOT/03_targetless_colmap_aruco_scale"

restore_all() {
  set +e
  echo
  echo "===== RESTORE SHARED BASELINE + GLOBAL AP03 RESULT ====="

  if [ -d "$BACKUP_ROOT/raw_images" ]; then
    rm -rf "$BASE_RAW"
    mkdir -p "$BASE_SHARED"
    cp -a "$BACKUP_ROOT/raw_images" "$BASE_RAW"
  fi

  if [ -d "$BACKUP_ROOT/aruco_observations" ]; then
    rm -rf "$BASE_OBS"
    mkdir -p "$BASE_SHARED"
    cp -a "$BACKUP_ROOT/aruco_observations" "$BASE_OBS"
  fi

  if [ -d "$BACKUP_ROOT/metadata" ]; then
    rm -rf "$BASE_META"
    mkdir -p "$BASE_SHARED"
    cp -a "$BACKUP_ROOT/metadata" "$BASE_META"
  fi

  if [ -d "$BACKUP_ROOT/03_targetless_colmap_aruco_scale" ]; then
    rm -rf "$AP03_RESULT"
    mkdir -p "$(dirname "$AP03_RESULT")"
    cp -a "$BACKUP_ROOT/03_targetless_colmap_aruco_scale" "$AP03_RESULT"
  fi

  echo "[OK] restored baseline/global AP03."
}
trap restore_all EXIT

install_variant() {
  local variant="$1"
  local raw="$DATASET_ROOT/$variant/raw_images"
  local obs="$OBS_ROOT/$variant"
  local cap="$CAPTURE_ROOT/$variant"

  echo
  echo "===== INSTALL FOV VARIANT AS SHARED BASELINE: $variant ====="

  if [ ! -d "$raw" ]; then
    echo "[ERROR] missing raw dataset: $raw"
    return 2
  fi

  if [ ! -f "$obs/shared_all_aruco_observations.csv" ]; then
    echo "[ERROR] missing shared observations: $obs/shared_all_aruco_observations.csv"
    return 2
  fi

  rm -rf "$BASE_RAW" "$BASE_OBS"
  mkdir -p "$BASE_SHARED"

  cp -a "$raw" "$BASE_RAW"
  cp -a "$obs" "$BASE_OBS"

  mkdir -p "$BASE_META"
  if [ -f "$cap/route_commanded.csv" ]; then
    cp "$cap/route_commanded.csv" "$BASE_META/route_commanded.csv"
    mkdir -p "$BASE_RAW/ap1_metadata"
    cp "$cap/route_commanded.csv" "$BASE_RAW/ap1_metadata/route_commanded.csv"
  fi
}

promote_single_final() {
  local variant="$1"
  local min_area="$2"

  python3 - "$variant" "$min_area" <<'PY'
from pathlib import Path
import csv
import sys
from statistics import mean

variant = sys.argv[1]
min_area = sys.argv[2]

root = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
src = root / "06_triangulated_ref_aruco_registration" / "ap03_static_cameras_ref_aruco_vs_gt.csv"
final = root / "07_final_results"
final.mkdir(parents=True, exist_ok=True)

wanted = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

if not src.exists():
    raise SystemExit(f"[FAIL] missing eval csv: {src}")

rows = []
with src.open(newline="", errors="replace") as f:
    for r in csv.DictReader(f):
        cam = r.get("entity_id", "")
        if cam in wanted:
            rows.append(r)

cams = sorted(r["entity_id"] for r in rows)
if cams != sorted(wanted):
    raise SystemExit(f"[FAIL] AP03-SINGLE {variant}: expected 4 cameras {wanted}, got {cams}")

out_rows = []
for r in rows:
    out_rows.append({
        "approach": "AP03_targetless_colmap_single_ref14_scale",
        "entity_type": "camera",
        "entity_id": r["entity_id"],
        "marker_id": "14",
        "translation_error_cm": r["translation_error_cm"],
        "rotation_error_deg": r["rotation_error_deg"],
        "dX_cm": r.get("delta_x_cm", ""),
        "dY_cm": r.get("delta_y_cm", ""),
        "dZ_cm": r.get("delta_z_cm", ""),
        "est_ref14_x_m": r.get("est_ref_aruco_x_m", ""),
        "est_ref14_y_m": r.get("est_ref_aruco_y_m", ""),
        "est_ref14_z_m": r.get("est_ref_aruco_z_m", ""),
        "gt_ref14_x_m": r.get("gt_ref_aruco_x_m", ""),
        "gt_ref14_y_m": r.get("gt_ref_aruco_y_m", ""),
        "gt_ref14_z_m": r.get("gt_ref_aruco_z_m", ""),
        "note": f"variant={variant}; single Ref14 metric registration; min_area_px2={min_area}; GT evaluation-only",
    })

fields = [
    "approach", "entity_type", "entity_id", "marker_id",
    "translation_error_cm", "rotation_error_deg",
    "dX_cm", "dY_cm", "dZ_cm",
    "est_ref14_x_m", "est_ref14_y_m", "est_ref14_z_m",
    "gt_ref14_x_m", "gt_ref14_y_m", "gt_ref14_z_m",
    "note",
]

csv_path = final / "AP03_FINAL_SINGLE_REF14_RESULT.csv"
txt_path = final / "AP03_FINAL_SINGLE_REF14_RESULT.txt"

with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

t_vals = [float(r["translation_error_cm"]) for r in out_rows]
r_vals = [float(r["rotation_error_deg"]) for r in out_rows]

text = []
text.append("AP03 FINAL SINGLE-REF14 RESULT")
text.append("==============================")
text.append("")
text.append(f"Variant: {variant}")
text.append("Method: Targetless COLMAP + Single Ref14 metric registration")
text.append("COLMAP targetless: yes")
text.append("ArUco usage: after reconstruction only")
text.append("GT usage: final evaluation only")
text.append(f"min_area_px2: {min_area}")
text.append("")
text.append(f"mean translation error [cm]: {mean(t_vals):.6f}")
text.append(f"mean rotation error [deg]: {mean(r_vals):.6f}")
text.append("")
text.append("Per-camera:")
for r in out_rows:
    text.append(f"- {r['entity_id']}: {float(r['translation_error_cm']):.6f} cm / {float(r['rotation_error_deg']):.6f} deg")
text.append("")
text.append(f"CSV: {csv_path}")
txt_path.write_text("\n".join(text) + "\n")

print(f"[OK] wrote {csv_path}")
print(f"[OK] wrote {txt_path}")
PY
}

validate_multi_final() {
  local variant="$1"

  python3 - "$variant" <<'PY'
from pathlib import Path
import csv
import sys

variant = sys.argv[1]
p = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv")
wanted = {"cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"}

if not p.exists():
    raise SystemExit(f"[FAIL] AP03-MULTI {variant}: missing {p}")

found = set()
with p.open(newline="", errors="replace") as f:
    for r in csv.DictReader(f):
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in wanted:
            found.add(cam)

if found != wanted:
    raise SystemExit(f"[FAIL] AP03-MULTI {variant}: expected {sorted(wanted)}, got {sorted(found)}")

print(f"[OK] AP03-MULTI {variant}: four-camera final CSV exists")
PY
}

run_ap03_multi() {
  local variant="$1"
  local reg="$AP03_RESULT/06_triangulated_ref_aruco_registration"
  local final="$AP03_RESULT/07_final_results"

  echo
  echo "===== RUN AP03-MULTI-ARUCO: $variant ====="

  mkdir -p "$reg" "$final"

  # Prefer explicit args. If the local script does not accept them, fall back to defaults.
  if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
      --out-root "$reg" \
      --final-root "$final"
  then
    validate_multi_final "$variant"
    return $?
  fi

  echo "[WARN] explicit AP03-MULTI call failed; trying script defaults"
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py
  validate_multi_final "$variant"
}

run_one_variant() {
  local variant="$1"
  local log="$LOG_ROOT/${variant}_AP03_SINGLE_MULTI.log"

  {
    echo
    echo "############################################################"
    echo "# FOV AP03 SINGLE + MULTI: $variant"
    echo "############################################################"

    install_variant "$variant" || return 2

    echo
    echo "===== CLEAN GLOBAL AP03 WORKDIR ====="
    rm -rf "$AP03_RESULT"
    mkdir -p "$AP03_RESULT"

    echo
    echo "===== 1/5 PREPARE COLMAP DATASET ====="
    python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/01_prepare_colmap_dataset.py \
      --moving-stride 1 \
      --max-moving 0

    echo
    echo "===== 2/5 RUN COLMAP SPARSE ====="
    bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/02_run_colmap_sparse.sh

    echo
    echo "===== 3/5 INSPECT COLMAP ====="
    python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/03_inspect_colmap_reconstruction.py

    local single_status="FAILED"
    local single_min="-"

    echo
    echo "===== 4/5 AP03-SINGLE-REF14 THRESHOLD SEARCH ====="

    for min_area in 1000 500 200 100 50 20 10 5 1; do
      echo
      echo "----- TRY SINGLE REF14: min_area_px2=$min_area -----"

      rm -rf "$AP03_RESULT/06_triangulated_ref_aruco_registration"
      mkdir -p "$AP03_RESULT/06_triangulated_ref_aruco_registration"

      if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06a_detect_ref14_scale_observations.py \
          --out-root "$AP03_RESULT/06_triangulated_ref_aruco_registration" \
          --min-area-px2 "$min_area" \
        && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06b_triangulate_ref14_corners.py \
          --out-root "$AP03_RESULT/06_triangulated_ref_aruco_registration" \
          --reproj-thresh-px 5 \
          --ransac-iters 1000 \
          --min-inliers 4 \
        && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06c_apply_ref14_registration_to_colmap.py \
          --out-root "$AP03_RESULT/06_triangulated_ref_aruco_registration" \
        && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06d_eval_ap03_static_cameras.py \
          --out-root "$AP03_RESULT/06_triangulated_ref_aruco_registration" \
        && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06e_make_ap03_final_report.py \
          --out-root "$AP03_RESULT/06_triangulated_ref_aruco_registration" \
        && promote_single_final "$variant" "$min_area"
      then
        single_status="OK"
        single_min="$min_area"
        echo "[OK] AP03-SINGLE-REF14 succeeded: $variant min_area_px2=$single_min"
        break
      else
        echo "[WARN] AP03-SINGLE failed for min_area_px2=$min_area"
      fi
    done

    echo
    echo "===== 5/5 AP03-MULTI-ARUCO ====="
    local multi_status="FAILED"
    if run_ap03_multi "$variant"; then
      multi_status="OK"
      echo "[OK] AP03-MULTI-ARUCO succeeded: $variant"
    else
      echo "[WARN] AP03-MULTI-ARUCO failed or incomplete: $variant"
    fi

    local snap="$AP03_OUT/$variant"
    mkdir -p "$snap"
    rm -rf "$snap/03_targetless_colmap_aruco_scale"
    cp -a "$AP03_RESULT" "$snap/03_targetless_colmap_aruco_scale"

    cat > "$snap/AP03_SINGLE_MULTI_RERUN_STATUS.txt" <<TXT
FOV AP03 rerun status
=====================

variant: $variant

AP03-SINGLE-REF14:
  status: $single_status
  min_area_px2: $single_min
  source: $snap/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv

AP03-MULTI-ARUCO:
  status: $multi_status
  source: $snap/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv

Notes:
- COLMAP remains targetless.
- ArUco is used only after reconstruction for metric registration.
- GT is used only for final evaluation.
TXT

    echo "[OK] snapshot saved: $snap/03_targetless_colmap_aruco_scale"
    echo "[OK] status saved: $snap/AP03_SINGLE_MULTI_RERUN_STATUS.txt"

    if [ "$single_status" = "OK" ] || [ "$multi_status" = "OK" ]; then
      return 0
    fi

    return 10
  } 2>&1 | tee "$log"
}

overall=0

for variant in "${VARIANTS[@]}"; do
  run_one_variant "$variant" || overall=1
done

echo
echo "===== FOV AP03 SINGLE/MULTI SUMMARY ====="

for variant in "${VARIANTS[@]}"; do
  echo
  echo "----- $variant -----"
  status="$AP03_OUT/$variant/AP03_SINGLE_MULTI_RERUN_STATUS.txt"
  if [ -f "$status" ]; then
    cat "$status"
  else
    echo "missing status file"
  fi
done

echo
echo "===== FOV AP03 FINAL FILES ====="
find "$AP03_OUT" -type f \( \
  -name "AP03_FINAL_SINGLE_REF14_RESULT.csv" -o \
  -name "AP03_FINAL_MULTI_ARUCO_RESULT.csv" -o \
  -name "AP03_SINGLE_MULTI_RERUN_STATUS.txt" \
\) | sort

cat > "$ABL_ROOT/99_summary/FOV_AP03_SINGLE_MULTI_RERUN_STATUS.txt" <<TXT
MOVING-CAMERA FOV AP03 SINGLE/MULTI RERUN STATUS
================================================

Rerun attempted for:
${VARIANTS[*]}

See per-variant status files under:
$AP03_OUT/<variant>/AP03_SINGLE_MULTI_RERUN_STATUS.txt

This script does not modify the RES ablation.
This script restores the shared baseline and global AP03 workdir at exit.
TXT

exit "$overall"
