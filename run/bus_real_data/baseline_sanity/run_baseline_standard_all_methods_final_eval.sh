#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export AUTO_CONFIRM=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RERUN_ROOT="results/bus_real_data/baseline_standard_rerun/${STAMP}"
LOG_ROOT="$RERUN_ROOT/logs"
BACKUP_ROOT="$RERUN_ROOT/backup_before_rerun"

BASE_SHARED="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1"
BASE_RAW="$BASE_SHARED/raw_images"
BASE_OBS="$BASE_SHARED/aruco_observations"
BASE_META="$BASE_SHARED/metadata"

RES_BASE="results/bus_real_data/ablation/moving_cam/res"
RES_VAR="res_1280x720_baseline"

mkdir -p "$RERUN_ROOT" "$LOG_ROOT" "$BACKUP_ROOT"

echo "===== WAIT IF FOV AP03 RUNNER IS ACTIVE ====="
while pgrep -af "08_run_fov_ap03_single_multi_all.sh" | grep -v grep >/dev/null 2>&1; do
  echo "[WAIT] FOV AP03 still active. Waiting 60s..."
  sleep 60
done
echo "[OK] no FOV AP03 wrapper detected"

echo
echo "===== BACKUP CURRENT GLOBAL STATE ====="
for p in \
  "$BASE_RAW" \
  "$BASE_OBS" \
  "$BASE_META" \
  "results/bus_real_data/01_marker_direct_relay_multimarker_multichain" \
  "results/bus_real_data/02_ref_marker_graph_ba" \
  "results/bus_real_data/03_targetless_colmap_aruco_scale" \
  "results/bus_real_data/90_approach_comparison_ref_aruco" \
  "results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT"
do
  if [ -e "$p" ]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "$p")"
    cp -a "$p" "$BACKUP_ROOT/$p"
    echo "[BACKUP] $p"
  fi
done

echo
echo "===== INSTALL VALIDATED BASELINE INPUT ====="
test -d "$RES_BASE/00_prepared_datasets/$RES_VAR/raw_images"
test -f "$RES_BASE/01_shared_observations/$RES_VAR/shared_all_aruco_observations.csv"

rm -rf "$BASE_RAW" "$BASE_OBS"
mkdir -p "$BASE_SHARED"

cp -a "$RES_BASE/00_prepared_datasets/$RES_VAR/raw_images" "$BASE_RAW"
cp -a "$RES_BASE/01_shared_observations/$RES_VAR" "$BASE_OBS"

mkdir -p "$BASE_META"
if [ -f "$RES_BASE/00_captures/$RES_VAR/route_commanded.csv" ]; then
  cp "$RES_BASE/00_captures/$RES_VAR/route_commanded.csv" "$BASE_META/route_commanded.csv"
  mkdir -p "$BASE_RAW/ap1_metadata"
  cp "$RES_BASE/00_captures/$RES_VAR/route_commanded.csv" "$BASE_RAW/ap1_metadata/route_commanded.csv"
fi

echo "[OK] baseline input installed from $RES_VAR"

echo
echo "===== CLEAN GLOBAL METHOD OUTPUTS ====="
rm -rf \
  "results/bus_real_data/01_marker_direct_relay_multimarker_multichain" \
  "results/bus_real_data/02_ref_marker_graph_ba" \
  "results/bus_real_data/03_targetless_colmap_aruco_scale" \
  "results/bus_real_data/90_approach_comparison_ref_aruco"

echo
echo "===== SCRIPT CHECK ====="
test -f run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh
test -f run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh
test -f run/bus_real_data/approach2_ref_marker_graph_ba/09_eval_ap02_gt_aligned_full_map.py
test -f run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
test -f run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py
test -f run/bus_real_data/approach3_targetless_colmap_aruco_scale/12_make_ap03_final_combined_result.py
test -f run/bus_real_data/baseline_sanity/write_99_final_results_for_report.py
echo "[OK] required scripts exist"

run_step() {
  local name="$1"
  shift
  echo
  echo "############################################################"
  echo "# RUN $name"
  echo "############################################################"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  local code=${PIPESTATUS[0]}
  if [ "$code" -ne 0 ]; then
    echo "[FAIL] $name exited with code $code"
    exit "$code"
  fi
  echo "[OK] $name"
}

run_step "AP01_PIPELINE" \
  bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh

run_step "AP02_PIPELINE" \
  bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh

run_step "AP02_OFFICIAL_FULL_MAP_EVAL" \
  python3 run/bus_real_data/approach2_ref_marker_graph_ba/09_eval_ap02_gt_aligned_full_map.py

run_step "AP03_SINGLE_FULL_PIPELINE" \
  bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh

echo
echo "############################################################"
echo "# RUN AP03_MULTI_ARUCO"
echo "############################################################"
if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
    --out-root results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration \
    --final-root results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results \
    2>&1 | tee "$LOG_ROOT/AP03_MULTI_ARUCO.log"
then
  echo "[OK] AP03_MULTI_ARUCO explicit call"
else
  echo "[WARN] AP03_MULTI_ARUCO explicit call failed; trying default call"
  run_step "AP03_MULTI_ARUCO_DEFAULT" \
    python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py
fi

run_step "AP03_COMBINED" \
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/12_make_ap03_final_combined_result.py

echo
echo "===== SNAPSHOT RERUN OUTPUTS ====="
mkdir -p "$RERUN_ROOT/global_outputs"

for p in \
  "results/bus_real_data/01_marker_direct_relay_multimarker_multichain" \
  "results/bus_real_data/02_ref_marker_graph_ba" \
  "results/bus_real_data/03_targetless_colmap_aruco_scale" \
  "results/bus_real_data/90_approach_comparison_ref_aruco"
do
  if [ -e "$p" ]; then
    cp -a "$p" "$RERUN_ROOT/global_outputs/$(basename "$p")"
    echo "[SNAPSHOT] $p"
  fi
done

echo
echo "===== APPLY SOURCE-GATED FINAL EVAL LAYER ====="
python3 run/bus_real_data/baseline_sanity/write_99_final_results_for_report.py \
  2>&1 | tee "$LOG_ROOT/WRITE_99_SOURCE_GATED_FINAL.log"

echo
echo "===== BASELINE RERUN VALIDATION REPORT ====="
python3 - "$RERUN_ROOT" <<'PY'
from pathlib import Path
import csv
import hashlib
import json
import sys
from statistics import mean

rerun_root = Path(sys.argv[1])
out = rerun_root / "BASELINE_STANDARD_RERUN_VALIDATION.txt"

CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def sha16(path):
    p = Path(path)
    if not p.exists():
        return "-"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def cam_rows(path):
    rows = read_csv(path)
    found = {}
    for r in rows:
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in CAMS:
            found[cam] = r
    return found

def metrics(path):
    rows = cam_rows(path)
    if sorted(rows) != sorted(CAMS):
        return "-", "-", ",".join(sorted(rows)) if rows else "-"
    t_keys = ["translation_error_cm", "cam_gt_vs_est_error_cm", "error_cm", "t_cm"]
    r_keys = ["rotation_error_deg", "rot_error_deg", "r_deg"]
    ts, rs = [], []
    for cam in CAMS:
        r = rows[cam]
        tv = rv = None
        for k in t_keys:
            if k in r and str(r[k]).strip() not in {"", "-"}:
                try:
                    tv = float(r[k])
                    break
                except Exception:
                    pass
        for k in r_keys:
            if k in r and str(r[k]).strip() not in {"", "-"}:
                try:
                    rv = float(r[k])
                    break
                except Exception:
                    pass
        if tv is not None:
            ts.append(tv)
        if rv is not None:
            rs.append(rv)
    mt = f"{mean(ts):.3f}" if len(ts) == 4 else "-"
    mr = f"{mean(rs):.3f}" if len(rs) == 4 else "-"
    return mt, mr, ",".join(sorted(rows))

files = {
    "rerun_AP01_legacy_output": Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/AP01_FINAL_RESULT.csv"),
    "rerun_AP02_official_full_map": Path("results/bus_real_data/02_ref_marker_graph_ba/08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv"),
    "rerun_AP03_single": Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv"),
    "rerun_AP03_multi": Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv"),
    "final_99_AP01": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP01/AP01_FINAL_RESULT.csv"),
    "final_99_AP02": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP02/AP02_FINAL_RESULT.csv"),
    "final_99_AP03_split": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP03/AP03_FINAL_RESULT.csv"),
    "final_99_clean": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"),
}

lines = []
lines.append("BASELINE STANDARD ALL-METHODS RERUN VALIDATION")
lines.append("==============================================")
lines.append("")
lines.append("Input:")
lines.append("- validated res_1280x720_baseline raw_images + shared ArUco observations")
lines.append("")
lines.append("Important policy:")
lines.append("- AP01 rerun pipeline is checked as a method run, but its legacy final CSV is not the accepted final metric.")
lines.append("- Accepted AP01 final metric is the source-gated camera-map SE(3) evaluation-only row.")
lines.append("- AP02 accepted metric is the official/full-map GT-aligned SE(3) evaluation.")
lines.append("- AP03 accepted metrics are split into AP03-SINGLE-REF14 and AP03-MULTI-ARUCO.")
lines.append("- GT camera poses are used only for final evaluation.")
lines.append("")
lines.append("Detected files:")
for name, path in files.items():
    lines.append(f"- {name}: {path if path.exists() else 'MISSING'} | sha16={sha16(path)}")

lines.append("")
lines.append("Parsable four-camera summaries:")
for name, path in files.items():
    if path.suffix.lower() == ".csv":
        mt, mr, cams = metrics(path)
        lines.append(f"- {name}: cams={cams} | mean_t_cm={mt} | mean_r_deg={mr}")

lines.append("")
lines.append("Final accepted baseline comparison:")
if files["final_99_clean"].exists():
    text = files["final_99_clean"].read_text(errors="replace")
    for line in text.splitlines():
        if "baseline_1280x720 |" in line:
            lines.append(line)
else:
    lines.append("MISSING")

# hard gate: source-gated 99 final files must exist and contain 4 cameras where relevant
errors = []
for key in ["final_99_AP01", "final_99_AP02"]:
    mt, mr, cams = metrics(files[key])
    if set(cams.split(",")) != set(CAMS):
        errors.append(f"{key} missing four cameras: {cams}")
for key in ["rerun_AP03_single", "rerun_AP03_multi"]:
    mt, mr, cams = metrics(files[key])
    if set(cams.split(",")) != set(CAMS):
        errors.append(f"{key} missing four cameras: {cams}")

lines.append("")
if errors:
    lines.append("VALIDATION: FAILED")
    lines.extend(f"- {e}" for e in errors)
else:
    lines.append("VALIDATION: PASSED")
    lines.append("All required final/source-gated outputs exist and contain the four-camera rig.")

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text())
PY

cp results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt \
  "$RERUN_ROOT/BASELINE_FINAL_CLEAN_COMPARISON.txt"

cp results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/_SOURCE_GATING_MANIFEST.json \
  "$RERUN_ROOT/SOURCE_GATING_MANIFEST.json"

echo
echo "===== DONE ====="
echo "[OK] rerun root: $RERUN_ROOT"
echo "[OK] logs:       $LOG_ROOT"
echo "[OK] final:      results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"
echo
echo "===== FINAL REPORT ====="
cat results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt
