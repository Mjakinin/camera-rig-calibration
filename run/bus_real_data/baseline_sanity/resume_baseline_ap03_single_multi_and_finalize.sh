#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export AUTO_CONFIRM=1

AP03_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
LATEST_RERUN_ROOT="$(ls -dt results/bus_real_data/baseline_standard_rerun/* 2>/dev/null | head -1 || true)"

if [ -z "$LATEST_RERUN_ROOT" ]; then
  LATEST_RERUN_ROOT="results/bus_real_data/baseline_standard_rerun/resume_$(date +%Y%m%d_%H%M%S)"
fi

LOG_ROOT="$LATEST_RERUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

echo "===== BASELINE AP03 RESUME ====="
echo "[INFO] rerun root: $LATEST_RERUN_ROOT"
echo "[INFO] log root:   $LOG_ROOT"

echo
echo "===== PRECHECK AP03 COLMAP OUTPUT ====="
if [ ! -d "$AP03_ROOT/02_colmap_sparse/sparse_txt/0" ]; then
  echo "[FAIL] missing COLMAP sparse_txt model: $AP03_ROOT/02_colmap_sparse/sparse_txt/0"
  echo "Need to rerun AP03 prepare/COLMAP first."
  exit 1
fi

if [ -f "$AP03_ROOT/03_reconstruction_inspection/ap03_colmap_inspection_report.txt" ]; then
  grep -E "registered static cameras|registered moving frames|static missing|COLMAP registered all static cameras" \
    "$AP03_ROOT/03_reconstruction_inspection/ap03_colmap_inspection_report.txt" || true
else
  echo "[WARN] missing inspection report; continuing because sparse_txt exists."
fi

promote_single_final() {
  local min_area="$1"

  python3 - "$min_area" <<'PY'
from pathlib import Path
import csv
import sys
from statistics import mean

min_area = sys.argv[1]

root = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
src = root / "06_triangulated_ref_aruco_registration" / "ap03_static_cameras_ref_aruco_vs_gt.csv"
final = root / "07_final_results"
final.mkdir(parents=True, exist_ok=True)

wanted = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

if not src.exists():
    raise SystemExit(f"[FAIL] missing AP03-SINGLE eval csv: {src}")

rows = []
with src.open(newline="", errors="replace") as f:
    for r in csv.DictReader(f):
        cam = r.get("entity_id", "")
        if cam in wanted:
            rows.append(r)

cams = sorted(r["entity_id"] for r in rows)
if cams != sorted(wanted):
    raise SystemExit(f"[FAIL] AP03-SINGLE expected four cameras {wanted}, got {cams}")

out_rows = []
for r in rows:
    out_rows.append({
        "approach": "AP03_targetless_colmap_single_ref14_metric_registration",
        "entity_type": "camera",
        "entity_id": r["entity_id"],
        "marker_id": "14",
        "translation_error_cm": r["translation_error_cm"],
        "rotation_error_deg": r["rotation_error_deg"],
        "dX_cm": r.get("delta_x_cm", r.get("dX_cm", "")),
        "dY_cm": r.get("delta_y_cm", r.get("dY_cm", "")),
        "dZ_cm": r.get("delta_z_cm", r.get("dZ_cm", "")),
        "est_ref14_x_m": r.get("est_ref_aruco_x_m", r.get("est_ref14_x_m", "")),
        "est_ref14_y_m": r.get("est_ref_aruco_y_m", r.get("est_ref14_y_m", "")),
        "est_ref14_z_m": r.get("est_ref_aruco_z_m", r.get("est_ref14_z_m", "")),
        "gt_ref14_x_m": r.get("gt_ref_aruco_x_m", r.get("gt_ref14_x_m", "")),
        "gt_ref14_y_m": r.get("gt_ref_aruco_y_m", r.get("gt_ref14_y_m", "")),
        "gt_ref14_z_m": r.get("gt_ref_aruco_z_m", r.get("gt_ref14_z_m", "")),
        "note": f"baseline standard rerun; targetless COLMAP + single Ref14 metric registration; min_area_px2={min_area}; GT evaluation-only",
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

txt = []
txt.append("AP03 FINAL SINGLE-REF14 RESULT — BASELINE STANDARD RERUN")
txt.append("========================================================")
txt.append("")
txt.append("Method: targetless COLMAP + single Ref14 metric registration")
txt.append("COLMAP targetless: yes")
txt.append("ArUco usage: post-reconstruction metric registration only")
txt.append("GT usage: final evaluation only")
txt.append(f"min_area_px2: {min_area}")
txt.append("")
txt.append(f"mean translation error [cm]: {mean(t_vals):.6f}")
txt.append(f"mean rotation error [deg]: {mean(r_vals):.6f}")
txt.append("")
txt.append("Per-camera:")
for r in out_rows:
    txt.append(f"- {r['entity_id']}: {float(r['translation_error_cm']):.6f} cm / {float(r['rotation_error_deg']):.6f} deg")
txt.append("")
txt.append(f"CSV: {csv_path}")

txt_path.write_text("\n".join(txt) + "\n", encoding="utf-8")

print(f"[OK] wrote {csv_path}")
print(f"[OK] wrote {txt_path}")
PY
}

validate_four_camera_csv() {
  local path="$1"
  local label="$2"

  python3 - "$path" "$label" <<'PY'
from pathlib import Path
import csv
import sys

path = Path(sys.argv[1])
label = sys.argv[2]
wanted = {"cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"}

if not path.exists():
    raise SystemExit(f"[FAIL] {label}: missing {path}")

found = set()
with path.open(newline="", errors="replace") as f:
    for r in csv.DictReader(f):
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in wanted:
            found.add(cam)

if found != wanted:
    raise SystemExit(f"[FAIL] {label}: expected {sorted(wanted)}, got {sorted(found)}")

print(f"[OK] {label}: four-camera CSV valid")
PY
}

echo
echo "===== AP03-SINGLE-REF14 THRESHOLD SEARCH ====="

single_status="FAILED"
single_min="-"

for min_area in 500 200 100 50 20 10 5 1; do
  echo
  echo "----- TRY AP03-SINGLE min_area_px2=$min_area -----"

  rm -rf "$AP03_ROOT/06_triangulated_ref_aruco_registration"
  mkdir -p "$AP03_ROOT/06_triangulated_ref_aruco_registration"

  if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06a_detect_ref14_scale_observations.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      --min-area-px2 "$min_area" \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06b_triangulate_ref14_corners.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      --reproj-thresh-px 5 \
      --ransac-iters 1000 \
      --min-inliers 4 \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06c_apply_ref14_registration_to_colmap.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06d_eval_ap03_static_cameras.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06e_make_ap03_final_report.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
    && promote_single_final "$min_area" \
    && validate_four_camera_csv "$AP03_ROOT/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv" "AP03-SINGLE-REF14"
  then
    single_status="OK"
    single_min="$min_area"
    echo "[OK] AP03-SINGLE succeeded with min_area_px2=$single_min"
    break
  else
    echo "[WARN] AP03-SINGLE failed with min_area_px2=$min_area"
  fi
done 2>&1 | tee "$LOG_ROOT/AP03_SINGLE_THRESHOLD_RESUME.log"

if [ "$single_status" != "OK" ]; then
  echo "[FAIL] AP03-SINGLE did not produce a valid four-camera final CSV."
  exit 1
fi

echo
echo "===== AP03-MULTI-ARUCO ====="
if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
    --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
    --final-root "$AP03_ROOT/07_final_results" \
    2>&1 | tee "$LOG_ROOT/AP03_MULTI_ARUCO_RESUME.log"
then
  echo "[OK] AP03-MULTI explicit call"
else
  echo "[WARN] explicit AP03-MULTI call failed; trying default"
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
    2>&1 | tee "$LOG_ROOT/AP03_MULTI_ARUCO_DEFAULT_RESUME.log"
fi

validate_four_camera_csv "$AP03_ROOT/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv" "AP03-MULTI-ARUCO"

echo
echo "===== AP03 COMBINED ====="
python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/12_make_ap03_final_combined_result.py \
  2>&1 | tee "$LOG_ROOT/AP03_COMBINED_RESUME.log"

echo
echo "===== SNAPSHOT AP03 RESUME OUTPUTS ====="
mkdir -p "$LATEST_RERUN_ROOT/global_outputs"
rm -rf "$LATEST_RERUN_ROOT/global_outputs/03_targetless_colmap_aruco_scale"
cp -a "$AP03_ROOT" "$LATEST_RERUN_ROOT/global_outputs/03_targetless_colmap_aruco_scale"
echo "[OK] snapshot: $LATEST_RERUN_ROOT/global_outputs/03_targetless_colmap_aruco_scale"

echo
echo "===== APPLY SOURCE-GATED 99 FINAL EVAL LAYER ====="
python3 run/bus_real_data/baseline_sanity/write_99_final_results_for_report.py \
  2>&1 | tee "$LOG_ROOT/WRITE_99_SOURCE_GATED_AFTER_AP03_RESUME.log"

echo
echo "===== BASELINE AP03 RESUME VALIDATION ====="
python3 - "$LATEST_RERUN_ROOT" "$single_min" <<'PY'
from pathlib import Path
import csv
import hashlib
import sys
from statistics import mean

rerun_root = Path(sys.argv[1])
single_min = sys.argv[2]

CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

files = {
    "AP03_SINGLE_RERUN": Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv"),
    "AP03_MULTI_RERUN": Path("results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv"),
    "FINAL_99_AP03_SPLIT": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP03/AP03_FINAL_RESULT.csv"),
    "FINAL_99_CLEAN": Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"),
}

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def sha16(path):
    if not path.exists():
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def metric(path):
    rows = read_csv(path)
    found = {}
    for r in rows:
        cam = r.get("entity_id") or r.get("camera") or r.get("cam") or ""
        if cam in CAMS:
            found[cam] = r
    if sorted(found) != sorted(CAMS):
        return "-", "-", ",".join(sorted(found)) if found else "-"
    ts, rs = [], []
    for cam in CAMS:
        r = found[cam]
        for tk in ["translation_error_cm", "t_cm", "error_cm"]:
            if tk in r and str(r[tk]).strip() not in {"", "-"}:
                ts.append(float(r[tk]))
                break
        for rk in ["rotation_error_deg", "r_deg", "rot_error_deg"]:
            if rk in r and str(r[rk]).strip() not in {"", "-"}:
                rs.append(float(r[rk]))
                break
    mt = f"{mean(ts):.3f}" if len(ts) == 4 else "-"
    mr = f"{mean(rs):.3f}" if len(rs) == 4 else "-"
    return mt, mr, ",".join(sorted(found))

lines = []
lines.append("BASELINE AP03 RESUME VALIDATION")
lines.append("===============================")
lines.append("")
lines.append(f"AP03-SINGLE threshold min_area_px2: {single_min}")
lines.append("")
lines.append("Files:")
for k, p in files.items():
    lines.append(f"- {k}: {p if p.exists() else 'MISSING'} | sha16={sha16(p)}")

lines.append("")
lines.append("Rerun AP03 metrics:")
for k in ["AP03_SINGLE_RERUN", "AP03_MULTI_RERUN"]:
    mt, mr, cams = metric(files[k])
    lines.append(f"- {k}: cams={cams} | mean_t_cm={mt} | mean_r_deg={mr}")

lines.append("")
lines.append("Accepted 99 final rows:")
if files["FINAL_99_CLEAN"].exists():
    for line in files["FINAL_99_CLEAN"].read_text(errors="replace").splitlines():
        if "baseline_1280x720 |" in line:
            lines.append(line)
else:
    lines.append("MISSING")

errors = []
for k in ["AP03_SINGLE_RERUN", "AP03_MULTI_RERUN"]:
    mt, mr, cams = metric(files[k])
    if set(cams.split(",")) != set(CAMS):
        errors.append(f"{k} missing cameras: {cams}")

lines.append("")
if errors:
    lines.append("VALIDATION: FAILED")
    lines.extend(f"- {e}" for e in errors)
else:
    lines.append("VALIDATION: PASSED")
    lines.append("AP03-SINGLE and AP03-MULTI rerun outputs contain all four cameras.")
    lines.append("99_FINAL_RESULTS_FOR_REPORT was regenerated through the source-gated final-eval layer.")

out = rerun_root / "BASELINE_AP03_RESUME_VALIDATION.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text())
PY

echo
echo "===== DONE AP03 RESUME ====="
echo "[OK] rerun root: $LATEST_RERUN_ROOT"
echo "[OK] final report: results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"
echo
cat results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt
