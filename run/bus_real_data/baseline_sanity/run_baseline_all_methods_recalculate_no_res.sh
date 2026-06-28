#!/usr/bin/env bash
set -u -o pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="run/bus_real_data:${PYTHONPATH:-}"
export AUTO_CONFIRM=1

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="results/bus_real_data/baseline_independent_recalc/${STAMP}"
LOG_ROOT="$RUN_ROOT/logs"
TMP_FINAL="$RUN_ROOT/99_FINAL_RESULTS_FOR_REPORT_RECALCULATED"
BACKUP_FINAL="results/bus_real_data/_backup_99_FINAL_RESULTS_FOR_REPORT_before_INDEPENDENT_RECALC_${STAMP}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$TMP_FINAL"

echo "===== BASELINE INDEPENDENT RECALC — NO RES SOURCES ====="
echo "[INFO] run root: $RUN_ROOT"

echo
echo "===== INPUT CHECK: SHARED BASELINE ONLY ====="
test -d results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
test -d results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations
echo "[OK] shared baseline raw_images + aruco_observations exist"

echo
echo "===== SCRIPT CHECK ====="
test -f run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh
test -f run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh
test -f run/bus_real_data/approach2_ref_marker_graph_ba/09_eval_ap02_gt_aligned_full_map.py
test -f run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
test -f run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py
echo "[OK] method scripts exist"

echo
echo "===== BACKUP CURRENT GLOBAL OUTPUTS ====="
mkdir -p "$RUN_ROOT/backup_before_run"
for p in \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/90_approach_comparison_ref_aruco \
  results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT
do
  if [ -e "$p" ]; then
    mkdir -p "$RUN_ROOT/backup_before_run/$(dirname "$p")"
    cp -a "$p" "$RUN_ROOT/backup_before_run/$p"
    echo "[BACKUP] $p"
  fi
done

echo
echo "===== CLEAN METHOD OUTPUTS ONLY ====="
rm -rf \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/90_approach_comparison_ref_aruco

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

run_step AP01_PIPELINE \
  bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh

run_step AP02_PIPELINE \
  bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh

run_step AP02_OFFICIAL_FULL_MAP_SE3_EVAL \
  python3 run/bus_real_data/approach2_ref_marker_graph_ba/09_eval_ap02_gt_aligned_full_map.py

echo
echo "############################################################"
echo "# RUN AP03 PIPELINE UNTIL COLMAP, ALLOW OLD REF14 FAIL"
echo "############################################################"
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh \
  2>&1 | tee "$LOG_ROOT/AP03_OLD_FULL_PIPELINE_ALLOWED_FAIL.log"
AP03_OLD_CODE=${PIPESTATUS[0]}
echo "[INFO] old AP03 full pipeline exit code: $AP03_OLD_CODE"

if [ ! -d results/bus_real_data/03_targetless_colmap_aruco_scale/02_colmap_sparse/sparse_txt/0 ]; then
  echo "[FAIL] AP03 COLMAP model missing after AP03 run"
  exit 1
fi
echo "[OK] AP03 COLMAP model exists"

AP03_ROOT="results/bus_real_data/03_targetless_colmap_aruco_scale"
AP03_LOG="$LOG_ROOT/AP03_SINGLE_MULTI_RECALC.log"
: > "$AP03_LOG"

echo
echo "===== AP03-SINGLE-REF14 THRESHOLD SEARCH =====" | tee -a "$AP03_LOG"
SINGLE_STATUS="FAILED"
SINGLE_MIN="-"

for MIN_AREA in 500 200 100 50 20 10 5 1; do
  echo | tee -a "$AP03_LOG"
  echo "----- AP03-SINGLE min_area_px2=$MIN_AREA -----" | tee -a "$AP03_LOG"

  rm -rf "$AP03_ROOT/06_triangulated_ref_aruco_registration"
  mkdir -p "$AP03_ROOT/06_triangulated_ref_aruco_registration" "$AP03_ROOT/07_final_results"

  if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06a_detect_ref14_scale_observations.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      --min-area-px2 "$MIN_AREA" \
      >> "$AP03_LOG" 2>&1 \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06b_triangulate_ref14_corners.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      --reproj-thresh-px 5 \
      --ransac-iters 1000 \
      --min-inliers 4 \
      >> "$AP03_LOG" 2>&1 \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06c_apply_ref14_registration_to_colmap.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      >> "$AP03_LOG" 2>&1 \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06d_eval_ap03_static_cameras.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      >> "$AP03_LOG" 2>&1 \
    && python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/06e_make_ap03_final_report.py \
      --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
      >> "$AP03_LOG" 2>&1
  then
    SRC="$AP03_ROOT/06_triangulated_ref_aruco_registration/ap03_static_cameras_ref_aruco_vs_gt.csv"
    DST="$AP03_ROOT/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv"
    if [ -f "$SRC" ]; then
      cp "$SRC" "$DST"
      SINGLE_STATUS="HAS_FOUR_CAMERA_CSV"
      SINGLE_MIN="$MIN_AREA"
      echo "[OK] AP03-SINGLE produced CSV with min_area_px2=$MIN_AREA" | tee -a "$AP03_LOG"
      break
    fi
  fi

  echo "[WARN] AP03-SINGLE failed for min_area_px2=$MIN_AREA" | tee -a "$AP03_LOG"
done

if [ "$SINGLE_STATUS" = "FAILED" ]; then
  echo "[WARN] AP03-SINGLE did not produce a final CSV"
fi

echo
echo "===== AP03-MULTI-ARUCO ====="
if python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
    --out-root "$AP03_ROOT/06_triangulated_ref_aruco_registration" \
    --final-root "$AP03_ROOT/07_final_results" \
    2>&1 | tee "$LOG_ROOT/AP03_MULTI_ARUCO_RECALC.log"
then
  echo "[OK] AP03-MULTI explicit call"
else
  echo "[WARN] explicit AP03-MULTI failed; trying default"
  python3 run/bus_real_data/approach3_targetless_colmap_aruco_scale/09_make_ap03_multi_aruco_final.py \
    2>&1 | tee "$LOG_ROOT/AP03_MULTI_ARUCO_RECALC_DEFAULT.log" || true
fi

echo
echo "===== SNAPSHOT GLOBAL RERUN OUTPUTS ====="
mkdir -p "$RUN_ROOT/global_outputs"
for p in \
  results/bus_real_data/01_marker_direct_relay_multimarker_multichain \
  results/bus_real_data/02_ref_marker_graph_ba \
  results/bus_real_data/03_targetless_colmap_aruco_scale \
  results/bus_real_data/90_approach_comparison_ref_aruco
do
  if [ -e "$p" ]; then
    cp -a "$p" "$RUN_ROOT/global_outputs/$(basename "$p")"
    echo "[SNAPSHOT] $p"
  fi
done

echo
echo "===== WRITE FINAL EVALUATOR: GLOBAL ONLY / NO RES ====="

cat > "$RUN_ROOT/eval_baseline_final_global_only_no_res.py" <<'PY'
from pathlib import Path
import csv, json, shutil, hashlib, math
from statistics import mean
import numpy as np
import sys

RUN_ROOT = Path(sys.argv[1])
TMP_FINAL = Path(sys.argv[2])

CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
SHORT = {"cam_edge_0":"cam0","cam_edge_1":"cam1","cam_edge_3":"cam3","cam_edge_5":"cam5"}

FORBIDDEN = "results/bus_real_data/ablation/moving_cam/res"

AP01_ROOT = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain")
AP02_ROOT = Path("results/bus_real_data/02_ref_marker_graph_ba")
AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")

def fail(msg):
    raise SystemExit("[FAIL] " + msg)

def forbid(path):
    s = str(path)
    if FORBIDDEN in s:
        fail("RES source forbidden: " + s)

def sha16(path):
    path = Path(path)
    if not path.exists():
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def read_csv(path):
    forbid(path)
    try:
        with Path(path).open(newline="", errors="replace") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def fmt(x):
    if x is None or x == "-":
        return "-"
    return f"{float(x):.3f}".rstrip("0").rstrip(".")

def classify(mean_t, mean_r, max_t, max_r):
    if mean_t > 100 or max_t > 300 or mean_r > 20 or max_r > 45:
        return "REJECT_UNSTABLE"
    if mean_t > 25 or max_t > 75 or mean_r > 2.0 or max_r > 5.0:
        return "UNSTABLE"
    return "OK"

def cam_key(r):
    return r.get("entity_id") or r.get("camera") or r.get("cam") or r.get("camera_id") or ""

def has_all_cams(rows):
    found = {cam_key(r) for r in rows if cam_key(r) in CAMS}
    return found == set(CAMS)

def get_val(r, names):
    for n in names:
        if n in r and str(r[n]).strip() not in {"", "-", "nan"}:
            return r[n]
    return None

def pos_triplet(r, prefix):
    if prefix == "est":
        opts = [
            ["est_ref14_x_m","est_ref14_y_m","est_ref14_z_m"],
            ["est_x_m","est_y_m","est_z_m"],
            ["estimated_x_m","estimated_y_m","estimated_z_m"],
            ["x_est_m","y_est_m","z_est_m"],
        ]
    else:
        opts = [
            ["gt_ref14_x_m","gt_ref14_y_m","gt_ref14_z_m"],
            ["gt_x_m","gt_y_m","gt_z_m"],
            ["ground_truth_x_m","ground_truth_y_m","ground_truth_z_m"],
            ["x_gt_m","y_gt_m","z_gt_m"],
        ]
    for keys in opts:
        if all(k in r and str(r[k]).strip() not in {"", "-"} for k in keys):
            return np.array([float(r[k]) for k in keys], dtype=float)
    return None

def euler_matrix(roll, pitch, yaw):
    r,p,y = map(lambda v: math.radians(float(v)), [roll,pitch,yaw])
    Rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
    Rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def quat_matrix(qw,qx,qy,qz):
    qw,qx,qy,qz = map(float, [qw,qx,qy,qz])
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
    qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy],
    ])

def rot_from_row(r, prefix):
    if prefix == "est":
        quat_sets = [
            ["est_qw","est_qx","est_qy","est_qz"],
            ["qw_est","qx_est","qy_est","qz_est"],
            ["est_ref14_qw","est_ref14_qx","est_ref14_qy","est_ref14_qz"],
        ]
        euler_sets = [
            ["est_ref14_roll_deg","est_ref14_pitch_deg","est_ref14_yaw_deg"],
            ["est_roll_deg","est_pitch_deg","est_yaw_deg"],
            ["roll_est_deg","pitch_est_deg","yaw_est_deg"],
        ]
    else:
        quat_sets = [
            ["gt_qw","gt_qx","gt_qy","gt_qz"],
            ["qw_gt","qx_gt","qy_gt","qz_gt"],
            ["gt_ref14_qw","gt_ref14_qx","gt_ref14_qy","gt_ref14_qz"],
        ]
        euler_sets = [
            ["gt_ref14_roll_deg","gt_ref14_pitch_deg","gt_ref14_yaw_deg"],
            ["gt_roll_deg","gt_pitch_deg","gt_yaw_deg"],
            ["roll_gt_deg","pitch_gt_deg","yaw_gt_deg"],
        ]
    for keys in quat_sets:
        if all(k in r and str(r[k]).strip() not in {"", "-"} for k in keys):
            return quat_matrix(*(r[k] for k in keys))
    for keys in euler_sets:
        if all(k in r and str(r[k]).strip() not in {"", "-"} for k in keys):
            return euler_matrix(*(r[k] for k in keys))
    return None

def se3_align(src_pts, dst_pts):
    src=np.asarray(src_pts,dtype=float); dst=np.asarray(dst_pts,dtype=float)
    sm=src.mean(axis=0); dm=dst.mean(axis=0)
    X=src-sm; Y=dst-dm
    U,S,Vt=np.linalg.svd(X.T@Y)
    R=Vt.T@U.T
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1
        R=Vt.T@U.T
    t=dm-R@sm
    return R,t,S

def rot_err(R_est, R_gt):
    R=R_gt.T@R_est
    v=(np.trace(R)-1)/2
    v=max(-1,min(1,v))
    return math.degrees(math.acos(v))

def all_csvs(root):
    return sorted(Path(root).glob("**/*.csv"))

def choose_ap01_source():
    candidates=[]
    for p in all_csvs(AP01_ROOT):
        rows=read_csv(p)
        if not has_all_cams(rows):
            continue
        usable=[]
        for r in rows:
            if cam_key(r) in CAMS and pos_triplet(r,"est") is not None and pos_triplet(r,"gt") is not None:
                usable.append(r)
        if len({cam_key(r) for r in usable}) == 4:
            rot_count=sum(1 for r in usable if rot_from_row(r,"est") is not None and rot_from_row(r,"gt") is not None)
            candidates.append((rot_count, len(str(p)), p))
    if not candidates:
        fail("AP01: no global CSV with four cams and est/gt position columns found")
    candidates.sort(key=lambda x:(-x[0], x[1]))
    return candidates[0][2]

def eval_ap01():
    p=choose_ap01_source()
    rows=read_csv(p)
    by={}
    for r in rows:
        c=cam_key(r)
        if c in CAMS and pos_triplet(r,"est") is not None and pos_triplet(r,"gt") is not None:
            by[c]=r
    est=[pos_triplet(by[c],"est") for c in CAMS]
    gt=[pos_triplet(by[c],"gt") for c in CAMS]
    R,t,S=se3_align(est,gt)
    out=[]
    missing_rot=False
    for c in CAMS:
        pe=R@pos_triplet(by[c],"est")+t
        pg=pos_triplet(by[c],"gt")
        te=np.linalg.norm(pe-pg)*100.0
        Re=rot_from_row(by[c],"est")
        Rg=rot_from_row(by[c],"gt")
        if Re is not None and Rg is not None:
            re=rot_err(R@Re,Rg)
        else:
            missing_rot=True
            raw=get_val(by[c],["rotation_error_deg","rot_error_deg","r_error_deg"])
            if raw is None:
                re=float("nan")
            else:
                re=float(raw)
        out.append({
            "method":"AP01",
            "status":"OK" if not missing_rot else "MISSING_ROTATION_SOURCE_USED_PRECOMPUTED",
            "entity_type":"camera",
            "entity_id":c,
            "translation_error_cm":fmt(te),
            "rotation_error_deg":fmt(re),
            "source":str(p),
            "source_sha16":sha16(p),
            "note":"global AP01 rerun; camera-map eval-only SE(3), no scale; rotation recomputed if pose rotations present",
        })
    return out, {"ap01_source":str(p), "singular_values":[float(x) for x in S], "missing_rotation_source":missing_rot}

def choose_ap02_source():
    priority=[]
    for p in all_csvs(AP02_ROOT) + all_csvs("results/bus_real_data/90_approach_comparison_ref_aruco"):
        s=str(p).lower()
        rows=read_csv(p)
        if not has_all_cams(rows):
            continue
        has_t=any("translation_error_cm" in r for r in rows)
        has_r=any(("rotation_error_deg" in r or "rot_error_deg" in r) for r in rows)
        if has_t and has_r:
            score=0
            if "full" in s and "map" in s: score+=100
            if "gt_aligned" in s or "aligned" in s: score+=50
            if "official" in s: score+=50
            if "08_final_results" in s: score+=20
            priority.append((score, len(s), p))
    if not priority:
        fail("AP02: no global official/full-map four-camera error CSV found")
    priority.sort(key=lambda x:(-x[0],x[1]))
    return priority[0][2]

def eval_error_csv(path, method, note, classify_errors):
    rows=read_csv(path)
    by={}
    for r in rows:
        c=cam_key(r)
        if c in CAMS:
            tval=get_val(r,["translation_error_cm","t_error_cm","err_t_cm","error_cm"])
            rval=get_val(r,["rotation_error_deg","rot_error_deg","r_error_deg","err_r_deg"])
            if tval is not None and rval is not None:
                by[c]=(float(tval),float(rval))
    if sorted(by) != sorted(CAMS):
        fail(f"{method}: expected four camera errors from {path}, got {sorted(by)}")
    ts=[by[c][0] for c in CAMS]; rs=[by[c][1] for c in CAMS]
    status=classify(mean(ts),mean(rs),max(ts),max(rs)) if classify_errors else "OK"
    return [{
        "method":method,
        "status":status,
        "entity_type":"camera",
        "entity_id":c,
        "translation_error_cm":fmt(by[c][0]),
        "rotation_error_deg":fmt(by[c][1]),
        "source":str(path),
        "source_sha16":sha16(path),
        "note":note,
    } for c in CAMS]

def summarize(rows):
    ts=[float(r["translation_error_cm"]) for r in rows if r["translation_error_cm"] != "-"]
    rs=[float(r["rotation_error_deg"]) for r in rows if r["rotation_error_deg"] != "-"]
    status=rows[0]["status"]
    row={
        "method":rows[0]["method"],
        "status":status,
        "mean_t_cm":fmt(mean(ts)) if len(ts)==4 else "-",
        "mean_r_deg":fmt(mean(rs)) if len(rs)==4 else "-",
        "max_t_cm":fmt(max(ts)) if len(ts)==4 else "-",
        "max_r_deg":fmt(max(rs)) if len(rs)==4 else "-",
        "source":rows[0]["source"],
        "source_sha16":rows[0]["source_sha16"],
        "note":rows[0]["note"],
    }
    by={r["entity_id"]:r for r in rows}
    for c in CAMS:
        row[SHORT[c]+"_t"]=by[c]["translation_error_cm"]
        row[SHORT[c]+"_r"]=by[c]["rotation_error_deg"]
    return row

def table(rows, fields):
    widths={f:max(len(f),*(len(str(r.get(f,""))) for r in rows)) for f in fields}
    return "\n".join(
        [" | ".join(f.ljust(widths[f]) for f in fields),
         "-+-".join("-"*widths[f] for f in fields)]
        + [" | ".join(str(r.get(f,"")).ljust(widths[f]) for f in fields) for r in rows]
    )

ap01, ap01_meta = eval_ap01()
ap02_src = choose_ap02_source()
ap02 = eval_error_csv(ap02_src, "AP02", "global AP02 rerun; official/full-map GT-aligned SE(3) preferred; no RES source", False)

ap03_single_src = AP03_ROOT / "07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv"
ap03_multi_src = AP03_ROOT / "07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv"

if ap03_single_src.exists():
    ap03_single = eval_error_csv(ap03_single_src, "AP03-SINGLE-REF14", "global AP03 rerun; Single Ref14 metric registration; GT eval-only", True)
else:
    ap03_single = [{
        "method":"AP03-SINGLE-REF14","status":"FAILED","entity_type":"camera","entity_id":c,
        "translation_error_cm":"-","rotation_error_deg":"-","source":str(ap03_single_src),
        "source_sha16":"-","note":"global AP03 rerun did not produce Single Ref14 final CSV"
    } for c in CAMS]

if ap03_multi_src.exists():
    ap03_multi = eval_error_csv(ap03_multi_src, "AP03-MULTI-ARUCO", "global AP03 rerun; Multi-ArUco metric registration; GT eval-only", True)
else:
    ap03_multi = [{
        "method":"AP03-MULTI-ARUCO","status":"FAILED","entity_type":"camera","entity_id":c,
        "translation_error_cm":"-","rotation_error_deg":"-","source":str(ap03_multi_src),
        "source_sha16":"-","note":"global AP03 rerun did not produce Multi-ArUco final CSV"
    } for c in CAMS]

all_methods = {
    "AP01": ap01,
    "AP02": ap02,
    "AP03-SINGLE-REF14": ap03_single,
    "AP03-MULTI-ARUCO": ap03_multi,
}
summaries=[summarize(all_methods[k]) for k in ["AP01","AP02","AP03-SINGLE-REF14","AP03-MULTI-ARUCO"]]

for r in summaries:
    forbid(r["source"])

fields_detail=["method","status","entity_type","entity_id","translation_error_cm","rotation_error_deg","source","source_sha16","note"]
fields_summary=["method","status","mean_t_cm","mean_r_deg","max_t_cm","max_r_deg","cam0_t","cam0_r","cam1_t","cam1_r","cam3_t","cam3_r","cam5_t","cam5_r","source","source_sha16","note"]

write_csv(TMP_FINAL/"AP01/AP01_FINAL_RESULT.csv", ap01, fields_detail)
write_csv(TMP_FINAL/"AP02/AP02_FINAL_RESULT.csv", ap02, fields_detail)
write_csv(TMP_FINAL/"AP03/AP03_FINAL_SINGLE_REF14_RESULT.csv", ap03_single, fields_detail)
write_csv(TMP_FINAL/"AP03/AP03_FINAL_MULTI_ARUCO_RESULT.csv", ap03_multi, fields_detail)
write_csv(TMP_FINAL/"AP03/AP03_FINAL_PER_CAMERA_RESULT.csv", ap03_single+ap03_multi, fields_detail)
write_csv(TMP_FINAL/"AP03/AP03_FINAL_RESULT.csv", [summaries[2], summaries[3]], fields_summary)
write_csv(TMP_FINAL/"BASELINE_FINAL_RECALCULATED_FROM_GLOBAL_RERUN.csv", summaries, fields_summary)

txt=[]
txt.append("BASELINE FINAL COMPARISON — INDEPENDENT GLOBAL RERUN")
txt.append("===================================================")
txt.append("")
txt.append("Source rule:")
txt.append("- NO values are copied from moving_cam/res.")
txt.append("- All sources are global baseline rerun outputs.")
txt.append("- AP01: camera-map evaluation-only SE(3), no scale, recomputed from global AP01 output.")
txt.append("- AP02: global official/full-map GT-aligned SE(3) evaluator output.")
txt.append("- AP03: global AP03 Single/Multi rerun outputs; unstable results stay unstable.")
txt.append("")
txt.append("Final comparison table:")
txt.append(table(summaries, ["method","status","mean_t_cm","mean_r_deg","max_t_cm","max_r_deg","cam0_t","cam0_r","cam1_t","cam1_r","cam3_t","cam3_r","cam5_t","cam5_r","note"]))
txt.append("")
txt.append("Sources:")
for r in summaries:
    txt.append(f"- {r['method']}: {r['source']} | sha16={r['source_sha16']}")
txt.append("")
txt.append("AP01 evaluator metadata:")
txt.append(json.dumps(ap01_meta, indent=2))
txt.append("")
txt.append("Interpretation:")
txt.append("- This is the honest independent baseline recalculation.")
txt.append("- If AP03 is REJECT_UNSTABLE here, then the current AP03 rerun did not reproduce the earlier accepted baseline.")
txt.append("- Do not replace those values with RES snapshot values in this mode.")

(TMP_FINAL/"BASELINE_FINAL_CLEAN_COMPARISON.txt").write_text("\n".join(txt)+"\n", encoding="utf-8")

manifest={
    "mode":"INDEPENDENT_GLOBAL_BASELINE_RERUN_NO_RES",
    "run_root":str(RUN_ROOT),
    "forbidden_source":FORBIDDEN,
    "ap01_meta":ap01_meta,
    "summaries":summaries,
}
(TMP_FINAL/"_RECALCULATED_FROM_GLOBAL_RERUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")

readme=[]
readme.append("99_FINAL_RESULTS_FOR_REPORT — INDEPENDENT GLOBAL BASELINE RERUN")
readme.append("==============================================================")
readme.append("")
readme.append("This folder was generated from global baseline rerun outputs only.")
readme.append("No results from moving_cam/res are accepted in this mode.")
readme.append("Use BASELINE_FINAL_CLEAN_COMPARISON.txt as the main report.")
(TMP_FINAL/"README.txt").write_text("\n".join(readme)+"\n", encoding="utf-8")

print("[OK] wrote tmp final:", TMP_FINAL)
print((TMP_FINAL/"BASELINE_FINAL_CLEAN_COMPARISON.txt").read_text())
PY

python3 "$RUN_ROOT/eval_baseline_final_global_only_no_res.py" "$RUN_ROOT" "$TMP_FINAL"

echo
echo "===== HARD GUARD: NO RES SOURCE PATHS ====="
if grep -RIn "results/bus_real_data/ablation/moving_cam/res" "$TMP_FINAL"; then
  echo "[FAIL] RES source leaked into independent recalculated report"
  exit 1
fi
echo "[OK] no RES source paths in recalculated report"

echo
echo "===== REPLACE 99_FINAL_RESULTS_FOR_REPORT WITH RECALCULATED GLOBAL REPORT ====="
if [ -d results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT ]; then
  cp -a results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT "$BACKUP_FINAL"
  echo "[BACKUP] old 99 final -> $BACKUP_FINAL"
fi

rm -rf results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT
cp -a "$TMP_FINAL" results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT

echo
echo "===== FINAL GLOBAL RECALCULATED REPORT ====="
cat results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt

echo
echo "===== FINAL TREE ====="
find results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT -maxdepth 3 -type f | sort

echo
echo "===== DONE ====="
echo "[OK] independent recalc run root: $RUN_ROOT"
echo "[OK] final report: results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/BASELINE_FINAL_CLEAN_COMPARISON.txt"
