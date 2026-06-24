#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import statistics
import math

ROOT = Path("results/bus_real_data")

AP01_DIR = ROOT / "01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference"
AP02_DIR = ROOT / "02_ref_marker_graph_ba/08_final_results"
AP03_DIR = ROOT / "03_targetless_colmap_aruco_scale/07_final_results"
COMP_DIR = ROOT / "90_approach_comparison_ref_aruco"

AP02_SRC_DIR = ROOT / "90_approach_comparison_ref_aruco/02_ref_marker_graph_ba"
AP03_SRC_DIR = ROOT / "03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration"

AP02_DIR.mkdir(parents=True, exist_ok=True)
AP03_DIR.mkdir(parents=True, exist_ok=True)
COMP_DIR.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def fnum(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return ""

def get(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return ""

def entity_name(row):
    return get(row, "entity_id", "camera", "marker", "name", "observer_id")

def marker_id(row):
    mid = get(row, "marker_id", "id")
    if mid != "":
        try:
            return str(int(float(mid)))
        except Exception:
            return str(mid)
    name = entity_name(row)
    m = re.search(r"(\d+)", str(name))
    if m:
        return str(int(m.group(1)))
    return ""

def is_ref14(row):
    name = str(entity_name(row))
    mid = marker_id(row)
    return mid == "14" or "ref_floor_14" in name or name.endswith("_14")

def find_pose_csv(root, required_kind):
    root = Path(root)
    candidates = sorted(root.rglob("*.csv"))
    scored = []
    for p in candidates:
        rows = read_csv(p)
        if not rows:
            continue
        cols = set(rows[0].keys())
        needed = {
            "est_ref_aruco_x_m", "est_ref_aruco_y_m", "est_ref_aruco_z_m",
            "gt_ref_aruco_x_m", "gt_ref_aruco_y_m", "gt_ref_aruco_z_m",
        }
        if not needed.issubset(cols):
            continue
        name = p.name.lower()
        score = 0
        if required_kind == "camera" and ("camera" in name or "static" in name):
            score += 10
        if required_kind == "marker" and "marker" in name:
            score += 10
        if "vs_gt" in name:
            score += 5
        if "ref_aruco" in name:
            score += 3
        scored.append((score, p))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None

def xyz_error_cm(row):
    try:
        ex = float(row["est_ref_aruco_x_m"])
        ey = float(row["est_ref_aruco_y_m"])
        ez = float(row["est_ref_aruco_z_m"])
        gx = float(row["gt_ref_aruco_x_m"])
        gy = float(row["gt_ref_aruco_y_m"])
        gz = float(row["gt_ref_aruco_z_m"])
        dx = 100.0 * (ex - gx)
        dy = 100.0 * (ey - gy)
        dz = 100.0 * (ez - gz)
        terr = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dx, dy, dz, terr
    except Exception:
        return None, None, None, None

def normalize_camera_rows(rows, approach):
    out = []
    for r in rows:
        name = entity_name(r)
        if not str(name).startswith("cam_edge_"):
            continue
        dx, dy, dz, terr_calc = xyz_error_cm(r)
        terr_src = get(r, "translation_error_cm", "t_err_cm")
        out.append({
            "approach": approach,
            "entity_type": "camera",
            "entity_id": name,
            "marker_id": "",
            "translation_error_cm": fnum(terr_src if terr_src != "" else terr_calc),
            "rotation_error_deg": fnum(get(r, "rotation_error_deg", "r_err_deg")),
            "dX_cm": fnum(dx),
            "dY_cm": fnum(dy),
            "dZ_cm": fnum(dz),
            "est_ref14_x_m": fnum(get(r, "est_ref_aruco_x_m"), 6),
            "est_ref14_y_m": fnum(get(r, "est_ref_aruco_y_m"), 6),
            "est_ref14_z_m": fnum(get(r, "est_ref_aruco_z_m"), 6),
            "gt_ref14_x_m": fnum(get(r, "gt_ref_aruco_x_m"), 6),
            "gt_ref14_y_m": fnum(get(r, "gt_ref_aruco_y_m"), 6),
            "gt_ref14_z_m": fnum(get(r, "gt_ref_aruco_z_m"), 6),
            "note": "T_ref14_camera_est compared to T_ref14_camera_gt",
        })
    return out

def normalize_marker_rows(rows, approach):
    out = []
    for r in rows:
        # Critical: do NOT output Ref14 as zero-error entity.
        # In a Ref14-relative report, T_ref14_ref14 is identity by definition.
        if is_ref14(r):
            continue

        name = entity_name(r)
        mid = marker_id(r)
        dx, dy, dz, terr_calc = xyz_error_cm(r)
        terr_src = get(r, "translation_error_cm", "t_err_cm")
        out.append({
            "approach": approach,
            "entity_type": "marker",
            "entity_id": name,
            "marker_id": mid,
            "translation_error_cm": fnum(terr_src if terr_src != "" else terr_calc),
            "rotation_error_deg": fnum(get(r, "rotation_error_deg", "r_err_deg")),
            "dX_cm": fnum(dx),
            "dY_cm": fnum(dy),
            "dZ_cm": fnum(dz),
            "est_ref14_x_m": fnum(get(r, "est_ref_aruco_x_m"), 6),
            "est_ref14_y_m": fnum(get(r, "est_ref_aruco_y_m"), 6),
            "est_ref14_z_m": fnum(get(r, "est_ref_aruco_z_m"), 6),
            "gt_ref14_x_m": fnum(get(r, "gt_ref_aruco_x_m"), 6),
            "gt_ref14_y_m": fnum(get(r, "gt_ref_aruco_y_m"), 6),
            "gt_ref14_z_m": fnum(get(r, "gt_ref_aruco_z_m"), 6),
            "note": "T_ref14_marker_est compared to T_ref14_marker_gt",
        })
    return out

def table(rows, fields):
    if not rows:
        return "No rows available."
    widths = [max(len(f), *(len(str(r.get(f, ""))) for r in rows)) for f in fields]
    lines = []
    lines.append(" | ".join(f.ljust(w) for f, w in zip(fields, widths)))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(" | ".join(str(r.get(f, "")).ljust(w) for f, w in zip(fields, widths)))
    return "\n".join(lines)

def md_table(rows, fields):
    if not rows:
        return "_No rows available._"
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    return "\n".join(lines)

def stats(rows, entity_type):
    vals_t = []
    vals_r = []
    for r in rows:
        if r["entity_type"] != entity_type:
            continue
        if r["translation_error_cm"] != "":
            vals_t.append(float(r["translation_error_cm"]))
        if r["rotation_error_deg"] != "":
            vals_r.append(float(r["rotation_error_deg"]))
    if not vals_t:
        return "count: 0"
    return (
        f"count: {len(vals_t)}, "
        f"mean_t_cm: {statistics.mean(vals_t):.3f}, "
        f"median_t_cm: {statistics.median(vals_t):.3f}, "
        f"mean_r_deg: {statistics.mean(vals_r):.3f}, "
        f"median_r_deg: {statistics.median(vals_r):.3f}"
    )

FIELDS = [
    "approach", "entity_type", "entity_id", "marker_id",
    "translation_error_cm", "rotation_error_deg",
    "dX_cm", "dY_cm", "dZ_cm",
    "est_ref14_x_m", "est_ref14_y_m", "est_ref14_z_m",
    "gt_ref14_x_m", "gt_ref14_y_m", "gt_ref14_z_m",
    "note",
]

SHOW_FIELDS = [
    "entity_type", "entity_id", "marker_id",
    "translation_error_cm", "rotation_error_deg",
    "dX_cm", "dY_cm", "dZ_cm",
    "est_ref14_x_m", "est_ref14_y_m", "est_ref14_z_m",
    "gt_ref14_x_m", "gt_ref14_y_m", "gt_ref14_z_m",
    "note",
]

# AP01
ap01_csv = AP01_DIR / "final_camera_poses_ref14_gt_eval.csv"
ap01_rows = normalize_camera_rows(read_csv(ap01_csv), "AP01_marker_direct_relay") if ap01_csv.exists() else []
write_csv(AP01_DIR / "AP01_FINAL_REF14_RELATIVE_EVALUATION.csv", ap01_rows, FIELDS)
(AP01_DIR / "AP01_FINAL_REF14_RELATIVE_EVALUATION.txt").write_text(f"""AP01 FINAL REF14-RELATIVE EVALUATION
====================================

Definition:
- Estimated relative pose: T_ref14_camera_est
- GT relative pose:        T_ref14_camera_gt = inverse(T_world_ref14_gt) * T_world_camera_gt
- Error:                  estimated relative pose vs GT relative pose

Scope:
- AP01 currently exports camera-rig results.
- It does not export a full estimated marker map for all markers in the final result folder.
- Therefore this clean report contains static cameras only.

Camera summary:
- {stats(ap01_rows, "camera")}

Camera poses relative to Ref14:
{table(ap01_rows, SHOW_FIELDS)}
""")

# AP02
ap02_cam_csv = find_pose_csv(AP02_SRC_DIR, "camera") or find_pose_csv(ROOT / "02_ref_marker_graph_ba", "camera")
ap02_marker_csv = find_pose_csv(AP02_SRC_DIR, "marker") or find_pose_csv(ROOT / "02_ref_marker_graph_ba", "marker")

if ap02_cam_csv is None:
    raise RuntimeError("Could not find AP02 camera-vs-GT CSV with est_ref_aruco/gt_ref_aruco columns.")
if ap02_marker_csv is None:
    raise RuntimeError("Could not find AP02 marker-vs-GT CSV with est_ref_aruco/gt_ref_aruco columns.")

ap02_rows = []
ap02_rows.extend(normalize_camera_rows(read_csv(ap02_cam_csv), "AP02_ref_marker_graph_ba"))
ap02_rows.extend(normalize_marker_rows(read_csv(ap02_marker_csv), "AP02_ref_marker_graph_ba"))

write_csv(AP02_DIR / "AP02_FINAL_REF14_RELATIVE_EVALUATION.csv", ap02_rows, FIELDS)

ap02_txt = f"""AP02 FINAL REF14-RELATIVE CAMERA AND MARKER EVALUATION
=====================================================

Definition:
- Estimated relative pose: T_ref14_entity_est
- GT relative pose:        T_ref14_entity_gt = inverse(T_world_ref14_gt) * T_world_entity_gt
- Error:                  estimated relative pose vs GT relative pose

Method note:
- AP02 is a reference-marker graph optimization.
- Ref14 simulation GT is NOT used during optimization.
- Ref14 defines the output coordinate frame.
- Therefore Ref14 is NOT listed as an entity-error row.
- Reporting Ref14 as 0.000 error would be misleading, because T_ref14_ref14 is identity by definition.
- Cameras and markers 0-13 are evaluated relative to Ref14.

Input files:
- cameras: {ap02_cam_csv}
- markers: {ap02_marker_csv}

Summary:
- cameras: {stats(ap02_rows, "camera")}
- markers excluding Ref14: {stats(ap02_rows, "marker")}

All AP02 evaluated entities relative to Ref14:
{table(ap02_rows, SHOW_FIELDS)}
"""

ap02_md = f"""# AP02 Final Ref14-Relative Camera and Marker Evaluation

## Definition

- Estimated relative pose: `T_ref14_entity_est`
- GT relative pose: `T_ref14_entity_gt = inverse(T_world_ref14_gt) * T_world_entity_gt`
- Error: estimated relative pose vs GT relative pose

## Method note

AP02 is a reference-marker graph optimization. Ref14 simulation GT is not used during optimization. Ref14 defines the output coordinate frame. Therefore Ref14 is not listed as an entity-error row. Reporting Ref14 as `0.000` error would be misleading, because `T_ref14_ref14` is identity by definition. Cameras and markers 0-13 are evaluated relative to Ref14.

## Summary

- Cameras: {stats(ap02_rows, "camera")}
- Markers excluding Ref14: {stats(ap02_rows, "marker")}

## All AP02 evaluated entities relative to Ref14

{md_table(ap02_rows, SHOW_FIELDS)}
"""

for p in [
    AP02_DIR / "AP02_FINAL_REF14_RELATIVE_EVALUATION.txt",
    AP02_DIR / "AP02_FINAL_REPORT.txt",
    AP02_DIR / "AP02_FINAL_CLEAN_REPORT.txt",
    AP02_DIR / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.txt",
    AP02_SRC_DIR / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.txt",
]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ap02_txt)

for p in [
    AP02_DIR / "AP02_FINAL_REF14_RELATIVE_EVALUATION.md",
    AP02_DIR / "AP02_FINAL_REPORT.md",
    AP02_DIR / "AP02_FINAL_CLEAN_REPORT.md",
    AP02_DIR / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.md",
    AP02_SRC_DIR / "AP02_FINAL_READABLE_REF_ARUCO_REPORT.md",
]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ap02_md)

# AP03
ap03_cam_csv = find_pose_csv(AP03_SRC_DIR, "camera") or find_pose_csv(ROOT / "90_approach_comparison_ref_aruco/03_targetless_colmap_aruco_scale", "camera")
ap03_rows = normalize_camera_rows(read_csv(ap03_cam_csv), "AP03_targetless_colmap_aruco_scale") if ap03_cam_csv else []

write_csv(AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.csv", ap03_rows, FIELDS)

ap03_existing = AP03_SRC_DIR / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt"
ap03_extra = ap03_existing.read_text() if ap03_existing.exists() else ""

ap03_txt = f"""AP03 FINAL REF14-RELATIVE CAMERA EVALUATION
===========================================

Definition:
- Estimated relative pose: T_ref14_camera_est
- GT relative pose:        T_ref14_camera_gt = inverse(T_world_ref14_gt) * T_world_camera_gt
- Error:                  estimated relative pose vs GT relative pose

Method note:
- AP03 uses targetless COLMAP/SfM to reconstruct cameras.
- Ref14 is used after COLMAP for metric Sim3 scale/frame registration.
- AP03 currently does not export a full estimated marker map for markers 0-13.
- Therefore this clean report contains static cameras and Ref14 corner-quality information from the original AP03 report.

Input file:
- cameras: {ap03_cam_csv}

Camera summary:
- {stats(ap03_rows, "camera")}

Camera poses relative to Ref14:
{table(ap03_rows, SHOW_FIELDS)}

Original AP03 scale/registration report:
--------------------------------------
{ap03_extra}
"""

for p in [
    AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.txt",
    AP03_DIR / "AP03_FINAL_REPORT.txt",
]:
    p.write_text(ap03_txt)

(AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.md").write_text(
    "# AP03 Final Ref14-Relative Camera Evaluation\n\n"
    + "AP03 uses targetless COLMAP/SfM to reconstruct cameras. Ref14 is used after COLMAP for metric Sim3 scale/frame registration. AP03 currently does not export a full estimated marker map for markers 0-13.\n\n"
    + md_table(ap03_rows, SHOW_FIELDS)
)
(AP03_DIR / "AP03_FINAL_REPORT.md").write_text((AP3 := AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.md").read_text())

# Common comparison
all_rows = []
all_rows.extend(ap01_rows)
all_rows.extend(ap02_rows)
all_rows.extend(ap03_rows)
write_csv(COMP_DIR / "FINAL_REF14_RELATIVE_ALL_APPROACHES.csv", all_rows, FIELDS)

comparison_txt = f"""FINAL REF14-RELATIVE METHOD COMPARISON
======================================

Common evaluation definition:
- Estimated relative pose: T_ref14_entity_est
- GT relative pose:        T_ref14_entity_gt = inverse(T_world_ref14_gt) * T_world_entity_gt
- Error:                  estimated relative pose vs GT relative pose

Important:
- GT is used only for final evaluation.
- Ref14 simulation GT is not used inside AP01/AP02/AP03 optimization.
- Ref14 defines the output coordinate frame and is therefore not listed as a separate zero-error entity row.

AP01:
- final file: {AP01_DIR / "AP01_FINAL_REF14_RELATIVE_EVALUATION.txt"}
- scope: static cameras
- camera summary: {stats(ap01_rows, "camera")}

AP02:
- final file: {AP02_DIR / "AP02_FINAL_REF14_RELATIVE_EVALUATION.txt"}
- scope: static cameras + marker map 0-13 relative to Ref14
- camera summary: {stats(ap02_rows, "camera")}
- marker summary excluding Ref14: {stats(ap02_rows, "marker")}

AP03:
- final file: {AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.txt"}
- scope: static cameras relative to Ref14; original report also contains Ref14 corner quality
- camera summary: {stats(ap03_rows, "camera")}

Combined CSV:
- {COMP_DIR / "FINAL_REF14_RELATIVE_ALL_APPROACHES.csv"}
"""
(COMP_DIR / "FINAL_REF14_RELATIVE_METHOD_COMPARISON.txt").write_text(comparison_txt)

print("[OK] wrote clean Ref14-relative reports with no fake Ref14 zero-error row")
print("[OK] AP01:", AP01_DIR / "AP01_FINAL_REF14_RELATIVE_EVALUATION.txt")
print("[OK] AP02:", AP02_DIR / "AP02_FINAL_REF14_RELATIVE_EVALUATION.txt")
print("[OK] AP03:", AP03_DIR / "AP03_FINAL_REF14_RELATIVE_EVALUATION.txt")
print("[OK] comparison:", COMP_DIR / "FINAL_REF14_RELATIVE_METHOD_COMPARISON.txt")
