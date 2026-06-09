#!/usr/bin/env python3
import csv
import re
import shutil
from pathlib import Path

ROOT = Path("results/beintelli_bus_model")
DATASET = ROOT / "colmap" / "moving_route_v8_static_8_aruco_boards"
DET_DIR = DATASET / "aruco_no_gt_detections"
NO_GT = ROOT / "no_gt_results"
VIS = ROOT / "multi_static_8_station_visibility"
DEBUG_MOVING = ROOT / "debug_moving_pose"

REPORT = ROOT / "_clean_no_gt_report_m2"
REPORT.mkdir(parents=True, exist_ok=True)

def read_text(path):
    return path.read_text() if path.exists() else ""

def parse_metric(text, key):
    m = re.search(rf"{re.escape(key)}:\s*([-+0-9.eE]+)", text)
    return m.group(1) if m else ""

def parse_summary_value(text, key):
    m = re.search(rf"{re.escape(key)}:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

# ---------------------------------------------------------------------
# 01 Station visibility summary
# ---------------------------------------------------------------------
vis_out = REPORT / "01_station_visibility_summary.txt"
with vis_out.open("w") as out:
    out.write("STATION VISIBILITY SUMMARY\n")
    out.write("==========================\n\n")
    out.write("Meaning:\n")
    out.write("  pose_valid = board pose could be estimated by solvePnP\n")
    out.write("  failed_not_enough_markers = not enough marker IDs of this station visible\n\n")

    for summary in sorted(VIS.glob("*/aruco_board_pose_summary.txt")):
        out.write("\n" + "=" * 100 + "\n")
        out.write(str(summary) + "\n")
        text = read_text(summary)
        for line in text.splitlines():
            if any(k in line for k in [
                "camera:",
                "status:",
                "used_ids:",
                "num_used_markers:",
                "reprojection_rmse_px:",
            ]):
                out.write(line + "\n")

# ---------------------------------------------------------------------
# 02 COLMAP no-GT coverage
# ---------------------------------------------------------------------
colmap_out = REPORT / "02_colmap_no_gt_coverage.txt"

images_dir = DATASET / "images"
images_txt = DATASET / "sparse_txt" / "images.txt"
points_txt = DATASET / "sparse_txt" / "points3D.txt"

total_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))

registered = 0
if images_txt.exists():
    lines = images_txt.read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                int(parts[0])
                float(parts[1])
                registered += 1
            except Exception:
                pass

points = 0
if points_txt.exists():
    for line in points_txt.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            points += 1

ratio = 100.0 * registered / total_images if total_images else 0.0

colmap_out.write_text(f"""COLMAP V8 NO-GT DATASET COVERAGE
================================

dataset:
  {DATASET}

total captured images:
  {total_images}

registered images:
  {registered}

registration ratio:
  {ratio:.2f} %

sparse points:
  {points}

important files:
  images:     {images_dir}
  images.txt: {images_txt}
  points3D:   {points_txt}
""")

# ---------------------------------------------------------------------
# 03 Moving board detections
# ---------------------------------------------------------------------
det_out = REPORT / "03_moving_board_detection_counts.csv"
det_rows = []

for csv_path in sorted(DET_DIR.glob("*_moving_images.csv")):
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r.get("status") == "pose_valid"]
    best = None
    if valid:
        best = sorted(
            valid,
            key=lambda r: (-int(r["num_used_markers"]), float(r["reprojection_rmse_px"]))
        )[0]

    det_rows.append({
        "csv": str(csv_path),
        "valid": len(valid),
        "total": len(rows),
        "valid_percent": f"{100.0 * len(valid) / len(rows):.2f}" if rows else "0.00",
        "best_image": "" if best is None else best["image_name"],
        "best_used_ids": "" if best is None else best["used_ids"],
        "best_rmse_px": "" if best is None else best["reprojection_rmse_px"],
    })

with det_out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()) if det_rows else ["csv"])
    writer.writeheader()
    writer.writerows(det_rows)

# ---------------------------------------------------------------------
# 04 Pair evaluation table
# ---------------------------------------------------------------------
pair_csv = REPORT / "04_pair_evaluation_table.csv"
pair_md = REPORT / "04_pair_evaluation_table.md"
pair_rows = []

for pair_dir in sorted(NO_GT.glob("*_board_scaled_colmap")):
    pair = pair_dir.name.replace("_board_scaled_colmap", "")

    eval_txt = pair_dir / "evaluation_against_static_gt.txt"
    summary_txt = pair_dir / "summary_no_gt.txt"

    eval_text = read_text(eval_txt)
    summary_text = read_text(summary_txt)

    row = {
        "pair": pair,
        "baseline_est_m": parse_metric(eval_text, "baseline_est_m"),
        "baseline_gt_m": parse_metric(eval_text, "baseline_gt_m"),
        "baseline_error_cm": parse_metric(eval_text, "baseline_error_cm"),
        "translation_error_cm": parse_metric(eval_text, "translation_error_cm"),
        "rotation_error_deg": parse_metric(eval_text, "rotation_error_deg"),
        "scale_pairs": parse_summary_value(summary_text, "scale pairs"),
        "scale": parse_summary_value(summary_text, "scale"),
        "sim3_center_rmse_m": parse_summary_value(summary_text, "sim3 center rmse m"),
        "front_image": parse_summary_value(summary_text, "front image"),
        "rear_image": parse_summary_value(summary_text, "rear image"),
    }
    pair_rows.append(row)

pair_rows_sorted = sorted(
    pair_rows,
    key=lambda r: float(r["translation_error_cm"]) if r["translation_error_cm"] else 9999.0
)

if pair_rows_sorted:
    with pair_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows_sorted)

    with pair_md.open("w") as f:
        f.write("# No-GT Front-Rear Pair Evaluation\n\n")
        f.write("| Pair | Baseline est [m] | Baseline error [cm] | Translation error [cm] | Rotation error [deg] | Scale pairs | Sim3 RMSE [m] |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in pair_rows_sorted:
            f.write(
                f"| {r['pair']} | {r['baseline_est_m']} | {r['baseline_error_cm']} | "
                f"{r['translation_error_cm']} | {r['rotation_error_deg']} | "
                f"{r['scale_pairs']} | {r['sim3_center_rmse_m']} |\n"
            )

# ---------------------------------------------------------------------
# 05 Copy important debug images and summaries
# ---------------------------------------------------------------------
img_report = REPORT / "debug_images"
img_report.mkdir(exist_ok=True)

# Static / moving visibility debug images.
for station_dir in sorted(VIS.glob("*")):
    dbg = station_dir / "debug_images"
    if dbg.exists():
        for img in dbg.glob("*.png"):
            copy_if_exists(img, img_report / "station_visibility" / station_dir.name / img.name)

# Manual anchor observations used in F3/R1 run.
for d in [
    DEBUG_MOVING / "F3_manual_fixed_world",
    DEBUG_MOVING / "R1_manual_fixed_world",
]:
    if d.exists():
        for img in (d / "debug_images").glob("*.png"):
            copy_if_exists(img, img_report / "manual_anchor_debug" / d.name / img.name)
        copy_if_exists(d / "aruco_board_pose_observations.csv", REPORT / "anchor_observations" / d.name / "aruco_board_pose_observations.csv")
        copy_if_exists(d / "aruco_board_pose_summary.txt", REPORT / "anchor_observations" / d.name / "aruco_board_pose_summary.txt")

# Detection debug images from image-folder detections.
for d in sorted(DET_DIR.glob("debug_*")):
    if d.exists():
        for img in d.glob("*.jpg"):
            copy_if_exists(img, img_report / "moving_image_detection_debug" / d.name / img.name)
        for img in d.glob("*.png"):
            copy_if_exists(img, img_report / "moving_image_detection_debug" / d.name / img.name)

# Pair summaries.
for pair_dir in sorted(NO_GT.glob("*_board_scaled_colmap")):
    pair = pair_dir.name.replace("_board_scaled_colmap", "")
    copy_if_exists(pair_dir / "summary_no_gt.txt", REPORT / "pair_summaries" / pair / "summary_no_gt.txt")
    copy_if_exists(pair_dir / "evaluation_against_static_gt.txt", REPORT / "pair_summaries" / pair / "evaluation_against_static_gt.txt")
    copy_if_exists(pair_dir / "T_front_rear_no_gt.csv", REPORT / "pair_summaries" / pair / "T_front_rear_no_gt.csv")

# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------
readme = REPORT / "README_WHAT_TO_SHOW.txt"
readme.write_text("""CLEAN NO-GT REPORT CONTENTS
===========================

Use these files for the milestone/progress presentation:

01_station_visibility_summary.txt
  Shows which ArUco station can be used by which camera.
  Important: valid front anchors are F3/F4. Valid rear anchors are R1/R3.

02_colmap_no_gt_coverage.txt
  Shows COLMAP registration quality for V8.

03_moving_board_detection_counts.csv
  Shows how many moving-camera frames detected each board.

04_pair_evaluation_table.csv / .md
  Main result table.
  Translation error and rotation error compare estimated T_front_rear against static-camera GT.
  GT is evaluation-only, not used in estimation.

debug_images/
  Use selected images from station_visibility and manual_anchor_debug for slides.

pair_summaries/
  Full no-GT result summaries and final matrices.

Suggested presentation result:
  Best current translation result: F3_R3.
  Explain F3_R3 as best current station pair by 3D translation error.
""")

print("")
print("[OK] Clean report written to:")
print(REPORT)
print("")
print("Main files:")
print(" ", REPORT / "01_station_visibility_summary.txt")
print(" ", REPORT / "02_colmap_no_gt_coverage.txt")
print(" ", REPORT / "03_moving_board_detection_counts.csv")
print(" ", REPORT / "04_pair_evaluation_table.md")
print(" ", REPORT / "README_WHAT_TO_SHOW.txt")
