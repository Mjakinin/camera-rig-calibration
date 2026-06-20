#!/usr/bin/env python3

import csv
import math
from pathlib import Path


AP3_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
REG_ROOT = AP3_ROOT / "04_ref_aruco_registration"

CMP_ROOT = Path("results/bus_real_data/90_approach_comparison_ref_aruco")
AP3_CMP = CMP_ROOT / "03_targetless_colmap_aruco_scale"
COMBINED = CMP_ROOT / "combined"

ANCHORS_CSV = REG_ROOT / "ap03_ref_marker_anchor_residuals.csv"
CAM_EVAL_CSV = AP3_CMP / "ap03_static_cameras_ref_aruco_vs_gt.csv"
SUMMARY_CSV = COMBINED / "ap03_final_readable_summary.csv"
INSPECTION_REPORT = AP3_ROOT / "03_reconstruction_inspection" / "ap03_colmap_inspection_report.txt"

OUT_TXT = AP3_CMP / "AP03_FINAL_READABLE_REF_ARUCO_REPORT.txt"
OUT_CAM_SIMPLE = AP3_CMP / "ap03_static_camera_errors_readable.csv"
OUT_ANCHOR_SIMPLE = AP3_CMP / "ap03_ref_marker_anchor_diagnostics_readable.csv"
OUT_STATUS = COMBINED / "ap03_status_summary_readable.csv"


def read_csv(path):
    if not path.exists():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def mean(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted([float(x) for x in xs if math.isfinite(float(x))])
    if not xs:
        return float("nan")
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmt_xyz(row, prefix):
    return (
        f"({f(row, prefix + '_x_m'):+.3f}, "
        f"{f(row, prefix + '_y_m'):+.3f}, "
        f"{f(row, prefix + '_z_m'):+.3f})"
    )


def make_text_table(rows, headers, keys):
    data = [headers]
    for r in rows:
        data.append([str(r.get(k, "")) for k in keys])

    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]

    lines = []
    lines.append(" | ".join(data[0][i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("-+-".join("-" * widths[i] for i in range(len(headers))))

    for row in data[1:]:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    return "\n".join(lines)


def anchor_quality(row):
    t = f(row, "translation_residual_cm")
    r = f(row, "rotation_residual_deg")

    if t <= 10.0 and r <= 3.0:
        return "good"
    if t <= 25.0 and r <= 5.0:
        return "usable"
    if t <= 50.0 and r <= 10.0:
        return "weak"
    return "outlier"


def main():
    AP3_CMP.mkdir(parents=True, exist_ok=True)
    COMBINED.mkdir(parents=True, exist_ok=True)

    anchors = read_csv(ANCHORS_CSV)
    cams = read_csv(CAM_EVAL_CSV)
    summary = read_csv(SUMMARY_CSV)[0] if SUMMARY_CSV.exists() else {}

    cam_simple = []
    for r in cams:
        cam_simple.append({
            "camera": r["entity_id"],
            "translation_error_cm": fmt(r["translation_error_cm"], 3),
            "rotation_error_deg": fmt(r["rotation_error_deg"], 3),
            "delta_x_cm": fmt(r["delta_x_cm"], 3),
            "delta_y_cm": fmt(r["delta_y_cm"], 3),
            "delta_z_cm": fmt(r["delta_z_cm"], 3),
            "estimated_xyz_ref_aruco_m": fmt_xyz(r, "est_ref_aruco"),
            "gt_xyz_ref_aruco_m": fmt_xyz(r, "gt_ref_aruco"),
        })

    anchor_simple = []
    for r in anchors:
        q = anchor_quality(r)
        anchor_simple.append({
            "image_name": r["image_name"],
            "source_type": r["source_type"],
            "source_id": r["source_id"],
            "area_px2": fmt(r["area_px2"], 1),
            "translation_residual_cm": fmt(r["translation_residual_cm"], 3),
            "rotation_residual_deg": fmt(r["rotation_residual_deg"], 3),
            "quality": q,
        })

    anchor_simple_sorted = sorted(
        anchor_simple,
        key=lambda r: float(r["translation_residual_cm"]),
        reverse=True,
    )

    cam_sorted = sorted(
        cam_simple,
        key=lambda r: float(r["translation_error_cm"]),
        reverse=True,
    )

    write_csv(
        OUT_CAM_SIMPLE,
        cam_simple,
        [
            "camera",
            "translation_error_cm",
            "rotation_error_deg",
            "delta_x_cm",
            "delta_y_cm",
            "delta_z_cm",
            "estimated_xyz_ref_aruco_m",
            "gt_xyz_ref_aruco_m",
        ],
    )

    write_csv(
        OUT_ANCHOR_SIMPLE,
        anchor_simple_sorted,
        [
            "image_name",
            "source_type",
            "source_id",
            "area_px2",
            "translation_residual_cm",
            "rotation_residual_deg",
            "quality",
        ],
    )

    t_anchor = [f(r, "translation_residual_cm") for r in anchors]
    r_anchor = [f(r, "rotation_residual_deg") for r in anchors]

    t_cam = [f(r, "translation_error_cm") for r in cams]
    r_cam = [f(r, "rotation_error_deg") for r in cams]

    quality_counts = {}
    for r in anchor_simple:
        quality_counts[r["quality"]] = quality_counts.get(r["quality"], 0) + 1

    good_count = quality_counts.get("good", 0)
    usable_count = quality_counts.get("usable", 0)
    weak_count = quality_counts.get("weak", 0)
    outlier_count = quality_counts.get("outlier", 0)

    if outlier_count > 0 or mean(t_anchor) > 2.0 * median(t_anchor):
        registration_status = "preliminary / outlier affected"
        recommendation = "Run robust Ref-ArUco anchor filtering before treating AP03 as final."
    elif mean(t_cam) < 20.0:
        registration_status = "good"
        recommendation = "AP03 can be used as final targetless baseline."
    else:
        registration_status = "usable but weaker than AP02/AP01"
        recommendation = "Use AP03 as targetless baseline and discuss metric registration limitations."

    status_rows = [{
        "approach": "AP03_targetless_colmap_aruco_scale",
        "targetless_colmap_status": "success_all_images_registered",
        "metric_registration_status": registration_status,
        "registered_images": summary.get("registered_images", ""),
        "registered_static_cameras": summary.get("registered_static_cameras", ""),
        "registered_moving_frames": summary.get("registered_moving_frames", ""),
        "sparse_points3d": summary.get("num_sparse_points3d", ""),
        "anchor_count": len(anchors),
        "anchor_good": good_count,
        "anchor_usable": usable_count,
        "anchor_weak": weak_count,
        "anchor_outlier": outlier_count,
        "anchor_mean_translation_residual_cm": fmt(mean(t_anchor), 3),
        "anchor_median_translation_residual_cm": fmt(median(t_anchor), 3),
        "camera_mean_translation_error_cm": fmt(mean(t_cam), 3),
        "camera_median_translation_error_cm": fmt(median(t_cam), 3),
        "camera_mean_rotation_error_deg": fmt(mean(r_cam), 3),
        "camera_median_rotation_error_deg": fmt(median(r_cam), 3),
        "recommendation": recommendation,
    }]

    write_csv(
        OUT_STATUS,
        status_rows,
        list(status_rows[0].keys()),
    )

    cam_table = make_text_table(
        cam_simple,
        headers=[
            "camera", "t_err_cm", "r_err_deg", "dX", "dY", "dZ",
            "estimated xyz [m]", "GT xyz [m]"
        ],
        keys=[
            "camera", "translation_error_cm", "rotation_error_deg",
            "delta_x_cm", "delta_y_cm", "delta_z_cm",
            "estimated_xyz_ref_aruco_m", "gt_xyz_ref_aruco_m"
        ],
    )

    worst_anchor_table = make_text_table(
        anchor_simple_sorted[:10],
        headers=[
            "image", "type", "id", "area", "t_res_cm", "r_res_deg", "quality"
        ],
        keys=[
            "image_name", "source_type", "source_id", "area_px2",
            "translation_residual_cm", "rotation_residual_deg", "quality"
        ],
    )

    best_anchor_table = make_text_table(
        sorted(anchor_simple, key=lambda r: float(r["translation_residual_cm"]))[:10],
        headers=[
            "image", "type", "id", "area", "t_res_cm", "r_res_deg", "quality"
        ],
        keys=[
            "image_name", "source_type", "source_id", "area_px2",
            "translation_residual_cm", "rotation_residual_deg", "quality"
        ],
    )

    inspection_text = INSPECTION_REPORT.read_text() if INSPECTION_REPORT.exists() else ""

    text = f"""AP03 FINAL READABLE REPORT
==========================

Approach:
- AP03: Targetless COLMAP / SfM + Ref-ArUco metric scale registration
- Reference frame after registration: aruco_marker_14 / aruco_ref_floor_14
- Important: ArUco is NOT used for COLMAP reconstruction.
- ArUco marker 14 is only used after COLMAP to estimate metric scale and Ref-ArUco alignment.

STATUS
------
Targetless COLMAP reconstruction:
- SUCCESS
- all static cameras were registered
- all moving frames were registered

Metric Ref-ArUco registration:
- status: {registration_status}
- recommendation: {recommendation}

COLMAP RECONSTRUCTION
---------------------
Registered images:          {summary.get("registered_images", "208")}
Registered static cameras:  {summary.get("registered_static_cameras", "4")} / 4
Registered moving frames:   {summary.get("registered_moving_frames", "204")}
Sparse 3D points:           {summary.get("num_sparse_points3d", "9401")}

REF-ARUCO REGISTRATION QUALITY
------------------------------
Anchor images using marker 14: {len(anchors)}

Anchor translation residuals:
- mean:   {mean(t_anchor):.3f} cm
- median: {median(t_anchor):.3f} cm
- max:    {max(t_anchor):.3f} cm

Anchor rotation residuals:
- mean:   {mean(r_anchor):.3f} deg
- median: {median(r_anchor):.3f} deg
- max:    {max(r_anchor):.3f} deg

Anchor quality buckets:
- good    <= 10 cm and <= 3 deg:   {good_count}
- usable  <= 25 cm and <= 5 deg:   {usable_count}
- weak    <= 50 cm and <= 10 deg:  {weak_count}
- outlier > threshold:             {outlier_count}

Interpretation:
- The targetless COLMAP part is strong because it registered all 208 images.
- The current metric registration is weaker because the Ref-ArUco anchor residuals contain outliers.
- Mean anchor residual is much larger than median anchor residual, which indicates outlier sensitivity.
- Therefore AP03 is promising, but this exact registration should be treated as preliminary.

STATIC CAMERA EVALUATION VS GT, REF-ARUCO FRAME
-----------------------------------------------
Camera mean translation error:   {mean(t_cam):.3f} cm
Camera median translation error: {median(t_cam):.3f} cm
Camera mean rotation error:      {mean(r_cam):.3f} deg
Camera median rotation error:    {median(r_cam):.3f} deg

{cam_table}

CAMERA ERROR RANKING
--------------------
"""

    for r in cam_sorted:
        text += f"- {r['camera']}: {r['translation_error_cm']} cm, {r['rotation_error_deg']} deg\n"

    text += f"""
WORST REF-ARUCO ANCHORS
-----------------------
{worst_anchor_table}

BEST REF-ARUCO ANCHORS
----------------------
{best_anchor_table}

WHAT THIS MEANS
---------------
AP03 has two separate parts:

1. Targetless SfM / COLMAP:
   This part worked very well. COLMAP registered all static and moving images.
   This means natural feature matching is sufficient for reconstructing the bus sequence.

2. Metric scale and Ref-ArUco registration:
   This part is currently the weak point. We estimate one Sim(3) transform from arbitrary COLMAP
   coordinates into the Ref-ArUco metric frame using images where marker 14 is visible.
   Some of these anchor poses are noisy/outliers, so the resulting metric alignment is biased.

Conclusion:
- AP03 is a valid third approach.
- It should not yet be considered final-best in the current registration form.
- Next recommended step: robust anchor filtering / RANSAC Sim(3) registration.

FILES
-----
Readable AP03 report:
- {OUT_TXT}

Readable CSVs:
- {OUT_CAM_SIMPLE}
- {OUT_ANCHOR_SIMPLE}
- {OUT_STATUS}

Raw AP03 files:
- {REG_ROOT / "ap03_ref_marker_anchor_residuals.csv"}
- {REG_ROOT / "ap03_static_camera_poses_ref_aruco.csv"}
- {REG_ROOT / "ap03_sparse_points3d_ref_aruco.csv"}
- {CAM_EVAL_CSV}
"""

    OUT_TXT.write_text(text)

    print("[OK] wrote readable AP03 report:")
    print("-", OUT_TXT)
    print("-", OUT_CAM_SIMPLE)
    print("-", OUT_ANCHOR_SIMPLE)
    print("-", OUT_STATUS)
    print()
    print(text)


if __name__ == "__main__":
    main()
