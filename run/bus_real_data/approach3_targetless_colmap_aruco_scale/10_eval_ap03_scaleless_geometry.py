#!/usr/bin/env python3

import csv
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import STATIC_CAMERAS, WORLD_SDF_MOVING_CAMERA, REF_MARKER_ENTITY
from _shared.common.sdf_utils import gt_static_camera_poses_ref_aruco
from _shared.common.io_utils import ensure_dir, write_csv
from _shared.common.geometry import mean, median
from ap03_scale_common import load_best_colmap_model


AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
OUT_ROOT = AP03_ROOT / "08_scaleless_geometry_evaluation"
FINAL_ROOT = AP03_ROOT / "07_final_results"
FINAL_REPORT_ROOT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP03")


def source_id_from_image_name(name: str) -> str:
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    return name


def fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def md_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def load_colmap_static_camera_centers():
    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()

    centers = {}
    image_names = {}

    for image_name, payload in images.items():
        if not image_name.startswith("static_"):
            continue

        cam = source_id_from_image_name(image_name)
        if cam not in STATIC_CAMERAS:
            continue

        T_col_cam = payload["T_col_cam"]
        centers[cam] = np.asarray(T_col_cam[:3, 3], dtype=float)
        image_names[cam] = image_name

    missing = [cam for cam in STATIC_CAMERAS if cam not in centers]
    if missing:
        raise RuntimeError(f"Missing static cameras in COLMAP model: {missing}")

    return best_model, model_dir, centers, image_names, images, points3d


def load_gt_static_camera_centers_ref14():
    gt = gt_static_camera_poses_ref_aruco(
        WORLD_SDF_MOVING_CAMERA,
        STATIC_CAMERAS,
        REF_MARKER_ENTITY,
    )

    centers = {}
    for cam in STATIC_CAMERAS:
        centers[cam] = np.asarray(gt[cam][:3, 3], dtype=float)

    return centers


def compute_pairwise_rows(col_centers, gt_centers):
    raw_rows = []
    d_est = []
    d_gt = []

    for cam_a, cam_b in combinations(STATIC_CAMERAS, 2):
        de = dist(col_centers[cam_a], col_centers[cam_b])
        dg = dist(gt_centers[cam_a], gt_centers[cam_b])

        raw_rows.append({
            "cam_a": cam_a,
            "cam_b": cam_b,
            "d_colmap_raw": de,
            "d_gt_m": dg,
        })

        d_est.append(de)
        d_gt.append(dg)

    d_est = np.asarray(d_est, dtype=float)
    d_gt = np.asarray(d_gt, dtype=float)

    if np.any(d_est <= 1e-12):
        raise RuntimeError("At least one raw COLMAP camera distance is zero or invalid.")

    # Scale that best maps raw COLMAP pairwise distances to GT pairwise distances.
    # This is evaluation-only and does not use ArUco.
    s_pairwise = float(np.sum(d_est * d_gt) / np.sum(d_est * d_est))

    med_est = float(np.median(d_est))
    med_gt = float(np.median(d_gt))

    pairwise_rows = []
    for r in raw_rows:
        de = float(r["d_colmap_raw"])
        dg = float(r["d_gt_m"])

        d_scaled = s_pairwise * de
        scaled_err_cm = abs(d_scaled - dg) * 100.0

        norm_est = de / med_est
        norm_gt = dg / med_gt
        norm_abs_err = abs(norm_est - norm_gt)
        norm_rel_err_pct = 100.0 * norm_abs_err / max(abs(norm_gt), 1e-12)

        rr = dict(r)
        rr.update({
            "eval_only_best_fit_distance_scale": s_pairwise,
            "d_colmap_scaled_m": d_scaled,
            "scaled_distance_error_cm": scaled_err_cm,
            "d_colmap_norm_by_median": norm_est,
            "d_gt_norm_by_median": norm_gt,
            "normalized_distance_abs_error": norm_abs_err,
            "normalized_distance_rel_error_pct": norm_rel_err_pct,
        })
        pairwise_rows.append(rr)

    return pairwise_rows, s_pairwise


def compute_ratio_rows(pairwise_rows):
    ratio_rows = []

    for r1, r2 in combinations(pairwise_rows, 2):
        label_1 = f"{r1['cam_a']}-{r1['cam_b']}"
        label_2 = f"{r2['cam_a']}-{r2['cam_b']}"

        est_ratio = float(r1["d_colmap_raw"]) / float(r2["d_colmap_raw"])
        gt_ratio = float(r1["d_gt_m"]) / float(r2["d_gt_m"])

        abs_err = abs(est_ratio - gt_ratio)
        rel_err_pct = 100.0 * abs_err / max(abs(gt_ratio), 1e-12)

        ratio_rows.append({
            "ratio_id": f"{label_1}_over_{label_2}",
            "numerator_pair": label_1,
            "denominator_pair": label_2,
            "ratio_colmap_raw": est_ratio,
            "ratio_gt": gt_ratio,
            "ratio_abs_error": abs_err,
            "ratio_rel_error_pct": rel_err_pct,
        })

    return ratio_rows


def append_scaleless_section_to_report(report_path: Path, section: str):
    if not report_path.exists():
        return

    marker = "\n\nAP03 SCALE-LESS COLMAP GEOMETRY EVALUATION\n"
    text = report_path.read_text()

    if marker in text:
        text = text.split(marker)[0].rstrip()

    report_path.write_text(text.rstrip() + marker + section.strip() + "\n")


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(FINAL_ROOT)
    ensure_dir(FINAL_REPORT_ROOT)

    best_model, model_dir, col_centers, image_names, images, points3d = load_colmap_static_camera_centers()
    gt_centers = load_gt_static_camera_centers_ref14()

    pairwise_rows, s_pairwise = compute_pairwise_rows(col_centers, gt_centers)
    ratio_rows = compute_ratio_rows(pairwise_rows)

    pairwise_fields = [
        "cam_a", "cam_b",
        "d_colmap_raw", "d_gt_m",
        "eval_only_best_fit_distance_scale",
        "d_colmap_scaled_m",
        "scaled_distance_error_cm",
        "d_colmap_norm_by_median",
        "d_gt_norm_by_median",
        "normalized_distance_abs_error",
        "normalized_distance_rel_error_pct",
    ]

    ratio_fields = [
        "ratio_id",
        "numerator_pair",
        "denominator_pair",
        "ratio_colmap_raw",
        "ratio_gt",
        "ratio_abs_error",
        "ratio_rel_error_pct",
    ]

    write_csv(OUT_ROOT / "AP03_SCALELESS_PAIRWISE_CAMERA_DISTANCES.csv", pairwise_rows, pairwise_fields)
    write_csv(OUT_ROOT / "AP03_SCALELESS_DISTANCE_RATIOS.csv", ratio_rows, ratio_fields)

    write_csv(FINAL_ROOT / "AP03_SCALELESS_PAIRWISE_CAMERA_DISTANCES.csv", pairwise_rows, pairwise_fields)
    write_csv(FINAL_ROOT / "AP03_SCALELESS_DISTANCE_RATIOS.csv", ratio_rows, ratio_fields)

    write_csv(FINAL_REPORT_ROOT / "AP03_SCALELESS_PAIRWISE_CAMERA_DISTANCES.csv", pairwise_rows, pairwise_fields)
    write_csv(FINAL_REPORT_ROOT / "AP03_SCALELESS_DISTANCE_RATIOS.csv", ratio_rows, ratio_fields)

    scaled_errs = [float(r["scaled_distance_error_cm"]) for r in pairwise_rows]
    norm_errs = [float(r["normalized_distance_abs_error"]) for r in pairwise_rows]
    norm_rel = [float(r["normalized_distance_rel_error_pct"]) for r in pairwise_rows]
    ratio_abs = [float(r["ratio_abs_error"]) for r in ratio_rows]
    ratio_rel = [float(r["ratio_rel_error_pct"]) for r in ratio_rows]

    summary_rows = [
        ["static_camera_pairs", len(pairwise_rows)],
        ["distance_ratios", len(ratio_rows)],
        ["eval_only_best_fit_distance_scale", fmt(s_pairwise, 9)],
        ["mean_scaled_distance_error_cm", fmt(mean(scaled_errs))],
        ["median_scaled_distance_error_cm", fmt(median(scaled_errs))],
        ["mean_normalized_distance_abs_error", fmt(mean(norm_errs), 6)],
        ["median_normalized_distance_abs_error", fmt(median(norm_errs), 6)],
        ["mean_normalized_distance_rel_error_pct", fmt(mean(norm_rel))],
        ["median_normalized_distance_rel_error_pct", fmt(median(norm_rel))],
        ["mean_ratio_abs_error", fmt(mean(ratio_abs), 6)],
        ["median_ratio_abs_error", fmt(median(ratio_abs), 6)],
        ["mean_ratio_rel_error_pct", fmt(mean(ratio_rel))],
        ["median_ratio_rel_error_pct", fmt(median(ratio_rel))],
    ]

    pair_table = []
    for r in pairwise_rows:
        pair_table.append([
            f"{r['cam_a']}-{r['cam_b']}",
            fmt(r["d_colmap_raw"], 6),
            fmt(r["d_gt_m"], 3),
            fmt(r["d_colmap_scaled_m"], 3),
            fmt(r["scaled_distance_error_cm"]),
            fmt(r["d_colmap_norm_by_median"], 3),
            fmt(r["d_gt_norm_by_median"], 3),
            fmt(r["normalized_distance_rel_error_pct"]),
        ])

    # Keep ratio table compact: top 8 worst ratios by relative error.
    ratio_rows_sorted = sorted(ratio_rows, key=lambda r: float(r["ratio_rel_error_pct"]), reverse=True)
    ratio_table = []
    for r in ratio_rows_sorted[:8]:
        ratio_table.append([
            r["numerator_pair"],
            r["denominator_pair"],
            fmt(r["ratio_colmap_raw"], 4),
            fmt(r["ratio_gt"], 4),
            fmt(r["ratio_abs_error"], 4),
            fmt(r["ratio_rel_error_pct"]),
        ])

    report = f"""AP03 SCALE-LESS COLMAP GEOMETRY EVALUATION
==========================================

Purpose:
This evaluates the raw COLMAP static-camera geometry before ArUco Sim(3) metric registration.
COLMAP is scale-less, so raw COLMAP distances are not directly compared in meters.
Instead, we evaluate whether the relative rig geometry is correct up to scale.

Important:
- This is targetless geometry evaluation before metric registration.
- ArUco markers are not used for this scale-less metric.
- The best-fit pairwise distance scale below is evaluation-only.
- It is not the AP03 Single/Multi-ArUco Sim(3) registration scale.
- GT is used only for evaluation.

COLMAP model:
- best model: {best_model}
- model dir: {model_dir}
- registered images: {len(images)}
- sparse 3D points: {len(points3d)}
- static cameras evaluated: {len(STATIC_CAMERAS)}

Summary:
{md_table(["metric", "value"], summary_rows)}

Pairwise static-camera distances:
{md_table(["pair", "raw_colmap", "gt_m", "scaled_m", "err_cm", "norm_col", "norm_gt", "norm_rel_%"], pair_table)}

Worst distance-ratio errors:
{md_table(["num_pair", "den_pair", "ratio_col", "ratio_gt", "abs_err", "rel_%"], ratio_table)}

Interpretation:
If the normalized distance and ratio errors are small, COLMAP recovered the static-camera rig geometry well up to scale.
The later ArUco Sim(3) step should then mainly solve metric scale, rotation, and frame registration.
If these errors are large, the targetless reconstruction itself is geometrically inconsistent, independent of metric registration.
"""

    for p in [
        OUT_ROOT / "AP03_SCALELESS_GEOMETRY_EVALUATION.txt",
        FINAL_ROOT / "AP03_SCALELESS_GEOMETRY_EVALUATION.txt",
        FINAL_REPORT_ROOT / "AP03_SCALELESS_GEOMETRY_EVALUATION.txt",
    ]:
        p.write_text(report)

    # Append compact section to main AP03 final report in both locations.
    append_scaleless_section_to_report(FINAL_ROOT / "AP03_FINAL_RESULT.txt", report)
    append_scaleless_section_to_report(FINAL_REPORT_ROOT / "AP03_FINAL_RESULT.txt", report)

    print(report)
    print()
    print("[OK] wrote:")
    print(" ", OUT_ROOT / "AP03_SCALELESS_GEOMETRY_EVALUATION.txt")
    print(" ", OUT_ROOT / "AP03_SCALELESS_PAIRWISE_CAMERA_DISTANCES.csv")
    print(" ", OUT_ROOT / "AP03_SCALELESS_DISTANCE_RATIOS.csv")
    print(" ", FINAL_REPORT_ROOT / "AP03_SCALELESS_GEOMETRY_EVALUATION.txt")
    print(" ", FINAL_REPORT_ROOT / "AP03_SCALELESS_PAIRWISE_CAMERA_DISTANCES.csv")
    print(" ", FINAL_REPORT_ROOT / "AP03_SCALELESS_DISTANCE_RATIOS.csv")
    print("[OK] appended scale-less section to AP03_FINAL_RESULT.txt")


if __name__ == "__main__":
    main()
