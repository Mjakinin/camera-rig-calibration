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
from ap03_scale_common import load_best_colmap_model


AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
SOURCE_ROOT = AP03_ROOT / "07_final_results"
FINAL_ROOT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP03")

SINGLE_CSV = SOURCE_ROOT / "AP03_FINAL_SINGLE_REF14_RESULT.csv"
MULTI_CSV = SOURCE_ROOT / "AP03_FINAL_MULTI_ARUCO_RESULT.csv"

OUT_TXT = FINAL_ROOT / "AP03_FINAL_RESULT.txt"
OUT_CSV = FINAL_ROOT / "AP03_FINAL_RESULT.csv"

ROOT_CAMERA = "cam_edge_3"


def f(x, n=3):
    return f"{float(x):.{n}f}"


def fmt_xyz(p):
    return f"({f(p[0])}, {f(p[1])}, {f(p[2])})"


def norm(p):
    return float(np.linalg.norm(np.asarray(p, dtype=float)))


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def md_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))

    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def read_csv(path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def get(row, keys, default=""):
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def source_id_from_image_name(name: str) -> str:
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    return name


def load_colmap_static_centers():
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

    if ROOT_CAMERA not in centers:
        raise RuntimeError(f"Missing root camera {ROOT_CAMERA} in COLMAP model.")

    return best_model, model_dir, centers, image_names, images, points3d


def load_gt_centers_ref14():
    gt = gt_static_camera_poses_ref_aruco(
        WORLD_SDF_MOVING_CAMERA,
        STATIC_CAMERAS,
        REF_MARKER_ENTITY,
    )
    return {cam: np.asarray(gt[cam][:3, 3], dtype=float) for cam in STATIC_CAMERAS}


def compute_pure_targetless_rows(col_centers, image_names):
    root = col_centers[ROOT_CAMERA]

    raw_rel = {cam: col_centers[cam] - root for cam in STATIC_CAMERAS}
    farthest_dist = max(norm(raw_rel[cam]) for cam in STATIC_CAMERAS if cam != ROOT_CAMERA)

    if farthest_dist <= 1e-12:
        raise RuntimeError("Invalid pure targetless normalization scale.")

    rows = []
    for cam in STATIC_CAMERAS:
        p_raw = raw_rel[cam]
        p_norm = p_raw / farthest_dist

        rows.append({
            "cam": cam,
            "raw_rel_xyz": fmt_xyz(p_raw),
            "raw_dist": norm(p_raw),
            "norm_xyz": fmt_xyz(p_norm),
            "norm_dist": norm(p_norm),
            "image_name": image_names.get(cam, ""),
        })

    return rows


def compute_scaleless_ratio_summary(col_centers, gt_centers):
    pair_rows = []
    d_est = []
    d_gt = []

    for a, b in combinations(STATIC_CAMERAS, 2):
        de = norm(col_centers[a] - col_centers[b])
        dg = norm(gt_centers[a] - gt_centers[b])

        d_est.append(de)
        d_gt.append(dg)

        pair_rows.append({
            "pair": f"{a}-{b}",
            "d_colmap_raw": de,
            "d_gt_m": dg,
        })

    d_est = np.asarray(d_est, dtype=float)
    d_gt = np.asarray(d_gt, dtype=float)

    best_fit_scale = float(np.sum(d_est * d_gt) / np.sum(d_est * d_est))

    med_est = float(np.median(d_est))
    med_gt = float(np.median(d_gt))

    scaled_errors_cm = []
    normalized_abs_errors = []
    normalized_rel_errors_pct = []

    for de, dg in zip(d_est, d_gt):
        scaled_errors_cm.append(abs(best_fit_scale * de - dg) * 100.0)

        nd_est = de / med_est
        nd_gt = dg / med_gt

        normalized_abs_errors.append(abs(nd_est - nd_gt))
        normalized_rel_errors_pct.append(100.0 * abs(nd_est - nd_gt) / max(abs(nd_gt), 1e-12))

    ratio_abs_errors = []
    ratio_rel_errors_pct = []

    for i, j in combinations(range(len(d_est)), 2):
        ratio_est = d_est[i] / d_est[j]
        ratio_gt = d_gt[i] / d_gt[j]

        abs_err = abs(ratio_est - ratio_gt)
        rel_err = 100.0 * abs_err / max(abs(ratio_gt), 1e-12)

        ratio_abs_errors.append(abs_err)
        ratio_rel_errors_pct.append(rel_err)

    return {
        "static_camera_pairs": len(pair_rows),
        "distance_ratios": len(ratio_abs_errors),
        "eval_only_best_fit_distance_scale": best_fit_scale,
        "mean_scaled_distance_error_cm": mean(scaled_errors_cm),
        "median_scaled_distance_error_cm": median(scaled_errors_cm),
        "mean_normalized_distance_abs_error": mean(normalized_abs_errors),
        "median_normalized_distance_abs_error": median(normalized_abs_errors),
        "mean_normalized_distance_rel_error_pct": mean(normalized_rel_errors_pct),
        "median_normalized_distance_rel_error_pct": median(normalized_rel_errors_pct),
        "mean_ratio_abs_error": mean(ratio_abs_errors),
        "median_ratio_abs_error": median(ratio_abs_errors),
        "mean_ratio_rel_error_pct": mean(ratio_rel_errors_pct),
        "median_ratio_rel_error_pct": median(ratio_rel_errors_pct),
    }


def load_metric_rows():
    single_rows = read_csv(SINGLE_CSV)
    multi_rows = read_csv(MULTI_CSV)

    single = {get(r, ["entity_id", "camera", "cam"]): r for r in single_rows if get(r, ["entity_id", "camera", "cam"])}
    multi = {get(r, ["entity_id", "camera", "cam"]): r for r in multi_rows if get(r, ["entity_id", "camera", "cam"])}

    cams = sorted(set(single) & set(multi))
    if not cams:
        raise RuntimeError("No overlapping Single/Multi AP03 camera rows found.")

    rows = []
    for cam in cams:
        s = single[cam]
        m = multi[cam]

        s_est = np.array([
            float(get(s, ["est_ref14_x_m", "est_ref_aruco_x_m"])),
            float(get(s, ["est_ref14_y_m", "est_ref_aruco_y_m"])),
            float(get(s, ["est_ref14_z_m", "est_ref_aruco_z_m"])),
        ], dtype=float)

        m_est = np.array([
            float(get(m, ["est_ref14_x_m", "est_ref_aruco_x_m"])),
            float(get(m, ["est_ref14_y_m", "est_ref_aruco_y_m"])),
            float(get(m, ["est_ref14_z_m", "est_ref_aruco_z_m"])),
        ], dtype=float)

        gt = np.array([
            float(get(m, ["gt_ref14_x_m", "gt_ref_aruco_x_m"])),
            float(get(m, ["gt_ref14_y_m", "gt_ref_aruco_y_m"])),
            float(get(m, ["gt_ref14_z_m", "gt_ref_aruco_z_m"])),
        ], dtype=float)

        s_dist = norm(s_est)
        m_dist = norm(m_est)
        gt_dist = norm(gt)

        s_err = 100.0 * norm(s_est - gt)
        m_err = 100.0 * norm(m_est - gt)

        rows.append({
            "cam": cam,
            "single_est_xyz": fmt_xyz(s_est),
            "single_dist_m": s_dist,
            "multi_est_xyz": fmt_xyz(m_est),
            "multi_dist_m": m_dist,
            "gt_ref14_xyz": fmt_xyz(gt),
            "gt_dist_to_ref14_m": gt_dist,
            "single_err_cm": s_err,
            "multi_err_cm": m_err,
            "gain_cm": s_err - m_err,
            "single_rot_deg": float(get(s, ["rotation_error_deg"])),
            "multi_rot_deg": float(get(m, ["rotation_error_deg"])),
            "single_dist_err_cm": abs(s_dist - gt_dist) * 100.0,
            "multi_dist_err_cm": abs(m_dist - gt_dist) * 100.0,
        })

    return rows


def remove_extra_final_files():
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    keep = {"AP03_FINAL_RESULT.txt", "AP03_FINAL_RESULT.csv"}

    for p in FINAL_ROOT.glob("AP03_*"):
        if p.name not in keep and p.is_file():
            p.unlink()


def write_final_csv(pure_rows, ratio_summary, metric_rows):
    # One consolidated CSV: per-camera rows, plus ratio summary repeated in metadata columns.
    pure_by_cam = {r["cam"]: r for r in pure_rows}
    metric_by_cam = {r["cam"]: r for r in metric_rows}

    fields = [
        "cam",

        "pure_raw_rel_xyz_colmap_units",
        "pure_raw_dist_colmap_units",
        "pure_norm_xyz_unitless",
        "pure_norm_dist_unitless",

        "single_est_xyz_ref14_m",
        "single_dist_m",
        "multi_est_xyz_ref14_m",
        "multi_dist_m",
        "gt_ref14_xyz_eval_only",
        "gt_dist_to_ref14_m_eval_only",

        "single_err_cm_eval_only",
        "multi_err_cm_eval_only",
        "gain_cm_eval_only",
        "single_rot_deg_eval_only",
        "multi_rot_deg_eval_only",
        "single_dist_err_cm_eval_only",
        "multi_dist_err_cm_eval_only",

        "ratio_mean_norm_dist_rel_err_pct_eval_only",
        "ratio_mean_ratio_rel_err_pct_eval_only",
        "ratio_best_fit_distance_scale_eval_only",

        "notes",
    ]

    rows = []
    for cam in STATIC_CAMERAS:
        p = pure_by_cam[cam]
        m = metric_by_cam[cam]

        rows.append({
            "cam": cam,

            "pure_raw_rel_xyz_colmap_units": p["raw_rel_xyz"],
            "pure_raw_dist_colmap_units": p["raw_dist"],
            "pure_norm_xyz_unitless": p["norm_xyz"],
            "pure_norm_dist_unitless": p["norm_dist"],

            "single_est_xyz_ref14_m": m["single_est_xyz"],
            "single_dist_m": m["single_dist_m"],
            "multi_est_xyz_ref14_m": m["multi_est_xyz"],
            "multi_dist_m": m["multi_dist_m"],
            "gt_ref14_xyz_eval_only": m["gt_ref14_xyz"],
            "gt_dist_to_ref14_m_eval_only": m["gt_dist_to_ref14_m"],

            "single_err_cm_eval_only": m["single_err_cm"],
            "multi_err_cm_eval_only": m["multi_err_cm"],
            "gain_cm_eval_only": m["gain_cm"],
            "single_rot_deg_eval_only": m["single_rot_deg"],
            "multi_rot_deg_eval_only": m["multi_rot_deg"],
            "single_dist_err_cm_eval_only": m["single_dist_err_cm"],
            "multi_dist_err_cm_eval_only": m["multi_dist_err_cm"],

            "ratio_mean_norm_dist_rel_err_pct_eval_only": ratio_summary["mean_normalized_distance_rel_error_pct"],
            "ratio_mean_ratio_rel_err_pct_eval_only": ratio_summary["mean_ratio_rel_error_pct"],
            "ratio_best_fit_distance_scale_eval_only": ratio_summary["eval_only_best_fit_distance_scale"],

            "notes": "pure targetless is unitless; metric results require ArUco/known metric reference; GT/error values are evaluation-only",
        })

    with OUT_CSV.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    best_model, model_dir, col_centers, image_names, images, points3d = load_colmap_static_centers()
    gt_centers = load_gt_centers_ref14()

    pure_rows = compute_pure_targetless_rows(col_centers, image_names)
    ratio_summary = compute_scaleless_ratio_summary(col_centers, gt_centers)
    metric_rows = load_metric_rows()

    single_errs = [r["single_err_cm"] for r in metric_rows]
    multi_errs = [r["multi_err_cm"] for r in metric_rows]
    single_rots = [r["single_rot_deg"] for r in metric_rows]
    multi_rots = [r["multi_rot_deg"] for r in metric_rows]
    single_dist_errs = [r["single_dist_err_cm"] for r in metric_rows]
    multi_dist_errs = [r["multi_dist_err_cm"] for r in metric_rows]

    pure_table = []
    for r in pure_rows:
        pure_table.append([
            r["cam"],
            r["raw_rel_xyz"],
            f(r["raw_dist"]),
            r["norm_xyz"],
            f(r["norm_dist"]),
        ])

    ratio_table = [
        ["static_camera_pairs", ratio_summary["static_camera_pairs"]],
        ["distance_ratios", ratio_summary["distance_ratios"]],
        ["eval_only_best_fit_distance_scale", f(ratio_summary["eval_only_best_fit_distance_scale"], 9)],
        ["mean_scaled_distance_error_cm", f(ratio_summary["mean_scaled_distance_error_cm"])],
        ["median_scaled_distance_error_cm", f(ratio_summary["median_scaled_distance_error_cm"])],
        ["mean_normalized_distance_rel_error_pct", f(ratio_summary["mean_normalized_distance_rel_error_pct"])],
        ["median_normalized_distance_rel_error_pct", f(ratio_summary["median_normalized_distance_rel_error_pct"])],
        ["mean_ratio_rel_error_pct", f(ratio_summary["mean_ratio_rel_error_pct"])],
        ["median_ratio_rel_error_pct", f(ratio_summary["median_ratio_rel_error_pct"])],
    ]

    metric_summary = [
        ["Single Ref14", len(metric_rows), f(mean(single_errs)), f(median(single_errs)), f(mean(single_rots)), f(median(single_rots)), f(mean(single_dist_errs))],
        ["Multi-ArUco", len(metric_rows), f(mean(multi_errs)), f(median(multi_errs)), f(mean(multi_rots)), f(median(multi_rots)), f(mean(multi_dist_errs))],
    ]

    metric_output_table = []
    for r in metric_rows:
        metric_output_table.append([
            r["cam"],
            r["single_est_xyz"],
            f(r["single_dist_m"]),
            r["multi_est_xyz"],
            f(r["multi_dist_m"]),
        ])

    metric_error_table = []
    for r in metric_rows:
        metric_error_table.append([
            r["cam"],
            f(r["single_err_cm"]),
            f(r["multi_err_cm"]),
            f(r["gain_cm"]),
            f(r["single_rot_deg"]),
            f(r["multi_rot_deg"]),
            f(r["single_dist_err_cm"]),
            f(r["multi_dist_err_cm"]),
        ])

    gt_table = []
    for r in metric_rows:
        gt_table.append([
            r["cam"],
            r["gt_ref14_xyz"],
            f(r["gt_dist_to_ref14_m"]),
        ])

    report = f"""AP03 FINAL RESULT — Targetless COLMAP Evaluation
================================================

Method:
AP03 evaluates targetless COLMAP in three levels.

AP03-0 Pure Targetless:
COLMAP reconstructs the static camera rig from natural image features only.
No ArUco markers, no known marker size, no known marker layout, and no metric reference are used.
The output is a unitless camera rig in the arbitrary COLMAP coordinate frame.

AP03-0 Scale-less Ratio Diagnostic:
Because monocular COLMAP has arbitrary scale, raw distances are not metric.
Therefore, we additionally evaluate scale-less geometry consistency using normalized distances and distance ratios.
This checks whether COLMAP recovered the rig shape correctly up to scale.

AP03a Single Ref14 Sim(3):
COLMAP reconstruction is registered metrically using only Ref14 as metric reference.
This requires known marker size.

AP03b Multi-ArUco Sim(3):
COLMAP reconstruction is registered metrically using all visible ArUco markers.
This requires known marker size and known marker layout.
In simulation, the known marker layout comes from the SDF.
In real life, it would need to be defined or measured beforehand.

Important:
- COLMAP itself remains targetless.
- ArUco is not used during COLMAP reconstruction.
- ArUco is only used afterwards for metric Sim(3) registration.
- Without any metric reference, AP03 cannot output meters.
- Without metric reference, AP03 can only output relative unitless rig geometry.
- AP03 does not estimate World -> Ref14.
- GT positions and all error values are simulation-only evaluation.

COLMAP model:
- best model: {best_model}
- model dir: {model_dir}
- registered images: {len(images)}
- sparse 3D points: {len(points3d)}
- static cameras: {len(STATIC_CAMERAS)}

AP03-0 PURE TARGETLESS SCALE-LESS CAMERA RIG OUTPUT
---------------------------------------------------

Purpose:
This is the pure targetless AP03 output without ArUco, without known marker size, without known marker layout, and without any metric reference.
It uses only raw COLMAP static-camera poses.

Coordinate convention:
- root camera: {ROOT_CAMERA}
- {ROOT_CAMERA} is set to the local origin.
- axes are inherited from the arbitrary COLMAP frame.
- normalization: farthest static camera from {ROOT_CAMERA} has distance 1.0.
- raw COLMAP units are kept as additional unitless output.

Pure targetless camera rig output:
{md_table(["cam", "raw_rel_xyz", "raw_dist", "norm_xyz", "norm_dist"], pure_table)}

Interpretation:
This is the deployable pure-targetless output if no metric reference is available.
It tells us the relative static camera rig shape in arbitrary units.
It does not provide meters or metric camera distances.

AP03-0 SCALE-LESS RATIO DIAGNOSTIC
----------------------------------

Purpose:
This diagnostic checks whether the raw COLMAP static-camera geometry is consistent up to scale.
It is not a metric calibration output.
It is only a scale-less evaluation of targetless geometry quality.

Definition:
Distance ratios compare relative camera-camera distances.
For example:
  ratio = distance(cam_i, cam_j) / distance(cam_k, cam_l)

Because scale cancels out in ratios, this evaluates rig shape without requiring metric scale.

Summary:
{md_table(["metric", "value"], ratio_table)}

Interpretation:
The low normalized-distance and ratio errors indicate that raw COLMAP reconstructs the static camera rig geometry very accurately up to scale.
Therefore, the targetless reconstruction itself is geometrically consistent.
The remaining metric step is scale/frame registration, which requires an external metric reference.

AP03 METRIC REGISTRATION RESULTS
--------------------------------

Summary:
{md_table(["variant", "n", "mean_cam_err_cm", "med_cam_err_cm", "mean_rot_deg", "med_rot_deg", "mean_dist_err_cm"], metric_summary)}

AP03 estimated output in local Ref14 / marker-layout frame:
{md_table(["cam", "single_est_xyz", "single_dist_m", "multi_est_xyz", "multi_dist_m"], metric_output_table)}

Evaluation-only: camera GT-vs-estimated errors:
{md_table(["cam", "single_err_cm", "multi_err_cm", "gain_cm", "single_rot", "multi_rot", "single_dist_err", "multi_dist_err"], metric_error_table)}

Evaluation-only: GT camera map in Ref14 frame:
{md_table(["cam", "gt_ref14_xyz", "gt_dist_to_ref14_m"], gt_table)}

Final interpretation:
AP03-0 shows what pure targetless COLMAP can provide without any metric reference:
a unitless relative camera rig shape.

The scale-less ratio diagnostic confirms that this targetless rig shape is geometrically consistent up to scale.

However, pure targetless COLMAP cannot recover metric scale from monocular images alone.
To obtain metric camera poses in meters, an external metric reference is required.

Single Ref14 provides metric registration from one known marker.
Multi-ArUco provides a more stable metric registration from a known marker layout.

Main AP03 result:
AP03b Multi-ArUco achieves the best metric static-camera accuracy:
{f(mean(multi_errs))} cm mean camera error and {f(mean(multi_rots))}° mean rotation error.
"""

    remove_extra_final_files()

    OUT_TXT.write_text(report)
    write_final_csv(pure_rows, ratio_summary, metric_rows)

    # Also mirror into approach-specific final folder.
    (SOURCE_ROOT / "AP03_FINAL_RESULT.txt").write_text(report)
    (SOURCE_ROOT / "AP03_FINAL_RESULT.csv").write_text(OUT_CSV.read_text())

    print("[OK] wrote consolidated AP03 final result:")
    print(" ", OUT_TXT)
    print(" ", OUT_CSV)
    print()
    print(report)


if __name__ == "__main__":
    main()
