#!/usr/bin/env python3
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


MOVING_DET = Path("results/bus_real_data/03_moving_camera_sequence/moving_detections.csv")
COLMAP_IMAGES = Path("results/bus_real_data/04_colmap_moving_sequence/sparse_txt_best/images.txt")

OUT = Path("results/bus_real_data/04_colmap_moving_sequence/aruco_metric_scale")
OUT.mkdir(parents=True, exist_ok=True)

# Documented marker geometry:
# A4 sheet: 0.210 x 0.297 m
# ArUco marker: 0.170 x 0.170 m
MARKER_SIZE_M = 0.170

W, H = 1280.0, 720.0
CX, CY = W / 2.0, H / 2.0
HALF_DIAG = math.sqrt(CX * CX + CY * CY)


def pnp_ok(row):
    val = str(row.get("pnp_success", "True")).strip().lower()
    return val not in ("false", "0", "no", "none", "nan")


def f(row, key):
    return float(row[key])


def marker_area_px(row):
    pts = np.array([
        [f(row, "corner0_u"), f(row, "corner0_v")],
        [f(row, "corner1_u"), f(row, "corner1_v")],
        [f(row, "corner2_u"), f(row, "corner2_v")],
        [f(row, "corner3_u"), f(row, "corner3_v")],
    ], dtype=np.float32)
    return float(cv2.contourArea(pts))


def center_error_norm(row):
    du = f(row, "center_u") - CX
    dv = f(row, "center_v") - CY
    return math.sqrt(du * du + dv * dv) / HALF_DIAG


def T_from_R_t(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_from_detection(row):
    rvec = np.array([
        f(row, "rvec_x"),
        f(row, "rvec_y"),
        f(row, "rvec_z"),
    ], dtype=float)

    tvec = np.array([
        f(row, "tvec_x_m"),
        f(row, "tvec_y_m"),
        f(row, "tvec_z_m"),
    ], dtype=float)

    R, _ = cv2.Rodrigues(rvec)
    return T_from_R_t(R, tvec)


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ], dtype=float)


def load_colmap_poses():
    poses = {}

    if not COLMAP_IMAGES.exists():
        raise FileNotFoundError(COLMAP_IMAGES)

    for line in COLMAP_IMAGES.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "frame_" not in line or ".png" not in line:
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        name = parts[-1]
        m = re.search(r"frame_(\d+)\.png", name)
        if not m:
            continue

        frame = int(m.group(1))
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]

        R = qvec_to_rotmat(qvec)
        T_cw = T_from_R_t(R, tvec)  # COLMAP world -> camera
        poses[frame] = T_cw

    return poses


def load_observations(colmap_poses):
    by_marker = defaultdict(list)

    with MOVING_DET.open() as fp:
        for row in csv.DictReader(fp):
            if not pnp_ok(row):
                continue

            frame = int(row["frame"])
            marker_id = int(row["marker_id"])

            if frame not in colmap_poses:
                continue

            area = marker_area_px(row)
            center_norm = center_error_norm(row)
            dist_m = f(row, "distance_m")

            # Basic quality filter: ignore tiny/far/edge observations for scale.
            if area < 1200.0:
                continue
            if dist_m > 4.0:
                continue
            if center_norm > 0.95:
                continue

            row["_frame"] = frame
            row["_marker_id"] = marker_id
            row["_area_px"] = area
            row["_center_norm"] = center_norm
            row["_T_cam_marker"] = T_from_detection(row)

            by_marker[marker_id].append(row)

    for marker_id in by_marker:
        by_marker[marker_id].sort(key=lambda r: r["_frame"])

    return by_marker


def robust_median_scale(pairs):
    ratios = np.array([p["scale_ratio"] for p in pairs], dtype=float)

    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))

    if mad <= 1e-12:
        trimmed = ratios
        kept = pairs
    else:
        robust_sigma = 1.4826 * mad
        lo = median - 3.0 * robust_sigma
        hi = median + 3.0 * robust_sigma

        kept = [p for p in pairs if lo <= p["scale_ratio"] <= hi]
        trimmed = np.array([p["scale_ratio"] for p in kept], dtype=float)

        if len(trimmed) < max(10, 0.3 * len(ratios)):
            kept = pairs
            trimmed = ratios

    return {
        "scale": float(np.median(trimmed)),
        "mean": float(np.mean(trimmed)),
        "std": float(np.std(trimmed)),
        "raw_median": median,
        "raw_mad": mad,
        "num_raw": len(ratios),
        "num_kept": len(trimmed),
        "kept_pairs": kept,
    }


def main():
    colmap_poses = load_colmap_poses()
    by_marker = load_observations(colmap_poses)

    pairs = []

    for marker_id, obs in by_marker.items():
        n = len(obs)
        if n < 2:
            continue

        for a in range(n):
            for b in range(a + 1, n):
                ra = obs[a]
                rb = obs[b]

                fa = int(ra["_frame"])
                fb = int(rb["_frame"])
                gap = abs(fb - fa)

                # Avoid nearly identical frames and very long/disconnected jumps.
                if gap < 3:
                    continue
                if gap > 45:
                    continue

                T_a_marker = ra["_T_cam_marker"]
                T_b_marker = rb["_T_cam_marker"]

                # Metric relative motion from known-size ArUco PnP:
                # T_a_b maps camera_b coordinates into camera_a coordinates.
                T_a_b_metric = T_a_marker @ invT(T_b_marker)
                d_metric = float(np.linalg.norm(T_a_b_metric[:3, 3]))

                Tcw_a = colmap_poses[fa]
                Tcw_b = colmap_poses[fb]

                # COLMAP relative motion in camera coordinates, up to scale.
                T_a_b_colmap = Tcw_a @ invT(Tcw_b)
                d_colmap = float(np.linalg.norm(T_a_b_colmap[:3, 3]))

                if d_metric < 0.12:
                    continue
                if d_metric > 5.0:
                    continue
                if d_colmap < 1e-9:
                    continue

                ratio = d_metric / d_colmap

                if not np.isfinite(ratio):
                    continue
                if ratio <= 0:
                    continue

                pair_quality = math.sqrt(float(ra["_area_px"]) * float(rb["_area_px"]))

                pairs.append({
                    "marker_id": marker_id,
                    "frame_i": fa,
                    "frame_j": fb,
                    "frame_gap": gap,
                    "metric_translation_m": d_metric,
                    "colmap_translation_raw": d_colmap,
                    "scale_ratio": ratio,
                    "area_i_px": float(ra["_area_px"]),
                    "area_j_px": float(rb["_area_px"]),
                    "center_i_norm": float(ra["_center_norm"]),
                    "center_j_norm": float(rb["_center_norm"]),
                    "dist_i_m": float(ra["distance_m"]),
                    "dist_j_m": float(rb["distance_m"]),
                    "pair_quality": pair_quality,
                })

    if not pairs:
        raise RuntimeError("No valid ArUco/COLMAP overlap pairs found for metric scale estimation.")

    stats = robust_median_scale(pairs)
    scale = stats["scale"]

    scale_file = OUT / "metric_scale.txt"
    scale_file.write_text(f"{scale:.12g}\n")

    fields = [
        "marker_id", "frame_i", "frame_j", "frame_gap",
        "metric_translation_m", "colmap_translation_raw", "scale_ratio",
        "area_i_px", "area_j_px",
        "center_i_norm", "center_j_norm",
        "dist_i_m", "dist_j_m",
        "pair_quality",
    ]

    raw_csv = OUT / "scale_pairs_all.csv"
    with raw_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for p in pairs:
            w.writerow(p)

    kept_csv = OUT / "scale_pairs_kept_after_mad.csv"
    with kept_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for p in stats["kept_pairs"]:
            w.writerow(p)

    per_marker = defaultdict(list)
    for p in stats["kept_pairs"]:
        per_marker[p["marker_id"]].append(p["scale_ratio"])

    report = []
    report.append("No-GT COLMAP metric scale from ArUco overlap")
    report.append("============================================")
    report.append("")
    report.append("Scale source: known ArUco marker side length, not Gazebo GT.")
    report.append(f"Known marker side length [m]: {MARKER_SIZE_M:.3f}")
    report.append("")
    report.append(f"COLMAP registered poses loaded: {len(colmap_poses)}")
    report.append(f"Markers with usable observations: {len(by_marker)}")
    report.append(f"Raw scale pairs: {stats['num_raw']}")
    report.append(f"Kept scale pairs after robust MAD trim: {stats['num_kept']}")
    report.append("")
    report.append(f"Metric COLMAP scale: {scale:.12g}")
    report.append(f"Kept mean: {stats['mean']:.12g}")
    report.append(f"Kept std: {stats['std']:.12g}")
    report.append(f"Raw median: {stats['raw_median']:.12g}")
    report.append(f"Raw MAD: {stats['raw_mad']:.12g}")
    report.append("")
    report.append("Per-marker kept scale medians:")
    for marker_id in sorted(per_marker):
        vals = np.array(per_marker[marker_id], dtype=float)
        report.append(
            f"  marker {marker_id:02d}: "
            f"n={len(vals):4d}, median={float(np.median(vals)):.12g}, "
            f"mean={float(np.mean(vals)):.12g}, std={float(np.std(vals)):.12g}"
        )
    report.append("")
    report.append(f"Wrote: {scale_file}")
    report.append(f"Wrote: {raw_csv}")
    report.append(f"Wrote: {kept_csv}")

    report_txt = OUT / "scale_report.txt"
    report_txt.write_text("\n".join(report) + "\n")

    print(report_txt.read_text())


if __name__ == "__main__":
    main()
