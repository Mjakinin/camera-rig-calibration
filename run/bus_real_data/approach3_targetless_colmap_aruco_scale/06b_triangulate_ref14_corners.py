#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.io_utils import ensure_dir, read_csv, write_csv
from _shared.common.geometry import mean, median
from ap03_scale_common import (
    load_best_colmap_model,
    triangulate_ref14_corners,
    compute_ref14_sim3,
    save_sim3_metadata,
    REF_MARKER_ID,
    MARKER_LENGTH_M,
)

DEFAULT_OUT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reproj-thresh-px", type=float, default=5.0)
    ap.add_argument("--ransac-iters", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.out_root)

    obs_path = args.out_root / "ap03_ref14_corner_observations.csv"
    obs = read_csv(obs_path)

    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()

    X_col, corner_results = triangulate_ref14_corners(
        obs,
        images,
        cameras,
        ransac_iters=args.ransac_iters,
        reproj_thresh_px=args.reproj_thresh_px,
        min_inliers=args.min_inliers,
    )

    scale, R_ref_col, t_ref_col, fit_rows, fit_errors_cm = compute_ref14_sim3(X_col)

    combined = []
    for c, f in zip(corner_results, fit_rows):
        row = dict(c)
        row.update(f)
        combined.append(row)

    write_csv(
        args.out_root / "ap03_triangulated_ref14_corners_colmap_and_registered.csv",
        combined,
        [
            "corner_idx",
            "x_colmap", "y_colmap", "z_colmap",
            "obs_count", "inlier_count",
            "mean_reproj_px", "median_reproj_px", "max_reproj_px",
            "ideal_x_ref_m", "ideal_y_ref_m", "ideal_z_ref_m",
            "registered_x_ref_m", "registered_y_ref_m", "registered_z_ref_m",
            "corner_fit_error_cm",
        ],
    )

    save_sim3_metadata(
        args.out_root / "ap03_ref14_sim3_metadata.json",
        scale,
        R_ref_col,
        t_ref_col,
        extra={
            "stage": "06b_triangulate_ref14_corners",
            "best_colmap_model": best_model,
            "reference_marker_id": REF_MARKER_ID,
            "marker_length_m": MARKER_LENGTH_M,
            "reproj_thresh_px": args.reproj_thresh_px,
            "ransac_iters": args.ransac_iters,
            "min_inliers": args.min_inliers,
            "corner_fit_errors_cm": fit_errors_cm,
            "corner_fit_mean_error_cm": mean(fit_errors_cm),
            "corner_fit_median_error_cm": median(fit_errors_cm),
            "corner_triangulation_mean_reproj_px": mean([r["mean_reproj_px"] for r in corner_results]),
            "corner_triangulation_median_reproj_px": median([r["median_reproj_px"] for r in corner_results]),
        },
    )

    print("AP03 06b complete")
    print(f"scale_colmap_to_metric: {scale:.12f}")
    print(f"corner_fit_mean_error_cm: {mean(fit_errors_cm):.6f}")
    print(f"corner_reproj_mean_px: {mean([r['mean_reproj_px'] for r in corner_results]):.6f}")
    print(f"out: {args.out_root}")


if __name__ == "__main__":
    main()
