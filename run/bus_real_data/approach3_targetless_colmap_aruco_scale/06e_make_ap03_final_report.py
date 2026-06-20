#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.io_utils import ensure_dir, read_csv, write_csv, format_table
from _shared.common.geometry import mean, median
from ap03_scale_common import (
    REF_MARKER_ID,
    MARKER_LENGTH_M,
    load_best_colmap_model,
    AP3_CMP,
    COMBINED,
)

DEFAULT_OUT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ensure_dir(args.out_root)
    ensure_dir(AP3_CMP)
    ensure_dir(COMBINED)

    try:
        best_model, model_dir, cameras, images, points3d = load_best_colmap_model()
    except RuntimeError as exc:
        print(f"[WARN] COLMAP TXT model unavailable for report counts: {exc}")
        print("[WARN] Reusing existing AP03 registration/evaluation outputs.")
        best_model = "existing_outputs"
        model_dir = None
        cameras, images, points3d = {}, {}, {}

    detect_meta = json.loads((args.out_root / "06a_detection_metadata.json").read_text())
    sim3_meta = json.loads((args.out_root / "ap03_ref14_sim3_metadata.json").read_text())

    corner_rows = read_csv(args.out_root / "ap03_triangulated_ref14_corners_colmap_and_registered.csv")
    static_pose_rows = read_csv(args.out_root / "ap03_static_camera_poses_ref_aruco.csv")
    moving_pose_rows = read_csv(args.out_root / "ap03_moving_frame_poses_ref_aruco.csv")
    eval_rows = read_csv(args.out_root / "ap03_static_cameras_ref_aruco_vs_gt.csv")

    cam_t = [float(r["translation_error_cm"]) for r in eval_rows]
    cam_r = [float(r["rotation_error_deg"]) for r in eval_rows]

    sparse_point_rows = (
        read_csv(args.out_root / "ap03_sparse_points3d_ref_aruco.csv")
        if (args.out_root / "ap03_sparse_points3d_ref_aruco.csv").exists()
        else []
    )
    registered_images_count = len(images) or (len(static_pose_rows) + len(moving_pose_rows))
    sparse_points3d_count = len(points3d) or len(sparse_point_rows)

    summary = [{
        "approach": "AP03_targetless_colmap_repo_like_aruco_scale",
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "final_variant": "colmap_then_triangulated_ref14_corner_sim3",
        "registered_images": registered_images_count,
        "registered_static_cameras": len(static_pose_rows),
        "registered_moving_frames": len(moving_pose_rows),
        "num_sparse_points3d": sparse_points3d_count,
        "ref14_corner_observation_count": detect_meta["ref14_corner_observation_count"],
        "ref14_unique_anchor_images": detect_meta["ref14_unique_anchor_images"],
        "min_area_px2": detect_meta["min_area_px2"],
        "estimated_colmap_to_metric_scale": sim3_meta["scale_colmap_to_metric"],
        "corner_fit_mean_error_cm": sim3_meta["corner_fit_mean_error_cm"],
        "corner_fit_median_error_cm": sim3_meta["corner_fit_median_error_cm"],
        "corner_triangulation_mean_reproj_px": sim3_meta["corner_triangulation_mean_reproj_px"],
        "corner_triangulation_median_reproj_px": sim3_meta["corner_triangulation_median_reproj_px"],
        "camera_mean_translation_error_cm": mean(cam_t),
        "camera_median_translation_error_cm": median(cam_t),
        "camera_mean_rotation_error_deg": mean(cam_r),
        "camera_median_rotation_error_deg": median(cam_r),
    }]

    write_csv(COMBINED / "ap03_repo_like_final_summary.csv", summary, list(summary[0].keys()))

    cam_simple = []
    for r in eval_rows:
        cam_simple.append({
            "camera": r["entity_id"],
            "translation_error_cm": f"{float(r['translation_error_cm']):.3f}",
            "rotation_error_deg": f"{float(r['rotation_error_deg']):.3f}",
            "delta_x_cm": f"{float(r['delta_x_cm']):.3f}",
            "delta_y_cm": f"{float(r['delta_y_cm']):.3f}",
            "delta_z_cm": f"{float(r['delta_z_cm']):.3f}",
            "estimated_xyz_m": f"({float(r['est_ref_aruco_x_m']):+.3f}, {float(r['est_ref_aruco_y_m']):+.3f}, {float(r['est_ref_aruco_z_m']):+.3f})",
            "gt_xyz_m": f"({float(r['gt_ref_aruco_x_m']):+.3f}, {float(r['gt_ref_aruco_y_m']):+.3f}, {float(r['gt_ref_aruco_z_m']):+.3f})",
        })

    cam_table = format_table(
        cam_simple,
        ["camera", "t_err_cm", "r_err_deg", "dX", "dY", "dZ", "estimated xyz [m]", "GT xyz [m]"],
        ["camera", "translation_error_cm", "rotation_error_deg", "delta_x_cm", "delta_y_cm", "delta_z_cm", "estimated_xyz_m", "gt_xyz_m"],
    )

    corner_simple = []
    for r in corner_rows:
        corner_simple.append({
            "corner": r["corner_idx"],
            "obs": r["obs_count"],
            "inliers": r["inlier_count"],
            "mean_reproj_px": f"{float(r['mean_reproj_px']):.3f}",
            "median_reproj_px": f"{float(r['median_reproj_px']):.3f}",
            "fit_error_cm": f"{float(r['corner_fit_error_cm']):.4f}",
            "registered_xyz_m": f"({float(r['registered_x_ref_m']):+.4f}, {float(r['registered_y_ref_m']):+.4f}, {float(r['registered_z_ref_m']):+.4f})",
            "ideal_xyz_m": f"({float(r['ideal_x_ref_m']):+.4f}, {float(r['ideal_y_ref_m']):+.4f}, {float(r['ideal_z_ref_m']):+.4f})",
        })

    corner_table = format_table(
        corner_simple,
        ["corner", "obs", "inliers", "mean_repr", "med_repr", "fit_cm", "registered xyz [m]", "ideal xyz [m]"],
        ["corner", "obs", "inliers", "mean_reproj_px", "median_reproj_px", "fit_error_cm", "registered_xyz_m", "ideal_xyz_m"],
    )

    report = f"""AP03 FINAL REPO-LIKE SCALE REGISTRATION REPORT
=======================================================

Approach:
- AP03: Targetless COLMAP / SfM + ArUco-based metric scale registration.
- This final pipeline is split into explicit stages:
  06a detection, 06b triangulation/Sim3, 06c application, 06d GT eval, 06e report.
- The method follows the aruco-estimator-style idea:
  triangulate Ref-ArUco marker corners in COLMAP coordinates,
  then map them to ideal metric marker corners with known side length.

Input COLMAP model:
- model: {best_model}
- registered images: {registered_images_count}
- registered static cameras: {len(static_pose_rows)} / 4
- registered moving frames: {len(moving_pose_rows)}
- sparse 3D points: {sparse_points3d_count}

Ref-ArUco scale registration:
- reference marker id: {REF_MARKER_ID}
- marker length: {MARKER_LENGTH_M:.3f} m
- min marker area used: {float(detect_meta['min_area_px2']):.1f} px^2
- Ref14 corner observations: {detect_meta['ref14_corner_observation_count']}
- Ref14 anchor images: {detect_meta['ref14_unique_anchor_images']}
- estimated COLMAP-to-meter scale: {float(sim3_meta['scale_colmap_to_metric']):.9f}

Triangulated Ref14 corner quality:
- corner fit mean error: {float(sim3_meta['corner_fit_mean_error_cm']):.6f} cm
- corner fit median error: {float(sim3_meta['corner_fit_median_error_cm']):.6f} cm
- mean corner reprojection error: {float(sim3_meta['corner_triangulation_mean_reproj_px']):.6f} px
- median corner reprojection error: {float(sim3_meta['corner_triangulation_median_reproj_px']):.6f} px

{corner_table}

Static camera GT evaluation in Ref-ArUco frame:
- mean translation error: {mean(cam_t):.6f} cm
- median translation error: {median(cam_t):.6f} cm
- mean rotation error: {mean(cam_r):.6f} deg
- median rotation error: {median(cam_r):.6f} deg

{cam_table}
"""

    (args.out_root / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)
    (AP3_CMP / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)

    print(report)


if __name__ == "__main__":
    main()
