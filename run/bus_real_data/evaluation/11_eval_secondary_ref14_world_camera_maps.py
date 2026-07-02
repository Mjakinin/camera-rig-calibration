#!/usr/bin/env python3
import csv
import math
import json
from pathlib import Path

import cv2
import numpy as np

from _shared.common.sdf_utils import gt_static_camera_poses_ref_aruco
from _shared.common.geometry import R_to_rpy_deg
from _shared.common.constants import WORLD_SDF_MOVING_CAMERA, STATIC_CAMERAS as CONST_STATIC_CAMERAS, REF_MARKER_ENTITY


STATIC_CAMS = list(CONST_STATIC_CAMERAS)
FINAL = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")

METHOD_CANDIDATES = {
    "AP01": [
        "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/final_extrinsics_summary.csv",
        "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/final_static_camera_poses.csv",
    ],
    "AP02": [
        "results/bus_real_data/02_ref_marker_graph_ba/08_final_results/ap02_with_moving_static_camera_poses_ref_marker.csv",
        "results/bus_real_data/02_ref_marker_graph_ba/08_final_results/optimized_static_camera_poses_ref_marker.csv",
    ],
    "AP03": [
        "results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_ONLY_STATIC_CAMERA_POSES.csv",
        "results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
        "results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/ap03_marker_size_only_static_camera_poses.csv",
        "results/bus_real_data/03_targetless_colmap_aruco_scale/07_final_results/ap03_scaled_static_camera_poses.csv",
    ],
}


def norm_cam_name(v):
    if v is None:
        return None
    s = str(v).strip()
    s = s.replace("static_", "")
    s = s.replace(".png", "")
    s = s.replace(".jpg", "")
    for cam in STATIC_CAMS:
        if cam in s:
            return cam
    return None


def as_float(row, names):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return float(row[n])
    return None


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def quat_to_R(qx, qy, qz, qw):
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=float)


def parse_R(row):
    # rotation matrix columns
    mat_names = [
        ["r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"],
        ["R_00", "R_01", "R_02", "R_10", "R_11", "R_12", "R_20", "R_21", "R_22"],
    ]
    for names in mat_names:
        if all(n in row and str(row[n]).strip() != "" for n in names):
            vals = [float(row[n]) for n in names]
            return np.array(vals, dtype=float).reshape(3, 3)

    # quaternion
    qx = as_float(row, ["qx", "q_x", "quat_x"])
    qy = as_float(row, ["qy", "q_y", "quat_y"])
    qz = as_float(row, ["qz", "q_z", "quat_z"])
    qw = as_float(row, ["qw", "q_w", "quat_w"])
    if None not in (qx, qy, qz, qw):
        return quat_to_R(qx, qy, qz, qw)

    # Rodrigues
    rx = as_float(row, ["rvec_x", "rx", "rotvec_x"])
    ry = as_float(row, ["rvec_y", "ry", "rotvec_y"])
    rz = as_float(row, ["rvec_z", "rz", "rotvec_z"])
    if None not in (rx, ry, rz):
        R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=float))
        return R

    # RPY deg/rad
    roll = as_float(row, ["roll", "roll_rad", "roll_deg", "rpy_roll", "est_roll_deg"])
    pitch = as_float(row, ["pitch", "pitch_rad", "pitch_deg", "rpy_pitch", "est_pitch_deg"])
    yaw = as_float(row, ["yaw", "yaw_rad", "yaw_deg", "rpy_yaw", "est_yaw_deg"])
    if None not in (roll, pitch, yaw):
        deg_hint = any(k in row for k in ["roll_deg", "pitch_deg", "yaw_deg", "est_roll_deg", "est_pitch_deg", "est_yaw_deg"])
        if deg_hint or max(abs(roll), abs(pitch), abs(yaw)) > 2 * math.pi + 1e-6:
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
        return rpy_to_R(roll, pitch, yaw)

    raise RuntimeError(f"Could not parse rotation from columns: {list(row.keys())}")


def parse_pose_csv(path):
    path = Path(path)
    if not path.exists():
        return None

    rows = list(csv.DictReader(path.open()))
    poses = {}

    for row in rows:
        if "category" in row and str(row.get("category", "")).strip() not in ("", "main_no_gt"):
            continue

        cam = None
        for key in ["camera", "camera_name", "cam", "target_camera", "entity_id", "observer_id", "image", "image_name", "filename", "name"]:
            if key in row:
                cam = norm_cam_name(row.get(key))
                if cam:
                    break
        if cam not in STATIC_CAMS:
            continue

        x = as_float(row, ["x", "x_m", "tx", "t_x", "tvec_x", "tvec_x_m", "position_x", "est_x_m"])
        y = as_float(row, ["y", "y_m", "ty", "t_y", "tvec_y", "tvec_y_m", "position_y", "est_y_m"])
        z = as_float(row, ["z", "z_m", "tz", "t_z", "tvec_z", "tvec_z_m", "position_z", "est_z_m"])
        if None in (x, y, z):
            continue

        R = parse_R(row)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        poses[cam] = T

    if "final_extrinsics_summary.csv" in str(path) and "cam_edge_3" not in poses:
        poses["cam_edge_3"] = np.eye(4)

    if len(poses) < 2:
        return None
    return poses


def discover_pose_file(method):
    candidates = [Path(p) for p in METHOD_CANDIDATES[method]]

    # Fallback discovery for AP03/AP01 if names differ.
    if method == "AP03":
        root = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
        if root.exists():
            candidates += sorted(root.rglob("*camera*pose*.csv"))
            candidates += sorted(root.rglob("*static*camera*.csv"))

    if method == "AP01":
        root = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain")
        if root.exists():
            candidates += sorted(root.rglob("*extrinsic*.csv"))
            candidates += sorted(root.rglob("*camera*pose*.csv"))

    if method == "AP02":
        root = Path("results/bus_real_data/02_ref_marker_graph_ba")
        if root.exists():
            candidates += sorted(root.rglob("*static*camera*pose*.csv"))

    tried = []
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        tried.append(str(p))
        try:
            poses = parse_pose_csv(p)
        except Exception as e:
            tried.append(f"  parse_failed: {p}: {e}")
            continue
        if poses and all(cam in poses for cam in STATIC_CAMS):
            return p, poses
        if poses:
            tried.append(f"  partial {p}: {sorted(poses)}")

    raise RuntimeError(
        f"Could not find complete static camera pose CSV for {method}.\n"
        f"Tried:\n" + "\n".join(tried[:80])
    )


def rot_error_deg(R_est, R_gt):
    R = R_gt.T @ R_est
    v = (np.trace(R) - 1.0) / 2.0
    v = max(-1.0, min(1.0, float(v)))
    return math.degrees(math.acos(v))


def estimate_world_alignment(est_poses, gt_poses, cams):
    # Estimate R_A from rotations: R_gt ~= R_A @ R_est
    M = np.zeros((3, 3))
    for cam in cams:
        M += gt_poses[cam][:3, :3] @ est_poses[cam][:3, :3].T
    U, _, Vt = np.linalg.svd(M)
    R_A = U @ Vt
    if np.linalg.det(R_A) < 0:
        U[:, -1] *= -1
        R_A = U @ Vt

    # Translation from camera centers.
    t_terms = []
    for cam in cams:
        t_gt = gt_poses[cam][:3, 3]
        t_est = est_poses[cam][:3, 3]
        t_terms.append(t_gt - R_A @ t_est)
    t_A = np.mean(np.vstack(t_terms), axis=0)

    T_A = np.eye(4)
    T_A[:3, :3] = R_A
    T_A[:3, 3] = t_A
    return T_A


def apply_A(T_A, T):
    out = np.eye(4)
    out[:3, :3] = T_A[:3, :3] @ T[:3, :3]
    out[:3, 3] = T_A[:3, :3] @ T[:3, 3] + T_A[:3, 3]
    return out


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    gt = gt_static_camera_poses_ref_aruco(WORLD_SDF_MOVING_CAMERA, STATIC_CAMS, REF_MARKER_ENTITY)

    detail_rows = []
    summary_rows = []
    report = []

    report.append("SECONDARY REF14/WORLD-FRAME CAMERA-MAP VS GT EVALUATION")
    report.append("======================================================")
    report.append("")
    report.append("What this evaluates:")
    report.append("- estimated static-camera map vs GT static-camera map")
    report.append("- each method map is aligned to the GT Ref14/world camera map with a best-fit SE(3) transform")
    report.append("- no scale is estimated in this secondary evaluation")
    report.append("- GT is used only after method estimation for evaluation")
    report.append("- this is secondary; primary remains pairwise camera-to-camera extrinsics")
    report.append("")

    for method in ["AP01", "AP02", "AP03"]:
        try:
            src, est = discover_pose_file(method)
            cams = [c for c in STATIC_CAMS if c in est and c in gt]
            if len(cams) < 3:
                raise RuntimeError(f"{method}: need at least 3 cameras for SE3 alignment, got {cams}")
        except Exception as e:
            print(f"[WARN] skipping {method} in secondary evaluator: {e}")
            report.append("")
            report.append(f"{method}")
            report.append("-" * len(method))
            report.append(f"status: FAILED_MISSING_SOURCE")
            report.append(f"reason: {e}")
            summary_rows.append({
                "method": method,
                "mean_translation_error_cm": "",
                "mean_rotation_error_deg": "",
                "median_translation_error_cm": "",
                "median_rotation_error_deg": "",
                "max_translation_error_cm": "",
                "max_rotation_error_deg": "",
                "status": "FAILED_MISSING_SOURCE",
                "source_file": "",
                "camera_count": 0,
                "alignment": "SKIPPED",
            })
            continue

        T_A = estimate_world_alignment(est, gt, cams)

        errs_t = []
        errs_r = []

        report.append("")
        report.append(f"{method}")
        report.append("-" * len(method))
        report.append(f"source: {src}")
        report.append(f"alignment cameras: {', '.join(cams)}")

        for cam in STATIC_CAMS:
            T_est_aligned = apply_A(T_A, est[cam])
            T_gt = gt[cam]

            t_err_cm = float(np.linalg.norm(T_est_aligned[:3, 3] - T_gt[:3, 3]) * 100.0)
            r_err_deg = float(rot_error_deg(T_est_aligned[:3, :3], T_gt[:3, :3]))

            errs_t.append(t_err_cm)
            errs_r.append(r_err_deg)

            erpy = R_to_rpy_deg(T_est_aligned[:3, :3])
            grpy = R_to_rpy_deg(T_gt[:3, :3])

            detail_rows.append({
                "method": method,
                "camera": cam,
                "alignment": "SE3_all_static_cameras_no_scale",
                "source_file": str(src),
                "translation_error_cm": f"{t_err_cm:.6f}",
                "rotation_error_deg": f"{r_err_deg:.6f}",
                "aligned_est_x_m": f"{T_est_aligned[0,3]:.9f}",
                "aligned_est_y_m": f"{T_est_aligned[1,3]:.9f}",
                "aligned_est_z_m": f"{T_est_aligned[2,3]:.9f}",
                "gt_x_m": f"{T_gt[0,3]:.9f}",
                "gt_y_m": f"{T_gt[1,3]:.9f}",
                "gt_z_m": f"{T_gt[2,3]:.9f}",
                "aligned_est_roll_deg": f"{erpy[0]:.9f}",
                "aligned_est_pitch_deg": f"{erpy[1]:.9f}",
                "aligned_est_yaw_deg": f"{erpy[2]:.9f}",
                "gt_roll_deg": f"{grpy[0]:.9f}",
                "gt_pitch_deg": f"{grpy[1]:.9f}",
                "gt_yaw_deg": f"{grpy[2]:.9f}",
            })

            report.append(
                f"- {cam}: {t_err_cm:.3f} cm, {r_err_deg:.3f} deg | "
                f"aligned_est=({T_est_aligned[0,3]:+.3f}, {T_est_aligned[1,3]:+.3f}, {T_est_aligned[2,3]:+.3f}) m | "
                f"gt=({T_gt[0,3]:+.3f}, {T_gt[1,3]:+.3f}, {T_gt[2,3]:+.3f}) m"
            )

        summary_rows.append({
            "method": method,
            "status": "OK",
            "alignment": "SE3_all_static_cameras_no_scale",
            "source_file": str(src),
            "camera_count": len(STATIC_CAMS),
            "mean_translation_error_cm": f"{np.mean(errs_t):.6f}",
            "median_translation_error_cm": f"{np.median(errs_t):.6f}",
            "max_translation_error_cm": f"{np.max(errs_t):.6f}",
            "mean_rotation_error_deg": f"{np.mean(errs_r):.6f}",
            "median_rotation_error_deg": f"{np.median(errs_r):.6f}",
            "max_rotation_error_deg": f"{np.max(errs_r):.6f}",
        })

        report.append(
            f"summary: mean {np.mean(errs_t):.3f} cm / {np.mean(errs_r):.3f} deg, "
            f"median {np.median(errs_t):.3f} cm / {np.median(errs_r):.3f} deg"
        )

    detail_csv = FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv"
    summary_csv = FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv"
    report_txt = FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt"
    meta_json = FINAL / "SECONDARY_REF14_WORLD_CAMERA_MAP_METADATA.json"

    with detail_csv.open("w", newline="") as f:
        fieldnames = sorted({k for row in detail_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(detail_rows)

    with summary_csv.open("w", newline="") as f:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    report.append("")
    report.append("Output files:")
    report.append(f"- {detail_csv}")
    report.append(f"- {summary_csv}")
    report.append(f"- {report_txt}")
    report.append(f"- {meta_json}")

    report_txt.write_text("\n".join(report) + "\n")

    meta_json.write_text(json.dumps({
        "evaluation": "secondary_ref14_world_camera_map_vs_gt",
        "alignment": "SE3_all_static_cameras_no_scale",
        "gt_used_only_for_evaluation": True,
        "primary_metric_remains": "pairwise_static_camera_to_camera_extrinsics",
        "static_cameras": STATIC_CAMS,
        "summary_rows": summary_rows,
    }, indent=2) + "\n")

    print("[OK] wrote:", detail_csv)
    print("[OK] wrote:", summary_csv)
    print("[OK] wrote:", report_txt)
    print("[OK] wrote:", meta_json)
    print()
    print(report_txt.read_text())


if __name__ == "__main__":
    main()
