#!/usr/bin/env python3
from pathlib import Path
import json
import math

import cv2
import numpy as np

from run.bus_real_data._shared.common.constants import (
    STATIC_CAMERAS,
    REF_MARKER_ID,
    REF_MARKER_ENTITY,
    MARKER_LENGTH_M,
    WORLD_SDF_MOVING_CAMERA,
    ARUCO_DICT_NAME,
)
from run.bus_real_data._shared.common.io_utils import ensure_dir, read_csv, write_csv
from run.bus_real_data._shared.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    rvec_to_R,
    make_T,
    mean,
    median,
    umeyama_similarity,
    apply_sim3_point,
    apply_sim3_pose,
    trans_error_cm,
    rot_error_deg,
    metric_pose_columns,
)
from run.bus_real_data._shared.common.aruco_utils import (
    marker_object_points,
    make_aruco_detector,
)
from run.bus_real_data._shared.common.colmap_io import (
    parse_cameras_txt,
    parse_images_txt,
    parse_points3D_txt,
)
from run.bus_real_data._shared.common.projection import (
    robust_triangulate_point,
    reproj_errors_px,
)
from run.bus_real_data._shared.common.sdf_utils import (
    gt_static_camera_poses_ref_aruco,
)


AP3_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
TXT_ROOT = AP3_ROOT / "02_colmap_sparse" / "sparse_txt"
IMAGE_DIR = AP3_ROOT / "01_colmap_dataset" / "images"
INSPECT_SUMMARY = AP3_ROOT / "03_reconstruction_inspection" / "colmap_model_summary.csv"

OUT_ROOT = AP3_ROOT / "06_triangulated_ref_aruco_registration"
AP3_CMP = AP3_ROOT / "07_final_results"
COMBINED = AP3_ROOT / "07_final_results"


def pose_fields():
    return [
        "entity_type", "entity_id", "source",
        "x_m", "y_m", "z_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rvec_x", "rvec_y", "rvec_z",
    ]


def pose_row(entity_type, entity_id, T, source):
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source": source,
        "x_m": T[0, 3],
        "y_m": T[1, 3],
        "z_m": T[2, 3],
        "roll_deg": rpy[0],
        "pitch_deg": rpy[1],
        "yaw_deg": rpy[2],
        "rvec_x": rvec[0],
        "rvec_y": rvec[1],
        "rvec_z": rvec[2],
    }


def source_id_from_image_name(name):
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    if name.startswith("moving_") and name.endswith(".png"):
        return name[len("moving_"):-len(".png")]
    return name


def parse_best_model_name(inspect_summary: Path = INSPECT_SUMMARY):
    rows = read_csv(inspect_summary)
    rows = sorted(
        rows,
        key=lambda r: (
            int(r["registered_static_cameras"]),
            int(r["registered_images"]),
            int(r["num_3d_points"]),
        ),
        reverse=True,
    )
    return rows[0]["model"]


def get_best_model_dir(
    txt_root: Path = TXT_ROOT,
    inspect_summary: Path = INSPECT_SUMMARY,
):
    best_model = parse_best_model_name(inspect_summary)
    model_dir = txt_root / best_model
    if not model_dir.exists():
        raise RuntimeError(f"Missing COLMAP model dir: {model_dir}")
    return best_model, model_dir


def load_best_colmap_model(
    txt_root: Path = TXT_ROOT,
    inspect_summary: Path = INSPECT_SUMMARY,
):
    best_model, model_dir = get_best_model_dir(
        txt_root, inspect_summary
    )
    cameras = parse_cameras_txt(model_dir / "cameras.txt")
    images = parse_images_txt(model_dir / "images.txt")
    points3d = parse_points3D_txt(model_dir / "points3D.txt")
    return best_model, model_dir, cameras, images, points3d


def detect_ref14_observations(images, min_area_px2):
    detect = make_aruco_detector(ARUCO_DICT_NAME)
    obs = []

    for image_name in sorted(images.keys()):
        image_path = IMAGE_DIR / image_name
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        if ids is None or len(ids) == 0:
            continue

        ids = ids.reshape(-1)
        for idx, marker_id in enumerate(ids.tolist()):
            if int(marker_id) != int(REF_MARKER_ID):
                continue

            pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))

            if area < float(min_area_px2):
                continue

            for corner_idx in range(4):
                obs.append({
                    "image_name": image_name,
                    "corner_idx": corner_idx,
                    "u": float(pts[corner_idx, 0]),
                    "v": float(pts[corner_idx, 1]),
                    "area_px2": area,
                })

    return obs


def triangulate_ref14_corners(obs, images, cameras, ransac_iters, reproj_thresh_px, min_inliers):
    corner_results = []
    X_col = []

    for corner_idx in range(4):
        corner_obs = [o for o in obs if int(o["corner_idx"]) == corner_idx]
        res = robust_triangulate_point(
            corner_obs,
            images,
            cameras,
            ransac_iters=ransac_iters,
            reproj_thresh_px=reproj_thresh_px,
            min_inliers=min_inliers,
            random_seed=7,
        )

        X = res["X"]
        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_errs = [errs[i] for i in res["inlier_idx"]]

        X_col.append(X)
        corner_results.append({
            "corner_idx": corner_idx,
            "x_colmap": X[0],
            "y_colmap": X[1],
            "z_colmap": X[2],
            "obs_count": res["obs_count"],
            "inlier_count": res["inlier_count"],
            "mean_reproj_px": mean(inlier_errs) if inlier_errs else float("nan"),
            "median_reproj_px": median(inlier_errs) if inlier_errs else float("nan"),
            "max_reproj_px": max(inlier_errs) if inlier_errs else float("nan"),
        })

    return np.asarray(X_col, dtype=np.float64), corner_results


def compute_ref14_sim3(X_col):
    Y_ref = marker_object_points(MARKER_LENGTH_M)
    scale, R_ref_col, t_ref_col = umeyama_similarity(X_col, Y_ref)

    fit_rows = []
    fit_errors_cm = []

    for i in range(4):
        p_est = apply_sim3_point(X_col[i], scale, R_ref_col, t_ref_col)
        p_gt = Y_ref[i]
        err_cm = 100.0 * float(np.linalg.norm(p_est - p_gt))
        fit_errors_cm.append(err_cm)

        fit_rows.append({
            "corner_idx": i,
            "ideal_x_ref_m": p_gt[0],
            "ideal_y_ref_m": p_gt[1],
            "ideal_z_ref_m": p_gt[2],
            "registered_x_ref_m": p_est[0],
            "registered_y_ref_m": p_est[1],
            "registered_z_ref_m": p_est[2],
            "corner_fit_error_cm": err_cm,
        })

    return scale, R_ref_col, t_ref_col, fit_rows, fit_errors_cm


def save_sim3_metadata(path, scale, R_ref_col, t_ref_col, extra=None):
    data = {
        "scale_colmap_to_metric": float(scale),
        "R_ref_col": np.asarray(R_ref_col, dtype=float).tolist(),
        "t_ref_col": np.asarray(t_ref_col, dtype=float).reshape(3).tolist(),
    }
    if extra:
        data.update(extra)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_sim3_metadata(path):
    data = json.loads(Path(path).read_text())
    scale = float(data["scale_colmap_to_metric"])
    R_ref_col = np.asarray(data["R_ref_col"], dtype=np.float64)
    t_ref_col = np.asarray(data["t_ref_col"], dtype=np.float64).reshape(3)
    return data, scale, R_ref_col, t_ref_col


def apply_registration_to_colmap(images, points3d, scale, R_ref_col, t_ref_col):
    static_rows = []
    moving_rows = []

    for image_name, payload in sorted(images.items()):
        T_ref_cam = apply_sim3_pose(payload["T_col_cam"], scale, R_ref_col, t_ref_col)

        if image_name.startswith("static_"):
            static_rows.append(
                pose_row("static_camera", source_id_from_image_name(image_name), T_ref_cam, "ap03_repo_like_triangulated_ref14")
            )
        elif image_name.startswith("moving_"):
            moving_rows.append(
                pose_row("moving_frame", source_id_from_image_name(image_name), T_ref_cam, "ap03_repo_like_triangulated_ref14")
            )

    point_rows = []
    for p in points3d:
        p_ref = apply_sim3_point(
            np.array([p["x_colmap"], p["y_colmap"], p["z_colmap"]], dtype=np.float64),
            scale,
            R_ref_col,
            t_ref_col,
        )
        point_rows.append({
            "point3d_id": p["point3d_id"],
            "x_ref_aruco_m": p_ref[0],
            "y_ref_aruco_m": p_ref[1],
            "z_ref_aruco_m": p_ref[2],
            "r": p["r"],
            "g": p["g"],
            "b": p["b"],
            "colmap_reprojection_error": p["error"],
        })

    return static_rows, moving_rows, point_rows


def eval_static_cameras_vs_gt(static_rows):
    gt = gt_static_camera_poses_ref_aruco(WORLD_SDF_MOVING_CAMERA, STATIC_CAMERAS, REF_MARKER_ENTITY)
    est = {r["entity_id"]: r for r in static_rows}

    eval_rows = []

    for cam in STATIC_CAMERAS:
        if cam not in est:
            print(f"[WARN] Missing AP03 static camera estimate: {cam}; skipping GT evaluation for this camera.")
            continue

        r = est[cam]
        T_est = make_T(
            rvec_to_R([float(r["rvec_x"]), float(r["rvec_y"]), float(r["rvec_z"])]),
            [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])],
        )
        T_gt = gt[cam]

        row = {
            "approach": "AP03_targetless_colmap_repo_like_aruco_scale",
            "entity_type": "static_camera",
            "entity_id": cam,
            "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
            "translation_error_cm": trans_error_cm(T_est, T_gt),
            "rotation_error_deg": rot_error_deg(T_est, T_gt),
            "delta_x_cm": 100.0 * (T_est[0, 3] - T_gt[0, 3]),
            "delta_y_cm": 100.0 * (T_est[1, 3] - T_gt[1, 3]),
            "delta_z_cm": 100.0 * (T_est[2, 3] - T_gt[2, 3]),
        }
        row.update(metric_pose_columns("est_ref_aruco", T_est))
        row.update(metric_pose_columns("gt_ref_aruco", T_gt))
        eval_rows.append(row)

    return eval_rows


def eval_fields():
    return [
        "approach", "entity_type", "entity_id", "reference_frame",
        "translation_error_cm", "rotation_error_deg",
        "delta_x_cm", "delta_y_cm", "delta_z_cm",
        "est_ref_aruco_x_m", "est_ref_aruco_y_m", "est_ref_aruco_z_m",
        "est_ref_aruco_roll_deg", "est_ref_aruco_pitch_deg", "est_ref_aruco_yaw_deg",
        "est_ref_aruco_rvec_x", "est_ref_aruco_rvec_y", "est_ref_aruco_rvec_z",
        "gt_ref_aruco_x_m", "gt_ref_aruco_y_m", "gt_ref_aruco_z_m",
        "gt_ref_aruco_roll_deg", "gt_ref_aruco_pitch_deg", "gt_ref_aruco_yaw_deg",
        "gt_ref_aruco_rvec_x", "gt_ref_aruco_rvec_y", "gt_ref_aruco_rvec_z",
    ]
