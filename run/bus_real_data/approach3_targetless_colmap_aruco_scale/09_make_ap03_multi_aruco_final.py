#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import (
    STATIC_CAMERAS,
    REF_MARKER_ID,
    REF_MARKER_ENTITY,
    MARKER_LENGTH_M,
    WORLD_SDF_MOVING_CAMERA,
    ARUCO_DICT_NAME,
)
from _shared.common.aruco_utils import make_aruco_detector, marker_object_points
from _shared.common.geometry import (
    make_T,
    invT,
    mean,
    median,
    umeyama_similarity,
    apply_sim3_point,
    apply_sim3_pose,
    rvec_to_R,
    R_to_rpy_deg,
    R_to_rvec,
    metric_pose_columns,
    trans_error_cm,
    rot_error_deg,
)
from _shared.common.io_utils import ensure_dir, read_csv, write_csv
from _shared.common.projection import robust_triangulate_point, reproj_errors_px
from _shared.common.sdf_utils import (
    parse_world_poses,
    sdf_marker_model_to_opencv_frame,
    gt_static_camera_poses_ref_aruco,
)
from ap03_scale_common import (
    load_best_colmap_model,
    IMAGE_DIR,
    AP3_CMP,
    COMBINED,
)

AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
OUT_ROOT = AP03_ROOT / "06_triangulated_ref_aruco_registration"
FINAL_ROOT = AP03_ROOT / "07_final_results"


def marker_entity(marker_id: int) -> str:
    marker_id = int(marker_id)
    if marker_id == int(REF_MARKER_ID):
        return REF_MARKER_ENTITY
    return f"marker_{marker_id:03d}"


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


def detect_all_aruco_observations(images, min_area_px2, marker_ids):
    detect = make_aruco_detector(ARUCO_DICT_NAME)
    marker_ids = {int(x) for x in marker_ids}

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
            marker_id = int(marker_id)
            if marker_id not in marker_ids:
                continue

            pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area < float(min_area_px2):
                continue

            for corner_idx in range(4):
                obs.append({
                    "image_name": image_name,
                    "marker_id": marker_id,
                    "corner_idx": corner_idx,
                    "u": float(pts[corner_idx, 0]),
                    "v": float(pts[corner_idx, 1]),
                    "area_px2": area,
                })

    return obs


def load_known_marker_corner_targets_ref14(marker_ids):
    poses = parse_world_poses(WORLD_SDF_MOVING_CAMERA)
    ref_name = marker_entity(REF_MARKER_ID)
    if ref_name not in poses:
        raise RuntimeError(f"Missing reference marker in SDF: {ref_name}")

    T_W_ref = sdf_marker_model_to_opencv_frame(poses[ref_name]["T_W_model"])
    T_ref_W = invT(T_W_ref)

    local_corners = marker_object_points(MARKER_LENGTH_M)
    targets = {}

    for marker_id in marker_ids:
        name = marker_entity(marker_id)
        if name not in poses:
            print(f"[WARN] marker {marker_id} / {name} missing in SDF, skipping")
            continue

        T_W_marker = sdf_marker_model_to_opencv_frame(poses[name]["T_W_model"])
        T_ref_marker = T_ref_W @ T_W_marker

        for corner_idx in range(4):
            p_local = np.ones(4, dtype=np.float64)
            p_local[:3] = local_corners[corner_idx]
            p_ref = T_ref_marker @ p_local
            targets[(int(marker_id), int(corner_idx))] = p_ref[:3].copy()

    return targets


def triangulate_all_marker_corners(obs, images, cameras, targets, ransac_iters, reproj_thresh_px, min_inliers):
    X_col = []
    Y_ref = []
    rows = []

    for key in sorted(targets.keys()):
        marker_id, corner_idx = key
        corner_obs = [
            o for o in obs
            if int(o["marker_id"]) == marker_id and int(o["corner_idx"]) == corner_idx
        ]

        if len(corner_obs) < min_inliers:
            continue

        try:
            res = robust_triangulate_point(
                corner_obs,
                images,
                cameras,
                ransac_iters=ransac_iters,
                reproj_thresh_px=reproj_thresh_px,
                min_inliers=min_inliers,
                random_seed=17 + marker_id * 10 + corner_idx,
            )
        except Exception as exc:
            print(f"[WARN] triangulation failed marker={marker_id} corner={corner_idx}: {exc}")
            continue

        X = res["X"]
        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_errs = [errs[i] for i in res["inlier_idx"]]

        X_col.append(np.asarray(X, dtype=np.float64))
        Y_ref.append(np.asarray(targets[key], dtype=np.float64))

        rows.append({
            "marker_id": marker_id,
            "entity_id": marker_entity(marker_id),
            "corner_idx": corner_idx,
            "x_colmap": X[0],
            "y_colmap": X[1],
            "z_colmap": X[2],
            "target_x_ref14_m": targets[key][0],
            "target_y_ref14_m": targets[key][1],
            "target_z_ref14_m": targets[key][2],
            "obs_count": res["obs_count"],
            "inlier_count": res["inlier_count"],
            "mean_reproj_px": mean(inlier_errs) if inlier_errs else float("nan"),
            "median_reproj_px": median(inlier_errs) if inlier_errs else float("nan"),
            "max_reproj_px": max(inlier_errs) if inlier_errs else float("nan"),
        })

    if len(X_col) < 4:
        raise RuntimeError(f"Too few triangulated marker corners: {len(X_col)}")

    return np.asarray(X_col, dtype=np.float64), np.asarray(Y_ref, dtype=np.float64), rows


def robust_sim3_with_outlier_filter(X_col, Y_ref, rows, max_fit_error_cm):
    scale, R, t = umeyama_similarity(X_col, Y_ref)

    fit_errors_cm = []
    for x, y in zip(X_col, Y_ref):
        p = apply_sim3_point(x, scale, R, t)
        fit_errors_cm.append(100.0 * float(np.linalg.norm(p - y)))

    med = median(fit_errors_cm)
    mad = median([abs(e - med) for e in fit_errors_cm])
    thresh = med + 3.0 * max(mad, 1e-9)
    thresh = min(thresh, float(max_fit_error_cm))

    keep_idx = [i for i, e in enumerate(fit_errors_cm) if e <= thresh]

    if len(keep_idx) >= 4 and len(keep_idx) < len(X_col):
        X2 = X_col[keep_idx]
        Y2 = Y_ref[keep_idx]
        scale, R, t = umeyama_similarity(X2, Y2)
    else:
        keep_idx = list(range(len(X_col)))

    final_errors_cm = []
    out_rows = []
    keep_set = set(keep_idx)

    for i, row in enumerate(rows):
        p = apply_sim3_point(X_col[i], scale, R, t)
        err = 100.0 * float(np.linalg.norm(p - Y_ref[i]))
        final_errors_cm.append(err)

        rr = dict(row)
        rr.update({
            "used_for_sim3": "yes" if i in keep_set else "no",
            "registered_x_ref14_m": p[0],
            "registered_y_ref14_m": p[1],
            "registered_z_ref14_m": p[2],
            "corner_fit_error_cm": err,
        })
        out_rows.append(rr)

    return scale, R, t, out_rows, final_errors_cm, keep_idx


def apply_registration_to_colmap(images, points3d, scale, R, t):
    static_rows = []
    moving_rows = []

    for image_name, payload in sorted(images.items()):
        T_ref_cam = apply_sim3_pose(payload["T_col_cam"], scale, R, t)
        if image_name.startswith("static_"):
            static_rows.append(
                pose_row(
                    "static_camera",
                    source_id_from_image_name(image_name),
                    T_ref_cam,
                    "ap03_multi_aruco_sim3_registration",
                )
            )
        elif image_name.startswith("moving_"):
            moving_rows.append(
                pose_row(
                    "moving_frame",
                    source_id_from_image_name(image_name),
                    T_ref_cam,
                    "ap03_multi_aruco_sim3_registration",
                )
            )

    point_rows = []
    for p in points3d:
        p_ref = apply_sim3_point(
            np.array([p["x_colmap"], p["y_colmap"], p["z_colmap"]], dtype=np.float64),
            scale,
            R,
            t,
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
    gt = gt_static_camera_poses_ref_aruco(
        WORLD_SDF_MOVING_CAMERA,
        STATIC_CAMERAS,
        REF_MARKER_ENTITY,
    )
    est = {r["entity_id"]: r for r in static_rows}
    eval_rows = []

    for cam in STATIC_CAMERAS:
        if cam not in est:
            raise RuntimeError(f"Missing AP03 static camera estimate: {cam}")

        r = est[cam]
        T_est = make_T(
            rvec_to_R([float(r["rvec_x"]), float(r["rvec_y"]), float(r["rvec_z"])]),
            [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])],
        )
        T_gt = gt[cam]

        row = {
            "approach": "AP03_targetless_colmap_multi_aruco_scale",
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


def fmt(x, nd=3):
    return f"{float(x):.{nd}f}"


def md_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--final-root", type=Path, default=FINAL_ROOT)
    ap.add_argument("--min-area-px2", type=float, default=1000.0)
    ap.add_argument("--reproj-thresh-px", type=float, default=5.0)
    ap.add_argument("--ransac-iters", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=4)
    ap.add_argument("--max-fit-error-cm", type=float, default=20.0)
    ap.add_argument("--marker-ids", type=str, default="0-14")
    args = ap.parse_args()

    ensure_dir(args.out_root)
    ensure_dir(args.final_root)
    ensure_dir(AP3_CMP)
    ensure_dir(COMBINED)

    if args.marker_ids == "0-14":
        marker_ids = list(range(15))
    else:
        marker_ids = [int(x) for x in args.marker_ids.split(",") if x.strip()]

    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()

    print("=== AP03 Multi-ArUco Registration ===")
    print(f"best_model: {best_model}")
    print(f"registered_images: {len(images)}")
    print(f"markers requested: {marker_ids}")

    obs = detect_all_aruco_observations(images, args.min_area_px2, marker_ids)
    if len(obs) < 8:
        raise RuntimeError(f"Too few ArUco observations: {len(obs)}")

    obs_fields = ["image_name", "marker_id", "corner_idx", "u", "v", "area_px2"]
    write_csv(args.out_root / "ap03_all_aruco_corner_observations.csv", obs, obs_fields)

    unique_markers_detected = sorted({int(o["marker_id"]) for o in obs})
    unique_images = sorted({o["image_name"] for o in obs})
    print(f"aruco_corner_observations: {len(obs)}")
    print(f"unique_marker_ids_detected: {unique_markers_detected}")
    print(f"unique_anchor_images: {len(unique_images)}")

    targets = load_known_marker_corner_targets_ref14(marker_ids)

    X_col, Y_ref, tri_rows = triangulate_all_marker_corners(
        obs,
        images,
        cameras,
        targets,
        ransac_iters=args.ransac_iters,
        reproj_thresh_px=args.reproj_thresh_px,
        min_inliers=args.min_inliers,
    )

    scale, R, t, corner_rows, fit_errors_cm, keep_idx = robust_sim3_with_outlier_filter(
        X_col,
        Y_ref,
        tri_rows,
        max_fit_error_cm=args.max_fit_error_cm,
    )

    corner_fields = [
        "marker_id", "entity_id", "corner_idx",
        "x_colmap", "y_colmap", "z_colmap",
        "target_x_ref14_m", "target_y_ref14_m", "target_z_ref14_m",
        "obs_count", "inlier_count",
        "mean_reproj_px", "median_reproj_px", "max_reproj_px",
        "used_for_sim3",
        "registered_x_ref14_m", "registered_y_ref14_m", "registered_z_ref14_m",
        "corner_fit_error_cm",
    ]
    write_csv(
        args.out_root / "ap03_triangulated_multi_aruco_corners_colmap_and_registered.csv",
        corner_rows,
        corner_fields,
    )

    # Compatibility copy for old report tooling if needed.
    write_csv(
        args.out_root / "ap03_triangulated_ref14_corners_colmap_and_registered.csv",
        corner_rows,
        corner_fields,
    )

    sim3_meta = {
        "stage": "09_make_ap03_multi_aruco_final",
        "registration_mode": "multi_aruco_known_marker_layout",
        "best_colmap_model": best_model,
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "marker_length_m": MARKER_LENGTH_M,
        "min_area_px2": args.min_area_px2,
        "reproj_thresh_px": args.reproj_thresh_px,
        "ransac_iters": args.ransac_iters,
        "min_inliers": args.min_inliers,
        "max_fit_error_cm": args.max_fit_error_cm,
        "requested_marker_ids": marker_ids,
        "detected_marker_ids": unique_markers_detected,
        "corner_observation_count": len(obs),
        "unique_anchor_images": len(unique_images),
        "triangulated_corner_count": len(corner_rows),
        "used_corner_count": len(keep_idx),
        "used_marker_ids": sorted({int(corner_rows[i]["marker_id"]) for i in keep_idx}),
        "scale_colmap_to_metric": float(scale),
        "R_ref_col": np.asarray(R, dtype=float).tolist(),
        "t_ref_col": np.asarray(t, dtype=float).reshape(3).tolist(),
        "corner_fit_mean_error_cm": mean(fit_errors_cm),
        "corner_fit_median_error_cm": median(fit_errors_cm),
        "corner_fit_max_error_cm": max(fit_errors_cm),
        "corner_triangulation_mean_reproj_px": mean([r["mean_reproj_px"] for r in corner_rows]),
        "corner_triangulation_median_reproj_px": median([r["median_reproj_px"] for r in corner_rows]),
    }
    (args.out_root / "ap03_ref14_sim3_metadata.json").write_text(json.dumps(sim3_meta, indent=2) + "\n")
    (args.out_root / "ap03_multi_aruco_sim3_metadata.json").write_text(json.dumps(sim3_meta, indent=2) + "\n")

    # Compatibility metadata for older report naming.
    detect_meta = {
        "stage": "09_make_ap03_multi_aruco_final",
        "registration_mode": "multi_aruco_known_marker_layout",
        "best_colmap_model": best_model,
        "model_dir": str(model_dir),
        "registered_images": len(images),
        "registered_cameras": len(cameras),
        "num_sparse_points3d": len(points3d),
        "reference_marker_id": REF_MARKER_ID,
        "marker_length_m": MARKER_LENGTH_M,
        "min_area_px2": args.min_area_px2,
        "ref14_corner_observation_count": len([o for o in obs if int(o["marker_id"]) == REF_MARKER_ID]),
        "ref14_unique_anchor_images": len({o["image_name"] for o in obs if int(o["marker_id"]) == REF_MARKER_ID}),
        "all_aruco_corner_observation_count": len(obs),
        "all_aruco_unique_anchor_images": len(unique_images),
        "detected_marker_ids": unique_markers_detected,
    }
    (args.out_root / "06a_detection_metadata.json").write_text(json.dumps(detect_meta, indent=2) + "\n")

    static_rows, moving_rows, point_rows = apply_registration_to_colmap(
        images,
        points3d,
        scale,
        R,
        t,
    )

    write_csv(args.out_root / "ap03_static_camera_poses_ref_aruco.csv", static_rows, pose_fields())
    write_csv(args.out_root / "ap03_moving_frame_poses_ref_aruco.csv", moving_rows, pose_fields())
    write_csv(
        args.out_root / "ap03_sparse_points3d_ref_aruco.csv",
        point_rows,
        ["point3d_id", "x_ref_aruco_m", "y_ref_aruco_m", "z_ref_aruco_m", "r", "g", "b", "colmap_reprojection_error"],
    )

    eval_rows = eval_static_cameras_vs_gt(static_rows)
    write_csv(args.out_root / "ap03_static_cameras_ref_aruco_vs_gt.csv", eval_rows, eval_fields())
    write_csv(AP3_CMP / "ap03_static_cameras_ref_aruco_vs_gt_repo_like.csv", eval_rows, eval_fields())

    cam_t = [float(r["translation_error_cm"]) for r in eval_rows]
    cam_r = [float(r["rotation_error_deg"]) for r in eval_rows]

    final_csv_rows = []
    for r in eval_rows:
        final_csv_rows.append({
            "approach": "AP03",
            "method": "Targetless COLMAP + Multi-ArUco Sim3",
            "evaluation": "Ref14-frame static camera evaluation",
            "entity_type": r["entity_type"],
            "entity_id": r["entity_id"],
            "translation_error_cm": r["translation_error_cm"],
            "rotation_error_deg": r["rotation_error_deg"],
            "delta_x_cm": r["delta_x_cm"],
            "delta_y_cm": r["delta_y_cm"],
            "delta_z_cm": r["delta_z_cm"],
            "est_ref14_x_m": r["est_ref_aruco_x_m"],
            "est_ref14_y_m": r["est_ref_aruco_y_m"],
            "est_ref14_z_m": r["est_ref_aruco_z_m"],
            "gt_ref14_x_m": r["gt_ref_aruco_x_m"],
            "gt_ref14_y_m": r["gt_ref_aruco_y_m"],
            "gt_ref14_z_m": r["gt_ref_aruco_z_m"],
        })

    final_fields = list(final_csv_rows[0].keys())
    write_csv(args.final_root / "AP03_FINAL_RESULT.csv", final_csv_rows, final_fields)
    write_csv(args.final_root / "AP03_FINAL_MULTI_ARUCO_RESULT.csv", final_csv_rows, final_fields)
    write_csv(args.final_root / "AP03_FINAL_REF14_RELATIVE_EVALUATION.csv", final_csv_rows, final_fields)

    cam_table_rows = []
    for r in final_csv_rows:
        cam_table_rows.append([
            r["entity_id"],
            fmt(r["translation_error_cm"]),
            fmt(r["rotation_error_deg"]),
            fmt(r["delta_x_cm"]),
            fmt(r["delta_y_cm"]),
            fmt(r["delta_z_cm"]),
            f"({fmt(r['est_ref14_x_m'])}, {fmt(r['est_ref14_y_m'])}, {fmt(r['est_ref14_z_m'])})",
            f"({fmt(r['gt_ref14_x_m'])}, {fmt(r['gt_ref14_y_m'])}, {fmt(r['gt_ref14_z_m'])})",
        ])

    marker_summary = {}
    for r in corner_rows:
        mid = int(r["marker_id"])
        marker_summary.setdefault(mid, {"corners": 0, "used": 0, "fit": [], "repr": []})
        marker_summary[mid]["corners"] += 1
        if r["used_for_sim3"] == "yes":
            marker_summary[mid]["used"] += 1
        marker_summary[mid]["fit"].append(float(r["corner_fit_error_cm"]))
        marker_summary[mid]["repr"].append(float(r["mean_reproj_px"]))

    marker_table_rows = []
    for mid in sorted(marker_summary.keys()):
        s = marker_summary[mid]
        marker_table_rows.append([
            marker_entity(mid),
            s["corners"],
            s["used"],
            fmt(mean(s["fit"])),
            fmt(mean(s["repr"])),
        ])

    report = f"""AP03 FINAL RESULT — Targetless COLMAP + Multi-ArUco Scale Registration
=====================================================================

Method:
COLMAP reconstructs static and moving cameras from natural image features.
The COLMAP reconstruction has arbitrary scale and coordinate frame.
Metric scale and Ref14-frame registration are now estimated using all visible ArUco markers with a known marker layout.

Important:
- COLMAP itself remains targetless.
- ArUco markers are used only after COLMAP for Sim(3) metric registration.
- This replaces the previous single-Ref14 registration with multi-ArUco registration.
- Known marker layout is used as the metric calibration reference.
- GT camera poses are used only for final error evaluation.

COLMAP input:
- model: {best_model}
- registered images: {len(images)}
- registered static cameras: {len(static_rows)} / 4
- registered moving frames: {len(moving_rows)}
- sparse 3D points: {len(points3d)}

Multi-ArUco registration:
- requested marker ids: {marker_ids}
- detected marker ids: {unique_markers_detected}
- corner observations: {len(obs)}
- unique anchor images: {len(unique_images)}
- triangulated marker corners: {len(corner_rows)}
- used marker corners after filtering: {len(keep_idx)}
- used marker ids: {sim3_meta["used_marker_ids"]}
- estimated COLMAP-to-meter scale: {float(scale):.9f}
- corner fit mean error: {mean(fit_errors_cm):.6f} cm
- corner fit median error: {median(fit_errors_cm):.6f} cm
- corner fit max error: {max(fit_errors_cm):.6f} cm
- mean corner reprojection error: {mean([r["mean_reproj_px"] for r in corner_rows]):.6f} px
- median corner reprojection error: {median([r["median_reproj_px"] for r in corner_rows]):.6f} px

Marker contribution:
{md_table(["marker", "triangulated_corners", "used_corners", "mean_fit_cm", "mean_repr_px"], marker_table_rows)}

Static camera results relative to Ref14:
- count: {len(eval_rows)}
- mean translation error: {mean(cam_t):.6f} cm
- median translation error: {median(cam_t):.6f} cm
- mean rotation error: {mean(cam_r):.6f} deg
- median rotation error: {median(cam_r):.6f} deg

{md_table(["camera", "t_err_cm", "r_err_deg", "dX_cm", "dY_cm", "dZ_cm", "estimated xyz [m]", "GT xyz [m]"], cam_table_rows)}
"""

    (args.final_root / "AP03_FINAL_RESULT.txt").write_text(report)
    (args.final_root / "AP03_FINAL_MULTI_ARUCO_RESULT.txt").write_text(report)
    (args.final_root / "AP03_FINAL_REF14_RELATIVE_EVALUATION.txt").write_text(report)
    (args.out_root / "AP03_FINAL_MULTI_ARUCO_SCALE_REGISTRATION_REPORT.txt").write_text(report)
    (args.out_root / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)
    (AP3_CMP / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)

    summary = [{
        "approach": "AP03_targetless_colmap_multi_aruco_scale",
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "final_variant": "colmap_then_multi_aruco_sim3",
        "registered_images": len(images),
        "registered_static_cameras": len(static_rows),
        "registered_moving_frames": len(moving_rows),
        "num_sparse_points3d": len(points3d),
        "corner_observation_count": len(obs),
        "unique_anchor_images": len(unique_images),
        "detected_marker_ids": ";".join(str(x) for x in unique_markers_detected),
        "used_marker_ids": ";".join(str(x) for x in sim3_meta["used_marker_ids"]),
        "estimated_colmap_to_metric_scale": scale,
        "corner_fit_mean_error_cm": mean(fit_errors_cm),
        "corner_fit_median_error_cm": median(fit_errors_cm),
        "corner_triangulation_mean_reproj_px": mean([r["mean_reproj_px"] for r in corner_rows]),
        "corner_triangulation_median_reproj_px": median([r["median_reproj_px"] for r in corner_rows]),
        "camera_mean_translation_error_cm": mean(cam_t),
        "camera_median_translation_error_cm": median(cam_t),
        "camera_mean_rotation_error_deg": mean(cam_r),
        "camera_median_rotation_error_deg": median(cam_r),
    }]
    write_csv(COMBINED / "ap03_repo_like_final_summary.csv", summary, list(summary[0].keys()))

    print(report)
    print()
    print("[OK] wrote:", args.final_root / "AP03_FINAL_RESULT.txt")
    print("[OK] wrote:", args.final_root / "AP03_FINAL_RESULT.csv")


if __name__ == "__main__":
    main()
