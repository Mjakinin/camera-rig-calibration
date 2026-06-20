#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ap02_common import (
    AP02_ROOT,
    DEFAULT_REF_MARKER_ID,
    ensure_dir,
    read_csv,
    write_csv,
    rvec_to_R,
    R_to_rvec,
    make_T,
    invT,
    pose_row,
    pose_fields,
)


OBS_CSV = AP02_ROOT / "02_aruco_observations" / "ap02_all_aruco_observations.csv"
INIT_ROOT = AP02_ROOT / "05_graph_initialization"


def is_success(row):
    return str(row.get("pnp_success", "")).strip().lower() in ["true", "1", "yes"]


def safe_float(row, key, default=float("nan")):
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def marker_object_points(marker_length_m):
    s = marker_length_m / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def T_from_pose_row(row):
    rvec = np.array([
        safe_float(row, "rvec_x"),
        safe_float(row, "rvec_y"),
        safe_float(row, "rvec_z"),
    ], dtype=np.float64)
    t = np.array([
        safe_float(row, "x_m"),
        safe_float(row, "y_m"),
        safe_float(row, "z_m"),
    ], dtype=np.float64)

    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(t)):
        raise RuntimeError(f"Invalid pose row: {row}")

    return make_T(rvec_to_R(rvec), t)


def load_initial_poses(mode):
    mode_dir = INIT_ROOT / mode

    marker_rows = read_csv(mode_dir / "initial_marker_poses_ref_marker.csv")
    static_rows = read_csv(mode_dir / "initial_static_camera_poses_ref_marker.csv")
    moving_rows_path = mode_dir / "initial_moving_frame_poses_ref_marker.csv"
    moving_rows = read_csv(moving_rows_path) if moving_rows_path.exists() else []

    marker_poses = {}
    observer_poses = {}

    for r in marker_rows:
        marker_poses[int(r["entity_id"])] = T_from_pose_row(r)

    for r in static_rows:
        observer_poses[r["entity_id"]] = T_from_pose_row(r)

    for r in moving_rows:
        observer_poses[r["entity_id"]] = T_from_pose_row(r)

    return marker_poses, observer_poses


def filter_observations(rows, mode, marker_poses, observer_poses):
    filtered = []
    for r in rows:
        if not is_success(r):
            continue

        if mode == "static_only" and r.get("observer_type") != "static":
            continue

        try:
            marker_id = int(float(r["marker_id"]))
        except Exception:
            continue

        observer_id = r["observer_id"]

        if marker_id not in marker_poses:
            continue
        if observer_id not in observer_poses:
            continue

        filtered.append(r)

    return filtered


def pack_params(marker_poses, observer_poses, ref_marker_id):
    names = []
    x = []

    for observer_id in sorted(observer_poses):
        T = observer_poses[observer_id]
        rvec = R_to_rvec(T[:3, :3])
        t = T[:3, 3]
        names.append(("observer", observer_id))
        x.extend([rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]])

    for marker_id in sorted(marker_poses):
        if marker_id == ref_marker_id:
            continue
        T = marker_poses[marker_id]
        rvec = R_to_rvec(T[:3, :3])
        t = T[:3, 3]
        names.append(("marker", marker_id))
        x.extend([rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]])

    return np.array(x, dtype=np.float64), names


def unpack_params(x, names, ref_marker_id):
    marker_poses = {
        ref_marker_id: np.eye(4, dtype=np.float64)
    }
    observer_poses = {}

    i = 0
    for kind, entity_id in names:
        rvec = x[i:i+3]
        t = x[i+3:i+6]
        i += 6
        T = make_T(rvec_to_R(rvec), t)

        if kind == "observer":
            observer_poses[entity_id] = T
        elif kind == "marker":
            marker_poses[int(entity_id)] = T

    return marker_poses, observer_poses


def project_point(K, P_cam):
    x, y, z = P_cam[:3]
    if z <= 1e-9:
        return None

    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return np.array([u, v], dtype=np.float64)


def observation_residuals(row, marker_poses, observer_poses):
    marker_id = int(float(row["marker_id"]))
    observer_id = row["observer_id"]

    T_ref_marker = marker_poses[marker_id]
    T_ref_observer = observer_poses[observer_id]
    T_observer_ref = invT(T_ref_observer)

    fx = safe_float(row, "fx")
    fy = safe_float(row, "fy")
    cx = safe_float(row, "cx")
    cy = safe_float(row, "cy")
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    marker_length_m = safe_float(row, "marker_length_m", 0.170)
    obj_pts = marker_object_points(marker_length_m)

    residuals = []

    for idx, P_marker in enumerate(obj_pts):
        observed = np.array([
            safe_float(row, f"corner{idx}_u"),
            safe_float(row, f"corner{idx}_v"),
        ], dtype=np.float64)

        if not np.all(np.isfinite(observed)):
            continue

        P_marker_h = np.array([P_marker[0], P_marker[1], P_marker[2], 1.0], dtype=np.float64)
        P_ref = T_ref_marker @ P_marker_h
        P_obs = T_observer_ref @ P_ref

        projected = project_point(K, P_obs)
        if projected is None:
            # Heavy penalty for points behind camera, but finite.
            residuals.extend([1000.0, 1000.0])
        else:
            residuals.extend((projected - observed).tolist())

    return residuals


def make_residual_function(observations, names, ref_marker_id):
    def fun(x):
        marker_poses, observer_poses = unpack_params(x, names, ref_marker_id)
        residuals = []
        for row in observations:
            residuals.extend(observation_residuals(row, marker_poses, observer_poses))
        return np.array(residuals, dtype=np.float64)

    return fun


def reprojection_error_rows(observations, marker_poses, observer_poses):
    rows = []
    all_abs = []

    for row in observations:
        residuals = observation_residuals(row, marker_poses, observer_poses)
        residuals = np.array(residuals, dtype=np.float64).reshape(-1, 2)

        if residuals.size == 0:
            continue

        errs = np.linalg.norm(residuals, axis=1)
        all_abs.extend(errs.tolist())

        rows.append({
            "observer_type": row.get("observer_type", ""),
            "observer_id": row.get("observer_id", ""),
            "frame_id": row.get("frame_id", ""),
            "marker_id": row.get("marker_id", ""),
            "num_corners": len(errs),
            "mean_reproj_px": float(np.mean(errs)),
            "median_reproj_px": float(np.median(errs)),
            "max_reproj_px": float(np.max(errs)),
        })

    return rows, np.array(all_abs, dtype=np.float64)


def run_ba(mode, ref_marker_id, max_nfev, moving_stride=1, max_moving_frames=0):
    out = ensure_dir(AP02_ROOT / "07_graph_ba" / mode)

    marker_init, observer_init = load_initial_poses(mode)
    all_rows = read_csv(OBS_CSV)
    observations = filter_observations(all_rows, mode, marker_init, observer_init)

    if mode == "with_moving":
        # Keep all static observations, but subsample moving frames.
        static_obs = [r for r in observations if r.get("observer_type") == "static"]
        moving_obs = [r for r in observations if r.get("observer_type") == "moving"]

        moving_frame_ids = sorted({r["observer_id"] for r in moving_obs})
        if moving_stride > 1:
            moving_frame_ids = moving_frame_ids[::moving_stride]
        if max_moving_frames and max_moving_frames > 0:
            moving_frame_ids = moving_frame_ids[:max_moving_frames]

        keep_moving = set(moving_frame_ids)
        observations = static_obs + [r for r in moving_obs if r["observer_id"] in keep_moving]

        used_observers = {r["observer_id"] for r in observations}
        observer_init = {k: v for k, v in observer_init.items() if k in used_observers}

        used_markers = {int(float(r["marker_id"])) for r in observations}
        marker_init = {k: v for k, v in marker_init.items() if k in used_markers or k == ref_marker_id}

        print(f"[INFO] with_moving subsampling: stride={moving_stride}, max_moving_frames={max_moving_frames}")
        print(f"[INFO] kept moving frames={len(keep_moving)}")
        print(f"[INFO] kept observations={len(observations)}")

    if len(observations) == 0:
        raise RuntimeError(f"No valid observations for mode {mode}")

    x0, names = pack_params(marker_init, observer_init, ref_marker_id)
    residual0_fun = make_residual_function(observations, names, ref_marker_id)
    r0 = residual0_fun(x0)
    r0_norm = np.linalg.norm(r0.reshape(-1, 2), axis=1)

    print(f"[INFO] mode={mode}")
    print(f"[INFO] variables poses={len(names)}, params={len(x0)}")
    print(f"[INFO] observations={len(observations)}, residual scalars={len(r0)}")
    print(f"[INFO] initial median reproj px={np.median(r0_norm):.3f}, mean={np.mean(r0_norm):.3f}")

    result = least_squares(
        residual0_fun,
        x0,
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=max_nfev,
        verbose=1,
    )

    marker_opt, observer_opt = unpack_params(result.x, names, ref_marker_id)
    err_rows, err_values = reprojection_error_rows(observations, marker_opt, observer_opt)

    static_pose_rows = []
    moving_pose_rows = []
    for observer_id, T in sorted(observer_opt.items()):
        if observer_id.startswith("cam_edge_"):
            static_pose_rows.append(pose_row("static_camera", observer_id, T, source=f"ba_{mode}"))
        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(pose_row("moving_frame", observer_id, T, source=f"ba_{mode}"))

    marker_pose_rows = []
    for marker_id, T in sorted(marker_opt.items()):
        marker_pose_rows.append(pose_row("marker", str(marker_id), T, source=f"ba_{mode}"))

    write_csv(out / "optimized_static_camera_poses_ref_marker.csv", static_pose_rows, pose_fields())
    write_csv(out / "optimized_moving_frame_poses_ref_marker.csv", moving_pose_rows, pose_fields())
    write_csv(out / "optimized_marker_poses_ref_marker.csv", marker_pose_rows, pose_fields())

    err_fields = [
        "observer_type",
        "observer_id",
        "frame_id",
        "marker_id",
        "num_corners",
        "mean_reproj_px",
        "median_reproj_px",
        "max_reproj_px",
    ]
    write_csv(out / "reprojection_errors_by_observation.csv", err_rows, err_fields)

    summary = [
        "AP02 Reference-marker Graph Bundle Adjustment",
        "============================================",
        "",
        f"Mode: {mode}",
        f"Reference marker id: {ref_marker_id}",
        "",
        f"Optimized variable poses: {len(names)}",
        f"Parameter count: {len(result.x)}",
        f"Marker observations used: {len(observations)}",
        f"Scalar residual count: {len(result.fun)}",
        "",
        "Initial reprojection error [px]:",
        f"- mean: {float(np.mean(r0_norm)):.6f}",
        f"- median: {float(np.median(r0_norm)):.6f}",
        f"- max: {float(np.max(r0_norm)):.6f}",
        "",
        "Final reprojection error [px]:",
        f"- mean: {float(np.mean(err_values)):.6f}",
        f"- median: {float(np.median(err_values)):.6f}",
        f"- max: {float(np.max(err_values)):.6f}",
        "",
        "Optimizer:",
        f"- success: {result.success}",
        f"- status: {result.status}",
        f"- message: {result.message}",
        f"- nfev: {result.nfev}",
        f"- cost: {float(result.cost):.6f}",
        "",
        "Outputs:",
        "- optimized_static_camera_poses_ref_marker.csv",
        "- optimized_moving_frame_poses_ref_marker.csv",
        "- optimized_marker_poses_ref_marker.csv",
        "- reprojection_errors_by_observation.csv",
        "",
    ]

    (out / "ba_summary.txt").write_text("\n".join(summary) + "\n")

    print("[OK] wrote", out)
    print("[OK] final median reproj px:", float(np.median(err_values)))
    print("[OK] final mean reproj px:", float(np.mean(err_values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static_only", "with_moving"], required=True)
    ap.add_argument("--ref-marker-id", type=int, default=DEFAULT_REF_MARKER_ID)
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--moving-stride", type=int, default=1,
                    help="Use only every Nth moving frame for with_moving BA.")
    ap.add_argument("--max-moving-frames", type=int, default=0,
                    help="Maximum number of moving frames to use. 0 means no limit.")
    args = ap.parse_args()

    run_ba(args.mode, args.ref_marker_id, args.max_nfev, args.moving_stride, args.max_moving_frames)


if __name__ == "__main__":
    main()
