#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


# Gazebo camera link -> optical frame convention:
# link:    x forward, y left, z up
# optical: x right,   y down, z forward
R_OPT_LINK = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)


def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_R_world_link(roll, pitch, yaw):
    # SDF/Gazebo fixed-axis convention for this project.
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def optical_T_world_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)


def qvec_to_R(qvec):
    qw, qx, qy, qz = qvec
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def read_colmap_images_txt(path):
    poses = {}
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {path}")

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            int(parts[0])
            qvec = np.array([float(v) for v in parts[1:5]], dtype=float)
            tvec = np.array([float(v) for v in parts[5:8]], dtype=float)
            name = parts[9]
        except Exception:
            continue

        R_cw = qvec_to_R(qvec)
        T_cw = make_T(R_cw, tvec)
        C_w = -R_cw.T @ tvec

        poses[name] = {
            "R_cw_colmap": R_cw,
            "t_cw_colmap": tvec,
            "T_cw_colmap": T_cw,
            "C_colmap": C_w,
        }

    return poses


def read_route_gt(path):
    route = {}
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"route_gt.csv not found: {path}")

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["image_name"]
            route[name] = {
                "pos": np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float),
                "rpy": np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])], dtype=float),
                "tag": row.get("tag", ""),
            }

    return route


def estimate_sim3_umeyama(src, dst):
    """
    Estimate dst ~= scale * R * src + t
    src: Nx3 COLMAP camera centers
    dst: Nx3 GT camera centers
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least 3 paired 3D points for Sim(3) alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_src = np.mean(np.sum(X * X, axis=1))
    scale = np.sum(D * np.diag(S)) / var_src
    t = mu_dst - scale * R @ mu_src

    return scale, R, t


def align_colmap_pose_to_gt(T_cw_colmap, C_colmap, scale, R_sim, t_sim):
    """
    COLMAP world -> GT world:
      X_gt = scale * R_sim * X_colmap + t_sim

    Returns aligned world-to-camera optical pose:
      X_camera = R_cw_gt * X_gt + t_cw_gt
    """
    R_cw_col = T_cw_colmap[:3, :3]
    R_wc_col = R_cw_col.T

    C_gt = scale * R_sim @ C_colmap + t_sim
    R_wc_gt = R_sim @ R_wc_col
    R_cw_gt = R_wc_gt.T
    t_cw_gt = -R_cw_gt @ C_gt

    return make_T(R_cw_gt, t_cw_gt)


def parse_vec_from_row(row, kind):
    cols = list(row.keys())

    explicit = [
        (f"{kind}_x", f"{kind}_y", f"{kind}_z"),
        (f"{kind}_x_m", f"{kind}_y_m", f"{kind}_z_m"),
        (f"{kind}_0", f"{kind}_1", f"{kind}_2"),
        (f"{kind}0", f"{kind}1", f"{kind}2"),
    ]

    for names in explicit:
        if all(n in row and row[n] not in ("", None) for n in names):
            return np.array([float(row[n]) for n in names], dtype=float)

    combined_candidates = [
        kind,
        f"{kind}_xyz",
        f"{kind}_xyz_m",
        f"{kind}_xyz_rad",
    ]

    for name in combined_candidates:
        if name in row and row[name]:
            raw = row[name].replace("[", "").replace("]", "").replace(";", ",")
            vals = [v.strip() for v in raw.split(",") if v.strip()]
            if len(vals) == 3:
                return np.array([float(v) for v in vals], dtype=float)

    lowered = {c.lower(): c for c in cols}

    def find_axis(axis):
        candidates = []
        for c in cols:
            lc = c.lower()
            if kind not in lc:
                continue
            if (
                lc.endswith("_" + axis)
                or lc.endswith(axis)
                or f"_{axis}_" in lc
                or f"{axis}_" in lc
            ):
                candidates.append(c)
        return candidates[0] if candidates else None

    names = [find_axis("x"), find_axis("y"), find_axis("z")]
    if all(names) and all(row[n] not in ("", None) for n in names):
        return np.array([float(row[n]) for n in names], dtype=float)

    raise KeyError(
        f"Could not parse {kind} vector from CSV row. Available columns: {cols}"
    )


def read_observation_pose(csv_path, camera_name):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Observation CSV not found: {csv_path}")

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        cam = row.get("camera", row.get("camera_name", ""))
        status = row.get("status", "")
        if cam == camera_name and status == "pose_valid":
            rvec = parse_vec_from_row(row, "rvec")
            tvec = parse_vec_from_row(row, "tvec")
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            return make_T(R, tvec)

    available = [(r.get("camera", r.get("camera_name", "")), r.get("status", "")) for r in rows]
    raise RuntimeError(
        f"No pose_valid row for camera '{camera_name}' in {csv_path}. "
        f"Available rows: {available}"
    )


def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def save_matrix(path, T):
    np.savetxt(path, T, delimiter=",", fmt="%.10f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--front_obs", required=True)
    ap.add_argument("--rear_obs", required=True)
    ap.add_argument("--front_frame", required=True)
    ap.add_argument("--rear_frame", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_txt = dataset_dir / "sparse_txt" / "images.txt"
    route_gt_csv = dataset_dir / "route_gt.csv"

    colmap = read_colmap_images_txt(images_txt)
    route = read_route_gt(route_gt_csv)

    common = sorted(set(colmap.keys()) & set(route.keys()))
    if len(common) < 3:
        raise RuntimeError("Not enough common COLMAP/GT frames for Sim(3) alignment.")

    src = np.array([colmap[name]["C_colmap"] for name in common], dtype=float)
    dst = np.array([route[name]["pos"] for name in common], dtype=float)

    scale, R_sim, t_sim = estimate_sim3_umeyama(src, dst)

    if args.front_frame not in colmap:
        raise RuntimeError(f"Front frame not registered in COLMAP: {args.front_frame}")
    if args.rear_frame not in colmap:
        raise RuntimeError(f"Rear frame not registered in COLMAP: {args.rear_frame}")

    T_mF_world = align_colmap_pose_to_gt(
        colmap[args.front_frame]["T_cw_colmap"],
        colmap[args.front_frame]["C_colmap"],
        scale,
        R_sim,
        t_sim,
    )

    T_mR_world = align_colmap_pose_to_gt(
        colmap[args.rear_frame]["T_cw_colmap"],
        colmap[args.rear_frame]["C_colmap"],
        scale,
        R_sim,
        t_sim,
    )

    # Relative moving-camera transform:
    # maps points from rear moving-camera optical frame into front moving-camera optical frame.
    T_mF_mR = T_mF_world @ inv_T(T_mR_world)

    # OpenCV solvePnP convention:
    # X_camera = R_camera_board * X_board + t_camera_board
    # Therefore these are T_camera_board.
    T_front_boardF = read_observation_pose(args.front_obs, "front_static_camera")
    T_movingF_boardF = read_observation_pose(args.front_obs, "moving_calib_camera")

    T_rear_boardR = read_observation_pose(args.rear_obs, "rear_static_camera")
    T_movingR_boardR = read_observation_pose(args.rear_obs, "moving_calib_camera")

    # Front anchor:
    # X_front = T_front_boardF * inv(T_movingF_boardF) * X_movingF
    T_front_mF = T_front_boardF @ inv_T(T_movingF_boardF)

    # Rear anchor:
    # X_movingR = T_movingR_boardR * inv(T_rear_boardR) * X_rear
    T_mR_rear = T_movingR_boardR @ inv_T(T_rear_boardR)

    # Full relay:
    # X_front = T_front_mF * T_mF_mR * T_mR_rear * X_rear
    T_front_rear_est = T_front_mF @ T_mF_mR @ T_mR_rear

    # GT static camera poses from the current SDF setup.
    T_front_world_gt = optical_T_world_from_gazebo_pose(
        -3.90, 0.0, 2.85,
        0.0, 0.69813170, 0.0
    )
    T_rear_world_gt = optical_T_world_from_gazebo_pose(
        5.70, 0.0, 2.85,
        0.0, 0.69813170, math.pi
    )
    T_front_rear_gt = T_front_world_gt @ inv_T(T_rear_world_gt)

    t_est = T_front_rear_est[:3, 3]
    t_gt = T_front_rear_gt[:3, 3]

    baseline_est = float(np.linalg.norm(t_est))
    baseline_gt = float(np.linalg.norm(t_gt))
    baseline_error_cm = (baseline_est - baseline_gt) * 100.0
    translation_error_cm = float(np.linalg.norm(t_est - t_gt) * 100.0)
    rot_error = rotation_error_deg(T_front_rear_est[:3, :3], T_front_rear_gt[:3, :3])

    save_matrix(output_dir / "T_front_rear_est.csv", T_front_rear_est)
    save_matrix(output_dir / "T_front_rear_gt.csv", T_front_rear_gt)
    save_matrix(output_dir / "T_mF_mR_colmap_aligned.csv", T_mF_mR)
    save_matrix(output_dir / "T_front_mF.csv", T_front_mF)
    save_matrix(output_dir / "T_mR_rear.csv", T_mR_rear)

    np.savetxt(output_dir / "sim3_colmap_to_gt_scale_R_t.csv",
               np.vstack([
                   np.array([[scale, 0.0, 0.0, 0.0]]),
                   np.hstack([R_sim, t_sim.reshape(3, 1)])
               ]),
               delimiter=",",
               fmt="%.10f")

    summary = f"""COLMAP + ArUco relay transform summary
=====================================

Dataset:
  {dataset_dir}

Front observation:
  {args.front_obs}
Front frame:
  {args.front_frame}
Front frame tag:
  {route.get(args.front_frame, {}).get("tag", "")}

Rear observation:
  {args.rear_obs}
Rear frame:
  {args.rear_frame}
Rear frame tag:
  {route.get(args.rear_frame, {}).get("tag", "")}

Important note:
  This first prototype uses Sim(3) alignment from COLMAP camera centers to Gazebo GT route_gt.csv
  to recover metric scale and GT-world alignment.
  This is evaluation-assisted and valid for the simulation proof-of-concept.
  A fully real pipeline must estimate scale from target constraints, multi-frame board observations,
  fixed known board layout, another metric sensor, or another metric prior.

Sim(3) COLMAP -> GT:
  scale: {scale:.10f}
  t:     {t_sim[0]:.10f}, {t_sim[1]:.10f}, {t_sim[2]:.10f}

Estimated T_front_rear:
{T_front_rear_est}

GT T_front_rear:
{T_front_rear_gt}

Metrics:
  baseline_est_m:       {baseline_est:.6f}
  baseline_gt_m:        {baseline_gt:.6f}
  baseline_error_cm:    {baseline_error_cm:.2f}
  translation_error_cm: {translation_error_cm:.2f}
  rotation_error_deg:   {rot_error:.2f}

Output files:
  T_front_rear_est.csv
  T_front_rear_gt.csv
  T_mF_mR_colmap_aligned.csv
  T_front_mF.csv
  T_mR_rear.csv
  sim3_colmap_to_gt_scale_R_t.csv
"""

    (output_dir / "summary.txt").write_text(summary)

    print(summary)


if __name__ == "__main__":
    main()
