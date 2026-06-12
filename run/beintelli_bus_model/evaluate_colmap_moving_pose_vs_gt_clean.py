#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np


def qvec_to_R(qvec):
    qw, qx, qy, qz = qvec
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def read_colmap_images(images_txt):
    poses = {}
    lines = Path(images_txt).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            try:
                qvec = np.array([float(v) for v in parts[1:5]], dtype=float)
                tvec = np.array([float(v) for v in parts[5:8]], dtype=float)
                name = parts[9]
            except Exception:
                i += 1
                continue

            R_cw = qvec_to_R(qvec)
            C_w = -R_cw.T @ tvec
            poses[name] = C_w
            i += 2
        else:
            i += 1
    return poses


def read_route(path):
    route = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            route[row["image_name"]] = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ], dtype=float)
    return route


def best_fit_align_colmap_to_gt(colmap_xyz, gt_xyz):
    """
    Evaluation-only alignment.

    COLMAP reconstructs the trajectory in an arbitrary coordinate system.
    For evaluation, we find the best scale, rotation and translation that put
    the COLMAP trajectory into the simulation coordinate system.
    """
    src = np.asarray(colmap_xyz, dtype=float)
    dst = np.asarray(gt_xyz, dtype=float)

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

    pred = scale * (R @ src.T).T + t
    return pred, scale, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    seq = Path(args.sequence_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    colmap = read_colmap_images(seq / "sparse_txt" / "images.txt")
    gt = read_route(seq / "route_commanded.csv")

    common = sorted(set(colmap.keys()) & set(gt.keys()))
    if len(common) < 3:
        raise RuntimeError(f"Need at least 3 common frames, got {len(common)}")

    colmap_xyz = np.array([colmap[name] for name in common], dtype=float)
    gt_xyz = np.array([gt[name] for name in common], dtype=float)

    aligned_xyz, scale, R, t = best_fit_align_colmap_to_gt(colmap_xyz, gt_xyz)

    err = aligned_xyz - gt_xyz
    err_norm = np.linalg.norm(err, axis=1)

    rows = []
    for name, p, g, e, en in zip(common, aligned_xyz, gt_xyz, err, err_norm):
        rows.append({
            "image_name": name,
            "colmap_eval_aligned_x": f"{p[0]:.8f}",
            "colmap_eval_aligned_y": f"{p[1]:.8f}",
            "colmap_eval_aligned_z": f"{p[2]:.8f}",
            "gt_x": f"{g[0]:.8f}",
            "gt_y": f"{g[1]:.8f}",
            "gt_z": f"{g[2]:.8f}",
            "err_x_m": f"{e[0]:.8f}",
            "err_y_m": f"{e[1]:.8f}",
            "err_z_m": f"{e[2]:.8f}",
            "err_norm_m": f"{en:.8f}",
        })

    with (out / "colmap_moving_pose_vs_gt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rmse = float(np.sqrt(np.mean(err_norm ** 2)))
    mean = float(np.mean(err_norm))
    median = float(np.median(err_norm))
    maxerr = float(np.max(err_norm))

    best_idx = int(np.argmin(err_norm))
    worst_idx = int(np.argmax(err_norm))

    summary = f"""COLMAP MOVING-CAMERA POSE VS SIMULATION GT
==========================================

Question:
How well does COLMAP reconstruct the moving_calib_camera trajectory?

Important:
This is evaluation only.
The final no-GT calibration pipeline does not use the simulation trajectory.

Why evaluation alignment is needed:
COLMAP reconstructs the camera path in its own coordinate system.
That coordinate system does not directly match the Gazebo/world coordinate system.
Therefore, for evaluation only, the COLMAP path is best-fit aligned to the simulation path.
Then the remaining position error is measured.

Registered/common frames:
  {len(common)}

Position error after evaluation alignment:
  RMSE:   {rmse * 100.0:.2f} cm
  mean:   {mean * 100.0:.2f} cm
  median: {median * 100.0:.2f} cm
  max:    {maxerr * 100.0:.2f} cm

Best frame:
  {common[best_idx]}  error={err_norm[best_idx] * 100.0:.2f} cm

Worst frame:
  {common[worst_idx]}  error={err_norm[worst_idx] * 100.0:.2f} cm

Takeaway:
COLMAP reconstructs the moving-camera trajectory with roughly {mean * 100.0:.1f} cm mean position error and {rmse * 100.0:.1f} cm RMSE after evaluation alignment.
"""
    (out / "summary_colmap_moving_pose_vs_gt.txt").write_text(summary)

    readme = """# 03 COLMAP Moving Pose vs Simulation GT

This folder evaluates how well COLMAP reconstructs the moving_calib_camera trajectory.

Important:
- This is evaluation only.
- The final no-GT front-rear chain does not use the simulation trajectory.
- COLMAP has its own arbitrary coordinate system.
- Therefore, for evaluation only, the COLMAP path is aligned to the simulation path.
- After that, the remaining position error is measured.

Files:
- summary_colmap_moving_pose_vs_gt.txt
- colmap_moving_pose_vs_gt.csv

This step is used to understand the quality of the moving-camera trajectory before composing the full front-rear camera chain.
"""
    (out / "README.md").write_text(readme)

    print(summary)


if __name__ == "__main__":
    main()
