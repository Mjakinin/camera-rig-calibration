#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np

R_OPT_LINK = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)

def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def rpy_to_R_world_link(roll, pitch, yaw):
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

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
                int(parts[0])
                qvec = np.array([float(v) for v in parts[1:5]], dtype=float)
                tvec = np.array([float(v) for v in parts[5:8]], dtype=float)
                name = parts[9]
            except Exception:
                i += 1
                continue
            R_cw = qvec_to_R(qvec)
            C_w = -R_cw.T @ tvec
            poses[name] = {"C_colmap": C_w, "R_cw": R_cw}
            i += 2
        else:
            i += 1
    return poses

def read_route_commanded(path):
    route = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            name = row["image_name"]
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
            rpy = np.array([float(row["roll"]), float(row["pitch"]), float(row["yaw"])], dtype=float)
            route[name] = {"pos": pos, "rpy": rpy}
    return route

def estimate_sim3_umeyama(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    seq = Path(args.sequence_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images_txt = seq / "sparse_txt" / "images.txt"
    route_csv = seq / "route_commanded.csv"

    colmap = read_colmap_images(images_txt)
    route = read_route_commanded(route_csv)

    common = sorted(set(colmap.keys()) & set(route.keys()))
    if len(common) < 3:
        raise RuntimeError(f"Need at least 3 common frames. Got {len(common)}")

    src = np.array([colmap[name]["C_colmap"] for name in common], dtype=float)
    dst = np.array([route[name]["pos"] for name in common], dtype=float)

    scale, R, t = estimate_sim3_umeyama(src, dst)
    pred = scale * (R @ src.T).T + t
    err = pred - dst
    err_norm = np.linalg.norm(err, axis=1)

    rows = []
    for name, p, gt, e, en in zip(common, pred, dst, err, err_norm):
        rows.append({
            "image_name": name,
            "colmap_aligned_x": f"{p[0]:.8f}",
            "colmap_aligned_y": f"{p[1]:.8f}",
            "colmap_aligned_z": f"{p[2]:.8f}",
            "gt_x": f"{gt[0]:.8f}",
            "gt_y": f"{gt[1]:.8f}",
            "gt_z": f"{gt[2]:.8f}",
            "err_x": f"{e[0]:.8f}",
            "err_y": f"{e[1]:.8f}",
            "err_z": f"{e[2]:.8f}",
            "err_norm_m": f"{en:.8f}",
        })

    with (out / "colmap_moving_camera_vs_commanded_gt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rmse = float(np.sqrt(np.mean(err_norm ** 2)))
    mean = float(np.mean(err_norm))
    median = float(np.median(err_norm))
    maxerr = float(np.max(err_norm))

    best_idx = int(np.argmin(err_norm))
    worst_idx = int(np.argmax(err_norm))

    text = f"""COLMAP MOVING-CAMERA TRAJECTORY VS COMMANDED/GT POSE
===================================================

This is evaluation only.
The no-GT calibration pipeline does not use route_commanded.csv.

sequence_dir:
  {seq}

Common registered frames:
  {len(common)}

Sim(3) COLMAP -> commanded/GT route:
  scale: {scale:.10f}

Position error:
  rmse_m:   {rmse:.6f}
  mean_m:   {mean:.6f}
  median_m: {median:.6f}
  max_m:    {maxerr:.6f}

Best frame:
  {common[best_idx]}  error_m={err_norm[best_idx]:.6f}

Worst frame:
  {common[worst_idx]}  error_m={err_norm[worst_idx]:.6f}

Output CSV:
  {out / 'colmap_moving_camera_vs_commanded_gt.csv'}
"""
    (out / "summary_colmap_moving_camera_vs_commanded_gt.txt").write_text(text)
    print(text)

if __name__ == "__main__":
    main()
