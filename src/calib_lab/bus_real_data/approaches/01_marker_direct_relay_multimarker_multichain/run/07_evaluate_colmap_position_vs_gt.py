#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def parse_frame_from_name(name):
    # frame_0123.png -> 123
    stem = Path(name).stem
    return int(stem.split("_")[1])


def read_colmap_centers(images_txt):
    centers = {}

    lines = images_txt.read_text(errors="ignore").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        if len(parts) >= 10:
            try:
                image_id = int(parts[0])
                qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
                tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
                camera_id = int(parts[8])
                name = parts[9]

                Rcw = qvec_to_rotmat(qvec)
                C = -Rcw.T @ tvec

                frame = parse_frame_from_name(name)
                centers[frame] = {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "image_name": name,
                    "colmap_x": float(C[0]),
                    "colmap_y": float(C[1]),
                    "colmap_z": float(C[2]),
                }

                # Skip points2D line.
                i += 2
                continue
            except Exception:
                pass

        i += 1

    return centers


def read_gt_route(route_csv):
    gt = {}

    with route_csv.open() as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])
            gt[frame] = {
                "gt_x": float(r["x"]),
                "gt_y": float(r["y"]),
                "gt_z": float(r["z"]),
                "gt_roll": float(r["roll"]),
                "gt_pitch": float(r["pitch"]),
                "gt_yaw": float(r["yaw"]),
            }

    return gt


def umeyama_sim3(X, Y):
    """
    Find s, R, t such that:
        Y ~= s * R * X + t

    X: Nx3 COLMAP centers
    Y: Nx3 GT centers
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_x = np.sum(Xc * Xc) / n
    scale = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - scale * (R @ mu_x)

    return scale, R, t


def apply_sim3(scale, R, t, X):
    return (scale * (R @ X.T)).T + t.reshape(1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="results/bus_real_data/05_moving_camera_sequence_run3")
    ap.add_argument("--colmap", default="results/bus_real_data/06_colmap_moving_sequence_run3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq_dir = Path(args.sequence)
    colmap_dir = Path(args.colmap)

    images_txt = colmap_dir / "sparse_txt_best" / "images.txt"
    route_csv = seq_dir / "route_commanded.csv"

    if not images_txt.exists():
        raise RuntimeError(f"Missing COLMAP images.txt: {images_txt}")

    if not route_csv.exists():
        raise RuntimeError(f"Missing GT route CSV: {route_csv}")

    out_dir = Path(args.out) if args.out else colmap_dir / "sim3_eval_vs_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap = read_colmap_centers(images_txt)
    gt = read_gt_route(route_csv)

    common_frames = sorted(set(colmap.keys()) & set(gt.keys()))

    if len(common_frames) < 3:
        raise RuntimeError(f"Need at least 3 common frames for Sim(3), got {len(common_frames)}")

    X = np.array([[colmap[f]["colmap_x"], colmap[f]["colmap_y"], colmap[f]["colmap_z"]] for f in common_frames], dtype=np.float64)
    Y = np.array([[gt[f]["gt_x"], gt[f]["gt_y"], gt[f]["gt_z"]] for f in common_frames], dtype=np.float64)

    scale, R, t = umeyama_sim3(X, Y)
    X_aligned = apply_sim3(scale, R, t, X)

    errors = np.linalg.norm(X_aligned - Y, axis=1)

    rmse = math.sqrt(float(np.mean(errors ** 2)))
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    max_err = float(np.max(errors))
    min_err = float(np.min(errors))

    rows = []

    for idx, frame in enumerate(common_frames):
        rows.append({
            "frame": frame,
            "image_name": colmap[frame]["image_name"],
            "colmap_x": X[idx, 0],
            "colmap_y": X[idx, 1],
            "colmap_z": X[idx, 2],
            "aligned_x": X_aligned[idx, 0],
            "aligned_y": X_aligned[idx, 1],
            "aligned_z": X_aligned[idx, 2],
            "gt_x": Y[idx, 0],
            "gt_y": Y[idx, 1],
            "gt_z": Y[idx, 2],
            "position_error_m": errors[idx],
            "gt_roll": gt[frame]["gt_roll"],
            "gt_pitch": gt[frame]["gt_pitch"],
            "gt_yaw": gt[frame]["gt_yaw"],
        })

    csv_path = out_dir / "sim3_aligned_trajectory_errors.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "frame", "image_name",
            "colmap_x", "colmap_y", "colmap_z",
            "aligned_x", "aligned_y", "aligned_z",
            "gt_x", "gt_y", "gt_z",
            "position_error_m",
            "gt_roll", "gt_pitch", "gt_yaw",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    transform_path = out_dir / "sim3_transform.txt"
    transform_path.write_text(
        "Sim(3) alignment: GT ~= scale * R * COLMAP + t\n\n"
        f"scale: {scale:.12f}\n\n"
        "R:\n"
        f"{R[0,0]:.12f} {R[0,1]:.12f} {R[0,2]:.12f}\n"
        f"{R[1,0]:.12f} {R[1,1]:.12f} {R[1,2]:.12f}\n"
        f"{R[2,0]:.12f} {R[2,1]:.12f} {R[2,2]:.12f}\n\n"
        "t:\n"
        f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f}\n"
    )

    report_path = out_dir / "sim3_evaluation_report.txt"
    report_path.write_text(
        "COLMAP moving-camera trajectory vs Ground Truth route\n"
        "=====================================================\n\n"
        "Evaluation type: Sim(3)-aligned position evaluation\n"
        "Important: This uses simulation Ground Truth and is for evaluation only.\n\n"
        f"Sequence: {seq_dir}\n"
        f"COLMAP model: {images_txt}\n"
        f"GT route: {route_csv}\n\n"
        f"Total GT frames: {len(gt)}\n"
        f"COLMAP registered frames: {len(colmap)}\n"
        f"Common evaluated frames: {len(common_frames)}\n\n"
        f"Scale: {scale:.12f}\n\n"
        "Position errors after Sim(3) alignment:\n"
        f"RMSE_m:   {rmse:.6f}\n"
        f"Mean_m:   {mean:.6f}\n"
        f"Median_m: {median:.6f}\n"
        f"Min_m:    {min_err:.6f}\n"
        f"Max_m:    {max_err:.6f}\n\n"
        f"CSV: {csv_path}\n"
        f"Transform: {transform_path}\n"
    )

    print()
    print("=== SIM(3) EVALUATION ===")
    print("GT frames:", len(gt))
    print("COLMAP registered frames:", len(colmap))
    print("Common evaluated frames:", len(common_frames))
    print("Scale:", scale)
    print("RMSE_m:", rmse)
    print("Mean_m:", mean)
    print("Median_m:", median)
    print("Max_m:", max_err)
    print()
    print("[OK] report:", report_path)
    print("[OK] csv:", csv_path)
    print("[OK] transform:", transform_path)


if __name__ == "__main__":
    main()
