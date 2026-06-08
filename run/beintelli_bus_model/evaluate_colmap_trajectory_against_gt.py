#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec):
    # COLMAP qvec order: qw, qx, qy, qz
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=float)


def parse_colmap_images_txt(path: Path):
    images = {}

    with path.open() as f:
        lines = [line.strip() for line in f]

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()

        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array(list(map(float, parts[1:5])), dtype=float)
            tvec = np.array(list(map(float, parts[5:8])), dtype=float)
            camera_id = int(parts[8])
            name = parts[9]

            # COLMAP stores world-to-camera:
            # x_cam = R_cw * x_world + t_cw
            # camera center in COLMAP world:
            # C = -R_cw^T * t_cw
            R_cw = qvec_to_rotmat(qvec)
            C = -R_cw.T @ tvec

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "R_cw": R_cw,
                "center_colmap": C,
            }

            i += 2  # skip 2D observations line
        else:
            i += 1

    return images


def parse_route_gt(path: Path):
    gt = {}

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["image_name"]] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "roll": float(row["roll"]),
                "pitch": float(row["pitch"]),
                "yaw": float(row["yaw"]),
                "tag": row.get("tag", ""),
            }

    return gt


def umeyama_similarity(X, Y):
    """
    Estimate similarity transform Y ~= s * R * X + t.
    X: Nx3 source points, COLMAP centers
    Y: Nx3 target points, Gazebo GT centers
    """
    assert X.shape == Y.shape
    n = X.shape[0]

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = np.sum(Xc * Xc) / n

    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_x
    t = mu_y - s * (R @ mu_x)

    return s, R, t


def write_ply(path: Path, points):
    """
    points: list of (x,y,z,r,g,b)
    """
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    images_txt = dataset / "sparse_txt" / "images.txt"
    route_gt = dataset / "route_gt.csv"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = dataset / "trajectory_eval"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_txt.exists():
        raise FileNotFoundError(images_txt)
    if not route_gt.exists():
        raise FileNotFoundError(route_gt)

    colmap = parse_colmap_images_txt(images_txt)
    gt = parse_route_gt(route_gt)

    common_names = sorted(set(colmap.keys()) & set(gt.keys()))

    if len(common_names) < 3:
        raise RuntimeError("Need at least 3 common registered images for Sim(3) alignment.")

    X = np.array([colmap[name]["center_colmap"] for name in common_names], dtype=float)
    Y = np.array([[gt[name]["x"], gt[name]["y"], gt[name]["z"]] for name in common_names], dtype=float)

    s, R, t = umeyama_similarity(X, Y)
    X_aligned = (s * (R @ X.T)).T + t

    errors = np.linalg.norm(X_aligned - Y, axis=1)

    rmse = math.sqrt(float(np.mean(errors ** 2)))
    mean = float(np.mean(errors))
    median = float(np.median(errors))
    max_err = float(np.max(errors))

    print("COLMAP TRAJECTORY VS GAZEBO GT")
    print("==============================")
    print(f"dataset:             {dataset}")
    print(f"registered/common:   {len(common_names)}")
    print(f"similarity scale:    {s:.6f}")
    print(f"position RMSE [m]:   {rmse:.6f}")
    print(f"position mean [m]:   {mean:.6f}")
    print(f"position median [m]: {median:.6f}")
    print(f"position max [m]:    {max_err:.6f}")
    print(f"position RMSE [cm]:  {rmse * 100.0:.2f}")
    print(f"position mean [cm]:  {mean * 100.0:.2f}")
    print(f"position max [cm]:   {max_err * 100.0:.2f}")

    # Save CSV
    csv_path = output_dir / "colmap_vs_gt_trajectory.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "tag",
            "colmap_x", "colmap_y", "colmap_z",
            "aligned_x", "aligned_y", "aligned_z",
            "gt_x", "gt_y", "gt_z",
            "pos_error_m",
        ])

        for name, xc, xa, yg, err in zip(common_names, X, X_aligned, Y, errors):
            writer.writerow([
                name,
                gt[name]["tag"],
                xc[0], xc[1], xc[2],
                xa[0], xa[1], xa[2],
                yg[0], yg[1], yg[2],
                err,
            ])

    # Save transform
    np.savetxt(output_dir / "sim3_colmap_to_gt_scale.txt", np.array([s]))
    np.savetxt(output_dir / "sim3_colmap_to_gt_R.csv", R, delimiter=",")
    np.savetxt(output_dir / "sim3_colmap_to_gt_t.csv", t.reshape(1, 3), delimiter=",")

    # Save PLY with GT and aligned COLMAP camera centers.
    # GT: green, aligned COLMAP: red.
    ply_points = []

    for yg in Y:
        ply_points.append((yg[0], yg[1], yg[2], 0, 255, 0))

    for xa in X_aligned:
        ply_points.append((xa[0], xa[1], xa[2], 255, 0, 0))

    write_ply(output_dir / "gt_green_colmap_aligned_red_camera_centers.ply", ply_points)

    # Save summary
    summary_path = output_dir / "colmap_vs_gt_summary.txt"
    summary_path.write_text(
        "COLMAP TRAJECTORY VS GAZEBO GT\n"
        "==============================\n"
        f"dataset:             {dataset}\n"
        f"registered/common:   {len(common_names)}\n"
        f"similarity scale:    {s:.6f}\n"
        f"position RMSE [m]:   {rmse:.6f}\n"
        f"position mean [m]:   {mean:.6f}\n"
        f"position median [m]: {median:.6f}\n"
        f"position max [m]:    {max_err:.6f}\n"
        f"position RMSE [cm]:  {rmse * 100.0:.2f}\n"
        f"position mean [cm]:  {mean * 100.0:.2f}\n"
        f"position max [cm]:   {max_err * 100.0:.2f}\n"
        f"\nCSV: {csv_path}\n"
    )

    print("")
    print(f"[OK] wrote CSV:     {csv_path}")
    print(f"[OK] wrote summary: {summary_path}")
    print(f"[OK] wrote PLY:     {output_dir / 'gt_green_colmap_aligned_red_camera_centers.ply'}")


if __name__ == "__main__":
    main()
