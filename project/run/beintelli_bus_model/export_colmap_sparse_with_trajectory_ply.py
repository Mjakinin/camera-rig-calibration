#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


def read_points3d(points_txt: Path):
    pts = []

    with points_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK...
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            pts.append((np.array([x, y, z], dtype=float), r, g, b))

    return pts


def load_sim3(eval_dir: Path):
    s = float(np.loadtxt(eval_dir / "sim3_colmap_to_gt_scale.txt"))
    R = np.loadtxt(eval_dir / "sim3_colmap_to_gt_R.csv", delimiter=",")
    t = np.loadtxt(eval_dir / "sim3_colmap_to_gt_t.csv", delimiter=",").reshape(3)
    return s, R, t


def transform_point(p, s, R, t):
    return s * (R @ p) + t


def read_traj_csv(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_ply(path: Path, vertices):
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    points_txt = dataset / "sparse_txt" / "points3D.txt"
    eval_dir = dataset / "trajectory_eval"
    traj_csv = eval_dir / "colmap_vs_gt_trajectory.csv"

    if not points_txt.exists():
        raise FileNotFoundError(points_txt)
    if not traj_csv.exists():
        raise FileNotFoundError(traj_csv)

    s, R, t = load_sim3(eval_dir)
    sparse_points = read_points3d(points_txt)
    traj_rows = read_traj_csv(traj_csv)

    vertices = []

    # Sparse COLMAP points transformed into GT coordinates.
    for p, r, g, b in sparse_points:
        pt = transform_point(p, s, R, t)
        vertices.append((pt[0], pt[1], pt[2], r, g, b))

    # GT camera centers: green.
    for row in traj_rows:
        vertices.append((
            float(row["gt_x"]),
            float(row["gt_y"]),
            float(row["gt_z"]),
            0, 255, 0
        ))

    # Aligned COLMAP camera centers: red.
    for row in traj_rows:
        vertices.append((
            float(row["aligned_x"]),
            float(row["aligned_y"]),
            float(row["aligned_z"]),
            255, 0, 0
        ))

    out = eval_dir / "sparse_points_gt_space_with_gt_green_colmap_red.ply"
    write_ply(out, vertices)

    print(f"[OK] wrote {out}")
    print(f"[INFO] sparse points: {len(sparse_points)}")
    print(f"[INFO] camera centers each: {len(traj_rows)}")


if __name__ == "__main__":
    main()
