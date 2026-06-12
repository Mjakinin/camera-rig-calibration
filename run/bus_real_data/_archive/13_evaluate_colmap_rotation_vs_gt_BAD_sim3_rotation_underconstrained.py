#!/usr/bin/env python3

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def rpy_to_rotmat(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotmat_to_rpy(R):
    # Matches R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rotation_angle_deg(R):
    v = (np.trace(R) - 1.0) / 2.0
    v = max(-1.0, min(1.0, float(v)))
    return math.degrees(math.acos(v))


def parse_frame_from_name(name):
    return int(Path(name).stem.split("_")[1])


def read_colmap_rotations(images_txt):
    out = {}

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
                image_name = parts[9]
                frame = parse_frame_from_name(image_name)

                # COLMAP qvec is world-to-camera. Camera-to-world is transpose.
                R_cw_colmap = qvec_to_rotmat(qvec)
                R_wc_colmap = R_cw_colmap.T

                out[frame] = {
                    "image_id": image_id,
                    "image_name": image_name,
                    "R_wc_colmap": R_wc_colmap,
                }

                i += 2
                continue
            except Exception:
                pass

        i += 1

    return out


def read_gt_rotations(route_csv):
    out = {}

    with route_csv.open() as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])
            roll = float(r["roll"])
            pitch = float(r["pitch"])
            yaw = float(r["yaw"])

            out[frame] = {
                "gt_roll": roll,
                "gt_pitch": pitch,
                "gt_yaw": yaw,
                "R_gt": rpy_to_rotmat(roll, pitch, yaw),
            }

    return out


def read_sim3_rotation(transform_txt):
    text = transform_txt.read_text()

    m = re.search(
        r"R:\s*\n"
        r"\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\n"
        r"\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\n"
        r"\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        text
    )

    if not m:
        raise RuntimeError(f"Could not parse R from {transform_txt}")

    vals = [float(v) for v in m.groups()]
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def estimate_constant_camera_axis_offset(R_colmap_to_gt_list, R_gt_list):
    """
    Find constant C so that:
        R_gt ~= R_colmap_to_gt @ C

    C absorbs the fixed coordinate-frame convention difference between
    COLMAP camera axes and the Gazebo moving camera entity axes.
    """
    M = np.zeros((3, 3), dtype=np.float64)

    for A, B in zip(R_colmap_to_gt_list, R_gt_list):
        M += A.T @ B

    U, S, Vt = np.linalg.svd(M)
    C = U @ Vt

    if np.linalg.det(C) < 0:
        U[:, -1] *= -1
        C = U @ Vt

    return C


def rmse(vals):
    vals = list(vals)
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="results/bus_real_data/03_moving_camera_sequence")
    ap.add_argument("--colmap", default="results/bus_real_data/04_colmap_moving_sequence")
    args = ap.parse_args()

    seq_dir = Path(args.sequence)
    colmap_dir = Path(args.colmap)

    images_txt = colmap_dir / "sparse_txt_best" / "images.txt"
    route_csv = seq_dir / "route_commanded.csv"
    transform_txt = colmap_dir / "sim3_eval_vs_gt" / "sim3_transform.txt"

    out_dir = colmap_dir / "sim3_eval_vs_gt" / "rotation_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap = read_colmap_rotations(images_txt)
    gt = read_gt_rotations(route_csv)
    R_sim3 = read_sim3_rotation(transform_txt)

    common = sorted(set(colmap.keys()) & set(gt.keys()))
    if len(common) < 3:
        raise RuntimeError("Need at least 3 common frames")

    R_colmap_to_gt_list = []
    R_gt_list = []

    for frame in common:
        R_colmap_to_gt = R_sim3 @ colmap[frame]["R_wc_colmap"]
        R_colmap_to_gt_list.append(R_colmap_to_gt)
        R_gt_list.append(gt[frame]["R_gt"])

    C = estimate_constant_camera_axis_offset(R_colmap_to_gt_list, R_gt_list)

    rows = []
    angle_errors = []

    for frame, R_colmap_to_gt, R_gt in zip(common, R_colmap_to_gt_list, R_gt_list):
        R_est = R_colmap_to_gt @ C
        R_err = R_gt.T @ R_est
        angle_deg = rotation_angle_deg(R_err)
        angle_errors.append(angle_deg)

        est_roll, est_pitch, est_yaw = rotmat_to_rpy(R_est)

        rows.append({
            "frame": frame,
            "image_name": colmap[frame]["image_name"],

            "est_roll": est_roll,
            "gt_roll": gt[frame]["gt_roll"],
            "roll_diff_wrapped": math.atan2(math.sin(est_roll - gt[frame]["gt_roll"]), math.cos(est_roll - gt[frame]["gt_roll"])),

            "est_pitch": est_pitch,
            "gt_pitch": gt[frame]["gt_pitch"],
            "pitch_diff_wrapped": math.atan2(math.sin(est_pitch - gt[frame]["gt_pitch"]), math.cos(est_pitch - gt[frame]["gt_pitch"])),

            "est_yaw": est_yaw,
            "gt_yaw": gt[frame]["gt_yaw"],
            "yaw_diff_wrapped": math.atan2(math.sin(est_yaw - gt[frame]["gt_yaw"]), math.cos(est_yaw - gt[frame]["gt_yaw"])),

            "rotation_error_deg": angle_deg,
        })

    csv_path = out_dir / "rotation_errors_by_frame.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "frame", "image_name",
            "est_roll", "gt_roll", "roll_diff_wrapped",
            "est_pitch", "gt_pitch", "pitch_diff_wrapped",
            "est_yaw", "gt_yaw", "yaw_diff_wrapped",
            "rotation_error_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    sorted_csv = out_dir / "rotation_errors_sorted_worst_first.csv"
    with sorted_csv.open("w", newline="") as f:
        fields = [
            "rank", "frame", "image_name", "rotation_error_deg",
            "roll_diff_deg", "pitch_diff_deg", "yaw_diff_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for rank, r in enumerate(sorted(rows, key=lambda x: x["rotation_error_deg"], reverse=True), start=1):
            w.writerow({
                "rank": rank,
                "frame": r["frame"],
                "image_name": r["image_name"],
                "rotation_error_deg": f"{r['rotation_error_deg']:.6f}",
                "roll_diff_deg": f"{math.degrees(r['roll_diff_wrapped']):.6f}",
                "pitch_diff_deg": f"{math.degrees(r['pitch_diff_wrapped']):.6f}",
                "yaw_diff_deg": f"{math.degrees(r['yaw_diff_wrapped']):.6f}",
            })

    C_path = out_dir / "estimated_camera_axis_offset.txt"
    C_path.write_text(
        "Constant camera-axis offset C\n"
        "=============================\n\n"
        "Used model:\n"
        "R_gt ~= R_sim3 * R_colmap_camera_to_world * C\n\n"
        "This C absorbs fixed COLMAP-vs-Gazebo camera axis convention differences.\n\n"
        "C:\n"
        f"{C[0,0]:.12f} {C[0,1]:.12f} {C[0,2]:.12f}\n"
        f"{C[1,0]:.12f} {C[1,1]:.12f} {C[1,2]:.12f}\n"
        f"{C[2,0]:.12f} {C[2,1]:.12f} {C[2,2]:.12f}\n"
    )

    summary_path = out_dir / "rotation_evaluation_report.txt"

    summary_path.write_text(
        "COLMAP moving-camera rotation vs Ground Truth route\n"
        "===================================================\n\n"
        "Evaluation type: Sim(3)-rotation + constant camera-axis offset alignment.\n"
        "Important: This uses simulation Ground Truth and is for evaluation only.\n\n"
        f"Evaluated registered frames: {len(common)}\n\n"
        "Rotation error is reported as geodesic SO(3) angle in degrees.\n"
        "This is more reliable than comparing Euler angles directly.\n\n"
        f"RMSE_deg:   {rmse(angle_errors):.6f}\n"
        f"Mean_deg:   {float(np.mean(angle_errors)):.6f}\n"
        f"Median_deg: {float(np.median(angle_errors)):.6f}\n"
        f"Min_deg:    {float(np.min(angle_errors)):.6f}\n"
        f"Max_deg:    {float(np.max(angle_errors)):.6f}\n\n"
        f"CSV: {csv_path}\n"
        f"Worst-first CSV: {sorted_csv}\n"
        f"Camera-axis offset: {C_path}\n"
    )

    print()
    print("=== ROTATION EVALUATION ===")
    print("evaluated frames:", len(common))
    print("RMSE_deg:", rmse(angle_errors))
    print("Mean_deg:", float(np.mean(angle_errors)))
    print("Median_deg:", float(np.median(angle_errors)))
    print("Max_deg:", float(np.max(angle_errors)))
    print()
    print("[OK] report:", summary_path)
    print("[OK] csv:", csv_path)
    print("[OK] sorted:", sorted_csv)


if __name__ == "__main__":
    main()
