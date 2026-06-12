#!/usr/bin/env python3

import argparse
import csv
import itertools
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


def rpy_to_rotmat(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotmat_to_rpy(R):
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

                # COLMAP stores world-to-camera.
                R_cw = qvec_to_rotmat(qvec)
                R_wc = R_cw.T

                out[frame] = {
                    "image_id": image_id,
                    "image_name": image_name,
                    "R_wc_colmap": R_wc,
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


def all_axis_convention_rotations():
    """
    All 24 proper axis permutation/sign rotations.
    These cover common fixed camera-axis convention offsets.
    """
    mats = []
    base = np.eye(3)

    for perm in itertools.permutations([0, 1, 2]):
        P = base[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            S = np.diag(signs)
            C = P @ S
            if np.linalg.det(C) > 0.5:
                mats.append(C)

    # remove duplicates
    unique = []
    for M in mats:
        if not any(np.allclose(M, U) for U in unique):
            unique.append(M)

    return unique


def estimate_global_rotation(col_rots, gt_rots, C):
    """
    Estimate A so that:
        R_gt ~= A * R_colmap * C

    This is Wahba/Kabsch on all camera basis vectors over all frames.
    """
    M = np.zeros((3, 3), dtype=np.float64)

    for Rc, Rg in zip(col_rots, gt_rots):
        B = Rc @ C
        A_target = Rg

        # Align the three basis vectors:
        # target vector = A_est * source vector
        for k in range(3):
            source = B[:, k].reshape(3, 1)
            target = A_target[:, k].reshape(3, 1)
            M += target @ source.T

    U, S, Vt = np.linalg.svd(M)
    A = U @ Vt

    if np.linalg.det(A) < 0:
        U[:, -1] *= -1
        A = U @ Vt

    return A


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

    out_dir = colmap_dir / "sim3_eval_vs_gt" / "rotation_eval_orientation_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap = read_colmap_rotations(images_txt)
    gt = read_gt_rotations(route_csv)

    common = sorted(set(colmap.keys()) & set(gt.keys()))
    if len(common) < 3:
        raise RuntimeError("Need at least 3 common frames")

    col_rots = [colmap[f]["R_wc_colmap"] for f in common]
    gt_rots = [gt[f]["R_gt"] for f in common]

    best = None

    for C_idx, C in enumerate(all_axis_convention_rotations()):
        A = estimate_global_rotation(col_rots, gt_rots, C)

        errs = []
        for Rc, Rg in zip(col_rots, gt_rots):
            R_est = A @ Rc @ C
            R_err = Rg.T @ R_est
            errs.append(rotation_angle_deg(R_err))

        score = rmse(errs)

        if best is None or score < best["rmse"]:
            best = {
                "C_idx": C_idx,
                "C": C,
                "A": A,
                "errs": errs,
                "rmse": score,
            }

    C = best["C"]
    A = best["A"]
    errs = best["errs"]

    rows = []
    for frame, Rc, Rg, err in zip(common, col_rots, gt_rots, errs):
        R_est = A @ Rc @ C
        eroll, epitch, eyaw = rotmat_to_rpy(R_est)

        rows.append({
            "frame": frame,
            "image_name": colmap[frame]["image_name"],
            "rotation_error_deg": err,

            "est_roll": eroll,
            "gt_roll": gt[frame]["gt_roll"],
            "roll_diff_deg_wrapped": math.degrees(math.atan2(math.sin(eroll - gt[frame]["gt_roll"]), math.cos(eroll - gt[frame]["gt_roll"]))),

            "est_pitch": epitch,
            "gt_pitch": gt[frame]["gt_pitch"],
            "pitch_diff_deg_wrapped": math.degrees(math.atan2(math.sin(epitch - gt[frame]["gt_pitch"]), math.cos(epitch - gt[frame]["gt_pitch"]))),

            "est_yaw": eyaw,
            "gt_yaw": gt[frame]["gt_yaw"],
            "yaw_diff_deg_wrapped": math.degrees(math.atan2(math.sin(eyaw - gt[frame]["gt_yaw"]), math.cos(eyaw - gt[frame]["gt_yaw"]))),
        })

    csv_path = out_dir / "rotation_errors_by_frame.csv"
    with csv_path.open("w", newline="") as f:
        fields = [
            "frame", "image_name", "rotation_error_deg",
            "est_roll", "gt_roll", "roll_diff_deg_wrapped",
            "est_pitch", "gt_pitch", "pitch_diff_deg_wrapped",
            "est_yaw", "gt_yaw", "yaw_diff_deg_wrapped",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    sorted_path = out_dir / "rotation_errors_sorted_worst_first.csv"
    with sorted_path.open("w", newline="") as f:
        fields = [
            "rank", "frame", "image_name", "rotation_error_deg",
            "roll_diff_deg_wrapped", "pitch_diff_deg_wrapped", "yaw_diff_deg_wrapped",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for rank, r in enumerate(sorted(rows, key=lambda x: x["rotation_error_deg"], reverse=True), start=1):
            w.writerow({
                "rank": rank,
                "frame": r["frame"],
                "image_name": r["image_name"],
                "rotation_error_deg": f"{r['rotation_error_deg']:.6f}",
                "roll_diff_deg_wrapped": f"{r['roll_diff_deg_wrapped']:.6f}",
                "pitch_diff_deg_wrapped": f"{r['pitch_diff_deg_wrapped']:.6f}",
                "yaw_diff_deg_wrapped": f"{r['yaw_diff_deg_wrapped']:.6f}",
            })

    transform_path = out_dir / "orientation_alignment_transforms.txt"
    transform_path.write_text(
        "Orientation alignment model\n"
        "===========================\n\n"
        "R_gt ~= A * R_colmap_camera_to_world * C\n\n"
        "A is a global world-frame rotation estimated from orientation correspondences.\n"
        "C is the best right-handed axis-convention offset among 24 axis permutations.\n\n"
        f"best_C_index: {best['C_idx']}\n\n"
        "A:\n"
        f"{A[0,0]:.12f} {A[0,1]:.12f} {A[0,2]:.12f}\n"
        f"{A[1,0]:.12f} {A[1,1]:.12f} {A[1,2]:.12f}\n"
        f"{A[2,0]:.12f} {A[2,1]:.12f} {A[2,2]:.12f}\n\n"
        "C:\n"
        f"{C[0,0]:.12f} {C[0,1]:.12f} {C[0,2]:.12f}\n"
        f"{C[1,0]:.12f} {C[1,1]:.12f} {C[1,2]:.12f}\n"
        f"{C[2,0]:.12f} {C[2,1]:.12f} {C[2,2]:.12f}\n"
    )

    summary_path = out_dir / "rotation_evaluation_report.txt"
    summary_path.write_text(
        "COLMAP moving-camera rotation vs Ground Truth route\n"
        "===================================================\n\n"
        "Evaluation type: orientation-aligned rotation evaluation.\n"
        "Important: This uses simulation Ground Truth and is for evaluation only.\n\n"
        "Why this exists:\n"
        "The previous rotation check used the Sim(3) rotation estimated from camera centers.\n"
        "Because the trajectory is almost 1D, that rotation is underconstrained for orientation evaluation.\n\n"
        "Model:\n"
        "R_gt ~= A * R_colmap_camera_to_world * C\n\n"
        f"Evaluated registered frames: {len(common)}\n"
        f"Best C index: {best['C_idx']}\n\n"
        "Rotation error is SO(3) geodesic angle in degrees.\n\n"
        f"RMSE_deg:   {rmse(errs):.6f}\n"
        f"Mean_deg:   {float(np.mean(errs)):.6f}\n"
        f"Median_deg: {float(np.median(errs)):.6f}\n"
        f"Min_deg:    {float(np.min(errs)):.6f}\n"
        f"Max_deg:    {float(np.max(errs)):.6f}\n\n"
        f"CSV: {csv_path}\n"
        f"Worst-first CSV: {sorted_path}\n"
        f"Alignment transforms: {transform_path}\n"
    )

    print()
    print("=== ORIENTATION-ALIGNED ROTATION EVALUATION ===")
    print("evaluated frames:", len(common))
    print("Best C index:", best["C_idx"])
    print("RMSE_deg:", rmse(errs))
    print("Mean_deg:", float(np.mean(errs)))
    print("Median_deg:", float(np.median(errs)))
    print("Min_deg:", float(np.min(errs)))
    print("Max_deg:", float(np.max(errs)))
    print()
    print("[OK] report:", summary_path)
    print("[OK] csv:", csv_path)
    print("[OK] sorted:", sorted_path)


if __name__ == "__main__":
    main()
