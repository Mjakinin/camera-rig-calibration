#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


R_OPT_LINK = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)


STATIC_CAMERA_POSES = {
    "front_static_camera": (-3.90, 0.0, 2.85, 0.0, 0.69813170, 0.0),
    "rear_static_camera":  ( 5.70, 0.0, 2.85, 0.0, 0.69813170, math.pi),
}


BOARD_GT_POSES = {
    # x, y, z, roll, pitch, yaw
    "F3": (-3.25,  0.71, 1.50, 0.0,        0.35,       0.0),
    "F4": (-2.46, -0.75, 1.62, 1.57079633, 1.57079633, 0.0),
    "R1": ( 3.02, -0.75, 1.55, 0.0,        0.30,       math.pi),
    "R3": ( 4.27,  0.93, 1.55, 0.0,        0.30,       1.57079633),
}


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
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def T_world_link_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    return make_T(R_world_link, np.array([x, y, z], dtype=float))


def T_optical_world_from_gazebo_camera_pose(x, y, z, roll, pitch, yaw):
    """
    Returns T_camera_world:
      X_camera = T_camera_world * X_world

    Gazebo link frame:
      x forward, y left, z up

    OpenCV optical frame:
      x right, y down, z forward
    """
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)


def T_static_world(camera_name):
    return T_optical_world_from_gazebo_camera_pose(*STATIC_CAMERA_POSES[camera_name])


def T_board_world(station):
    """
    Returns T_board_world:
      X_board = T_board_world * X_world
    """
    T_world_board = T_world_link_from_gazebo_pose(*BOARD_GT_POSES[station])
    return inv_T(T_world_board)


def T_world_board(station):
    return T_world_link_from_gazebo_pose(*BOARD_GT_POSES[station])


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def read_route(path):
    return {r["image_name"]: r for r in read_csv(path)}


def parse_value(text, key):
    m = re.search(rf"{re.escape(key)}:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def parse_pair_frames(pair_result_dir):
    summary = Path(pair_result_dir) / "summary_no_gt.txt"
    if not summary.exists():
        raise FileNotFoundError(summary)

    text = summary.read_text()
    front_image = parse_value(text, "front image")
    rear_image = parse_value(text, "rear image")

    if not front_image or not rear_image:
        raise RuntimeError(f"Could not parse front/rear images from {summary}")

    return front_image, rear_image


def route_row_to_T_moving_world(route_row):
    return T_optical_world_from_gazebo_camera_pose(
        float(route_row["x"]),
        float(route_row["y"]),
        float(route_row["z"]),
        float(route_row["roll"]),
        float(route_row["pitch"]),
        float(route_row["yaw"]),
    )


def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def eval_T(T_est, T_gt):
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    baseline_est = float(np.linalg.norm(t_est))
    baseline_gt = float(np.linalg.norm(t_gt))
    baseline_error_cm = (baseline_est - baseline_gt) * 100.0
    translation_error_cm = float(np.linalg.norm(t_est - t_gt) * 100.0)
    rotation_error = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])

    return {
        "baseline_est_m": baseline_est,
        "baseline_gt_m": baseline_gt,
        "baseline_error_cm": baseline_error_cm,
        "translation_error_cm": translation_error_cm,
        "rotation_error_deg": rotation_error,
    }


def save_matrix(path, T):
    np.savetxt(path, T, delimiter=",", fmt="%.10f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--chain_results_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    chain_results_dir = Path(args.chain_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route = read_route(sequence_dir / "route_commanded.csv")

    T_front_world_gt = T_static_world("front_static_camera")
    T_rear_world_gt = T_static_world("rear_static_camera")
    T_front_rear_gt = T_front_world_gt @ inv_T(T_rear_world_gt)

    pair_dirs = sorted(chain_results_dir.glob("*_board_scaled_colmap"))
    if not pair_dirs:
        raise RuntimeError(f"No pair dirs found in {chain_results_dir}")

    rows = []

    for pair_dir in pair_dirs:
        pair = pair_dir.name.replace("_board_scaled_colmap", "")
        front_station, rear_station = pair.split("_")

        front_image, rear_image = parse_pair_frames(pair_dir)

        if front_image not in route:
            raise RuntimeError(f"Missing front_image in route_commanded.csv: {front_image}")
        if rear_image not in route:
            raise RuntimeError(f"Missing rear_image in route_commanded.csv: {rear_image}")

        # Exact GT camera poses at the exact frames selected by the no-GT chain.
        T_movingF_world_gt = route_row_to_T_moving_world(route[front_image])
        T_movingR_world_gt = route_row_to_T_moving_world(route[rear_image])

        # Exact GT board poses.
        T_world_boardF_gt = T_world_board(front_station)
        T_world_boardR_gt = T_world_board(rear_station)

        # Exact GT camera-board transforms.
        T_front_boardF_gt = T_front_world_gt @ T_world_boardF_gt
        T_movingF_boardF_gt = T_movingF_world_gt @ T_world_boardF_gt

        T_rear_boardR_gt = T_rear_world_gt @ T_world_boardR_gt
        T_movingR_boardR_gt = T_movingR_world_gt @ T_world_boardR_gt

        # Chain.
        T_front_movingF_gt = T_front_boardF_gt @ inv_T(T_movingF_boardF_gt)
        T_movingF_movingR_gt = T_movingF_world_gt @ inv_T(T_movingR_world_gt)
        T_movingR_rear_gt = T_movingR_boardR_gt @ inv_T(T_rear_boardR_gt)

        T_front_rear_all_gt = (
            T_front_movingF_gt
            @ T_movingF_movingR_gt
            @ T_movingR_rear_gt
        )

        metrics = eval_T(T_front_rear_all_gt, T_front_rear_gt)

        out_pair = output_dir / pair
        out_pair.mkdir(parents=True, exist_ok=True)

        save_matrix(out_pair / "T_front_rear_all_gt_exact_frame.csv", T_front_rear_all_gt)
        save_matrix(out_pair / "T_front_rear_static_gt.csv", T_front_rear_gt)
        save_matrix(out_pair / "T_movingF_movingR_gt_exact_frame.csv", T_movingF_movingR_gt)

        front_route = route[front_image]
        rear_route = route[rear_image]

        row = {
            "pair": pair,
            "front_station": front_station,
            "rear_station": rear_station,
            "front_image": front_image,
            "rear_image": rear_image,
            "front_gt_x": front_route["x"],
            "front_gt_y": front_route["y"],
            "front_gt_z": front_route["z"],
            "front_gt_roll": front_route["roll"],
            "front_gt_pitch": front_route["pitch"],
            "front_gt_yaw": front_route["yaw"],
            "rear_gt_x": rear_route["x"],
            "rear_gt_y": rear_route["y"],
            "rear_gt_z": rear_route["z"],
            "rear_gt_roll": rear_route["roll"],
            "rear_gt_pitch": rear_route["pitch"],
            "rear_gt_yaw": rear_route["yaw"],
            "baseline_est_m": f"{metrics['baseline_est_m']:.10f}",
            "baseline_gt_m": f"{metrics['baseline_gt_m']:.10f}",
            "baseline_error_cm": f"{metrics['baseline_error_cm']:.10f}",
            "translation_error_cm": f"{metrics['translation_error_cm']:.10f}",
            "rotation_error_deg": f"{metrics['rotation_error_deg']:.10f}",
        }
        rows.append(row)

        text = f"""ALL-GT EXACT-FRAME CHAIN SANITY CHECK
====================================

This is the real GT sanity check.

Used GT components:
- GT front_static_camera pose
- GT rear_static_camera pose
- GT moving_calib_camera pose at the exact selected front/rear frames
- GT front/rear board poses from the SDF world

No COLMAP trajectory is used.
No PnP board pose is used.
No trajectory alignment is used.

Pair:
  {pair}

Selected frames:
  front image: {front_image}
  rear image:  {rear_image}

Front moving GT pose:
  x y z:          {front_route['x']}, {front_route['y']}, {front_route['z']}
  roll pitch yaw: {front_route['roll']}, {front_route['pitch']}, {front_route['yaw']}

Rear moving GT pose:
  x y z:          {rear_route['x']}, {rear_route['y']}, {rear_route['z']}
  roll pitch yaw: {rear_route['roll']}, {rear_route['pitch']}, {rear_route['yaw']}

Metrics against static-camera GT:
  baseline_est_m:       {metrics['baseline_est_m']:.10f}
  baseline_gt_m:        {metrics['baseline_gt_m']:.10f}
  baseline_error_cm:    {metrics['baseline_error_cm']:.10f}
  translation_error_cm: {metrics['translation_error_cm']:.10f}
  rotation_error_deg:   {metrics['rotation_error_deg']:.10f}

Expected:
  All errors should be approximately zero.
"""
        (out_pair / "summary_all_gt_exact_frame_chain.txt").write_text(text)

    eval_csv = output_dir / "all_gt_exact_frame_chain_evaluation.csv"
    with eval_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    eval_md = output_dir / "all_gt_exact_frame_chain_evaluation.md"
    with eval_md.open("w") as f:
        f.write("# All-GT exact-frame chain sanity check\n\n")
        f.write("This run uses GT static cameras, GT moving camera poses, and GT board poses.\n\n")
        f.write("No COLMAP trajectory, no PnP board poses, and no trajectory alignment are used.\n\n")
        f.write("| Pair | Front frame | Rear frame | Baseline error [cm] | Translation error [cm] | Rotation error [deg] |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['pair']} | {r['front_image']} | {r['rear_image']} | "
                f"{float(r['baseline_error_cm']):.6f} | "
                f"{float(r['translation_error_cm']):.6f} | "
                f"{float(r['rotation_error_deg']):.6f} |\n"
            )

    (output_dir / "README.md").write_text("""# 03 All-GT Exact-Frame Chain

This folder contains the true GT sanity check.

Used components:
- GT front_static_camera pose
- GT rear_static_camera pose
- GT moving_calib_camera pose from route_commanded.csv at the exact selected frames
- GT ArUco board poses from the SDF world

Not used:
- COLMAP trajectory
- ArUco-PnP board pose estimates
- trajectory alignment

Expected result:
All errors should be approximately zero.

Purpose:
This validates the transform chain algebra and frame convention.
""")

    print("")
    print("[OK] wrote:", eval_csv)
    print("[OK] wrote:", eval_md)
    print("")
    print("Quick view:")
    for r in rows:
        print(
            f"{r['pair']:6s} "
            f"baseline_error_cm={float(r['baseline_error_cm']): .6f} "
            f"translation_error_cm={float(r['translation_error_cm']): .6f} "
            f"rotation_error_deg={float(r['rotation_error_deg']): .6f}"
        )


if __name__ == "__main__":
    main()
