#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def T_from_xyz_rpy(pose):
    x, y, z, roll, pitch, yaw = pose
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def gazebo_link_T_cv_optical():
    """
    Gazebo camera is oriented as +X forward in our SDF setup.
    OpenCV optical frame is +Z forward, +X right, +Y down.

    Columns are OpenCV optical axes expressed in Gazebo link frame:
      x_optical/right = -Y_link
      y_optical/down  = -Z_link
      z_optical/front = +X_link
    """
    R = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def T_world_cv_camera_from_sdf_pose(pose):
    return T_from_xyz_rpy(pose) @ gazebo_link_T_cv_optical()


def T_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array(tvec, dtype=np.float64).reshape(3)
    return T


def inv(T):
    return np.linalg.inv(T)


def rotation_error_deg(R_est, R_gt):
    dR = R_est.T @ R_gt
    value = (np.trace(dR) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def read_valid_pose(csv_path, camera_name):
    with Path(csv_path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera"] != camera_name:
                continue

            if row["status"] != "pose_valid":
                raise RuntimeError(
                    f"{camera_name} in {csv_path} is not pose_valid. "
                    f"status={row['status']}, detected={row['detected_ids']}"
                )

            rvec = [
                float(row["rvec_x"]),
                float(row["rvec_y"]),
                float(row["rvec_z"]),
            ]

            tvec = [
                float(row["tvec_x_m"]),
                float(row["tvec_y_m"]),
                float(row["tvec_z_m"]),
            ]

            return {
                "camera": camera_name,
                "status": row["status"],
                "used_ids": row["used_ids"],
                "num_used_markers": int(row["num_used_markers"]),
                "num_points": int(row["num_points"]),
                "rmse_px": float(row["reprojection_rmse_px"]),
                "rvec": rvec,
                "tvec": tvec,
                "T_camera_board": T_from_rvec_tvec(rvec, tvec),
            }

    raise RuntimeError(f"Camera {camera_name} not found in {csv_path}")


def save_matrix(path, T):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in T:
            writer.writerow([f"{v:.10f}" for v in row])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--front_scene_csv",
        default="results/beintelli_bus_model/aruco_board_pose/relay_front_F1/aruco_board_pose_observations.csv",
    )
    parser.add_argument(
        "--rear_scene_csv",
        default="results/beintelli_bus_model/aruco_board_pose/relay_rear_R1/aruco_board_pose_observations.csv",
    )

    parser.add_argument(
        "--output_dir",
        default="results/beintelli_bus_model/aruco_board_relay/relay_F1_R1",
    )

    # SDF / Gazebo poses: x y z roll pitch yaw
    parser.add_argument("--front_static_pose", nargs=6, type=float, default=[-3.90, 0.0, 2.85, 0.0, 0.69813170, 0.0])
    parser.add_argument("--rear_static_pose", nargs=6, type=float, default=[5.70, 0.0, 2.85, 0.0, 0.69813170, 3.14159265])
    parser.add_argument("--moving_front_pose", nargs=6, type=float, default=[-1.20, 0.0, 1.65, 0.0, 0.0, 0.0])
    parser.add_argument("--moving_rear_pose", nargs=6, type=float, default=[1.80, 0.0, 1.65, 0.0, 0.0, 3.14159265])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # solvePnP returns T_camera_board:
    # X_camera = T_camera_board * X_board
    front_F = read_valid_pose(args.front_scene_csv, "front_static_camera")
    moving_F = read_valid_pose(args.front_scene_csv, "moving_calib_camera")

    rear_R = read_valid_pose(args.rear_scene_csv, "rear_static_camera")
    moving_R = read_valid_pose(args.rear_scene_csv, "moving_calib_camera")

    T_front_board_F = front_F["T_camera_board"]
    T_movingF_board_F = moving_F["T_camera_board"]

    T_rear_board_R = rear_R["T_camera_board"]
    T_movingR_board_R = moving_R["T_camera_board"]

    # Link from moving camera to static camera via the same board observation.
    # Maps moving_F camera frame -> front camera frame.
    T_front_movingF = T_front_board_F @ inv(T_movingF_board_F)

    # Maps moving_R camera frame -> rear camera frame.
    T_rear_movingR = T_rear_board_R @ inv(T_movingR_board_R)

    # Known moving-camera motion from Gazebo GT.
    T_world_front = T_world_cv_camera_from_sdf_pose(args.front_static_pose)
    T_world_rear = T_world_cv_camera_from_sdf_pose(args.rear_static_pose)
    T_world_movingF = T_world_cv_camera_from_sdf_pose(args.moving_front_pose)
    T_world_movingR = T_world_cv_camera_from_sdf_pose(args.moving_rear_pose)

    # Maps moving_R camera frame -> moving_F camera frame.
    T_movingF_movingR_gt = inv(T_world_movingF) @ T_world_movingR

    # Relay estimate:
    # rear -> moving_R -> moving_F -> front
    T_front_rear_est = (
        T_front_movingF
        @ T_movingF_movingR_gt
        @ inv(T_rear_movingR)
    )

    # Ground truth static transform:
    # maps rear camera frame -> front camera frame.
    T_front_rear_gt = inv(T_world_front) @ T_world_rear

    trans_est = T_front_rear_est[:3, 3]
    trans_gt = T_front_rear_gt[:3, 3]

    baseline_est = float(np.linalg.norm(trans_est))
    baseline_gt = float(np.linalg.norm(trans_gt))
    translation_error = float(np.linalg.norm(trans_est - trans_gt))
    rotation_error = rotation_error_deg(T_front_rear_est[:3, :3], T_front_rear_gt[:3, :3])

    save_matrix(output_dir / "T_front_rear_est.csv", T_front_rear_est)
    save_matrix(output_dir / "T_front_rear_gt.csv", T_front_rear_gt)
    save_matrix(output_dir / "T_front_movingF.csv", T_front_movingF)
    save_matrix(output_dir / "T_rear_movingR.csv", T_rear_movingR)
    save_matrix(output_dir / "T_movingF_movingR_gt.csv", T_movingF_movingR_gt)

    summary = (
        "BUS ARUCO BOARD RELAY SUMMARY\n"
        "==============================\n"
        "\n"
        "Relay convention:\n"
        "  T_front_rear maps points from rear camera optical frame into front camera optical frame.\n"
        "\n"
        "Input observations:\n"
        f"  front scene CSV: {args.front_scene_csv}\n"
        f"  rear scene CSV:  {args.rear_scene_csv}\n"
        "\n"
        "Front relay scene:\n"
        f"  front used IDs:         {front_F['used_ids']}\n"
        f"  front markers/points:   {front_F['num_used_markers']} / {front_F['num_points']}\n"
        f"  front RMSE px:          {front_F['rmse_px']:.6f}\n"
        f"  moving_F used IDs:      {moving_F['used_ids']}\n"
        f"  moving_F markers/pts:   {moving_F['num_used_markers']} / {moving_F['num_points']}\n"
        f"  moving_F RMSE px:       {moving_F['rmse_px']:.6f}\n"
        "\n"
        "Rear relay scene:\n"
        f"  rear used IDs:          {rear_R['used_ids']}\n"
        f"  rear markers/points:    {rear_R['num_used_markers']} / {rear_R['num_points']}\n"
        f"  rear RMSE px:           {rear_R['rmse_px']:.6f}\n"
        f"  moving_R used IDs:      {moving_R['used_ids']}\n"
        f"  moving_R markers/pts:   {moving_R['num_used_markers']} / {moving_R['num_points']}\n"
        f"  moving_R RMSE px:       {moving_R['rmse_px']:.6f}\n"
        "\n"
        "Estimated static-camera transform:\n"
        f"  est translation xyz [m]: {trans_est[0]:.6f}, {trans_est[1]:.6f}, {trans_est[2]:.6f}\n"
        f"  gt  translation xyz [m]: {trans_gt[0]:.6f}, {trans_gt[1]:.6f}, {trans_gt[2]:.6f}\n"
        f"  est baseline norm [m]:   {baseline_est:.6f}\n"
        f"  gt  baseline norm [m]:   {baseline_gt:.6f}\n"
        f"  baseline error [m]:      {baseline_est - baseline_gt:.6f}\n"
        f"  translation error [m]:   {translation_error:.6f}\n"
        f"  translation error [cm]:  {translation_error * 100.0:.2f}\n"
        f"  rotation error [deg]:    {rotation_error:.6f}\n"
        "\n"
        "Output matrices:\n"
        f"  {output_dir / 'T_front_rear_est.csv'}\n"
        f"  {output_dir / 'T_front_rear_gt.csv'}\n"
    )

    (output_dir / "relay_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
