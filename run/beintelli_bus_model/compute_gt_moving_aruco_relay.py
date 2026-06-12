#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

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
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out

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

def optical_T_world_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)

def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))

def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def T_camera_board_from_row(row):
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=float)
    tvec = np.array([float(row["tvec_x_m"]), float(row["tvec_y_m"]), float(row["tvec_z_m"])], dtype=float)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return make_T(R, tvec)

def read_static_observation_pose(path, camera_name):
    rows = read_csv(path)
    matches = [r for r in rows if r.get("camera") == camera_name and r.get("status") == "pose_valid"]
    if not matches:
        raise RuntimeError(f"No pose_valid row for {camera_name} in {path}")
    return T_camera_board_from_row(matches[0])

def valid_detection_rows(path, min_markers, max_rmse):
    rows = read_csv(path)
    out = []
    for r in rows:
        if r.get("status") != "pose_valid":
            continue
        try:
            n = int(r["num_used_markers"])
            rmse = float(r["reprojection_rmse_px"])
        except Exception:
            continue
        if n >= min_markers and rmse <= max_rmse:
            out.append(r)
    return out

def best_detection_row(rows, route):
    usable = [r for r in rows if r["image_name"] in route]
    if not usable:
        raise RuntimeError("No valid detection row also exists in route_commanded.csv")
    return sorted(usable, key=lambda r: (-int(r["num_used_markers"]), float(r["reprojection_rmse_px"])))[0]

def read_route_commanded(path):
    route = {}
    for r in read_csv(path):
        route[r["image_name"]] = r
    return route

def T_moving_world_from_route_row(row):
    return optical_T_world_from_gazebo_pose(
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
        float(row["roll"]),
        float(row["pitch"]),
        float(row["yaw"]),
    )

def write_matrix(path, T):
    np.savetxt(path, T, delimiter=",", fmt="%.10f")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--front_board_detections_csv", required=True)
    ap.add_argument("--rear_board_detections_csv", required=True)
    ap.add_argument("--front_static_obs_csv", required=True)
    ap.add_argument("--rear_static_obs_csv", required=True)
    ap.add_argument("--min_anchor_markers", type=int, default=2)
    ap.add_argument("--max_rmse_px", type=float, default=2.0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    dataset = Path(args.dataset_dir)
    route_path = dataset / "route_commanded.csv"
    route = read_route_commanded(route_path)

    front_rows = valid_detection_rows(args.front_board_detections_csv, args.min_anchor_markers, args.max_rmse_px)
    rear_rows = valid_detection_rows(args.rear_board_detections_csv, args.min_anchor_markers, args.max_rmse_px)

    front_row = best_detection_row(front_rows, route)
    rear_row = best_detection_row(rear_rows, route)

    front_image = front_row["image_name"]
    rear_image = rear_row["image_name"]

    T_front_boardF = read_static_observation_pose(args.front_static_obs_csv, "front_static_camera")
    T_rear_boardR = read_static_observation_pose(args.rear_static_obs_csv, "rear_static_camera")

    T_movingF_boardF = T_camera_board_from_row(front_row)
    T_movingR_boardR = T_camera_board_from_row(rear_row)

    T_front_movingF = T_front_boardF @ inv_T(T_movingF_boardF)
    T_movingR_rear = T_movingR_boardR @ inv_T(T_rear_boardR)

    T_movingF_world_gt = T_moving_world_from_route_row(route[front_image])
    T_movingR_world_gt = T_moving_world_from_route_row(route[rear_image])

    T_movingF_movingR_gt = T_movingF_world_gt @ inv_T(T_movingR_world_gt)
    T_front_rear = T_front_movingF @ T_movingF_movingR_gt @ T_movingR_rear

    T_front_world_gt = optical_T_world_from_gazebo_pose(-3.90, 0.0, 2.85, 0.0, 0.69813170, 0.0)
    T_rear_world_gt = optical_T_world_from_gazebo_pose(5.70, 0.0, 2.85, 0.0, 0.69813170, math.pi)
    T_gt = T_front_world_gt @ inv_T(T_rear_world_gt)

    t_est = T_front_rear[:3, 3]
    t_gt = T_gt[:3, 3]

    baseline_est = float(np.linalg.norm(t_est))
    baseline_gt = float(np.linalg.norm(t_gt))
    baseline_error_cm = (baseline_est - baseline_gt) * 100.0
    translation_error_cm = float(np.linalg.norm(t_est - t_gt) * 100.0)
    rot_error = rotation_error_deg(T_front_rear[:3, :3], T_gt[:3, :3])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_matrix(out / "T_front_rear_gt_moving_chain.csv", T_front_rear)
    write_matrix(out / "T_front_rear_static_gt.csv", T_gt)
    write_matrix(out / "T_movingF_movingR_from_gt_route.csv", T_movingF_movingR_gt)

    text = f"""GT-MOVING ARUCO RELAY RESULT
============================

This is diagnostic / upper-bound.
It uses route_commanded.csv for the moving-camera front-to-rear transform.
It does NOT evaluate COLMAP trajectory quality.

dataset_dir:             {dataset}
route_commanded_csv:     {route_path}

front image:             {front_image}
front used ids:          {front_row['used_ids']}
front rmse px:           {front_row['reprojection_rmse_px']}

rear image:              {rear_image}
rear used ids:           {rear_row['used_ids']}
rear rmse px:            {rear_row['reprojection_rmse_px']}

Metrics against static-camera GT:
  baseline_est_m:       {baseline_est:.6f}
  baseline_gt_m:        {baseline_gt:.6f}
  baseline_error_cm:    {baseline_error_cm:.2f}
  translation_error_cm: {translation_error_cm:.2f}
  rotation_error_deg:   {rot_error:.2f}

T_front_rear_gt_moving_chain:
{T_front_rear}
"""
    (out / "summary_gt_moving_chain.txt").write_text(text)
    print(text)

if __name__ == "__main__":
    main()
