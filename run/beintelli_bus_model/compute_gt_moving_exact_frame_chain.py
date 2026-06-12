#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np


R_OPT_LINK = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)


STATION_META = {
    "F3": {
        "det_csv": "F3_ids_00_05_moving_images.csv",
        "raw_glob": "F3_*_ids_00_05",
        "static_camera": "front_static_camera",
    },
    "F4": {
        "det_csv": "F4_ids_06_11_moving_images.csv",
        "raw_glob": "F4_*_ids_06_11",
        "static_camera": "front_static_camera",
    },
    "R1": {
        "det_csv": "R1_ids_24_29_moving_images.csv",
        "raw_glob": "R1_*_ids_24_29",
        "static_camera": "rear_static_camera",
    },
    "R3": {
        "det_csv": "R3_ids_12_17_moving_images.csv",
        "raw_glob": "R3_*_ids_12_17",
        "static_camera": "rear_static_camera",
    },
}


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


def T_optical_world_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    """
    Gazebo model/link frame:
      x forward, y left, z up

    OpenCV optical frame:
      x right, y down, z forward

    Returns:
      T_optical_world, so X_optical = T_optical_world * X_world
    """
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)


def static_gt_T_front_rear():
    T_front_world = T_optical_world_from_gazebo_pose(
        -3.90, 0.0, 2.85,
        0.0, 0.69813170, 0.0,
    )
    T_rear_world = T_optical_world_from_gazebo_pose(
        5.70, 0.0, 2.85,
        0.0, 0.69813170, math.pi,
    )
    return T_front_world @ inv_T(T_rear_world)


def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def read_route(path):
    return {r["image_name"]: r for r in read_csv(path)}


def parse_value(text, key):
    m = re.search(rf"{re.escape(key)}:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def parse_vec_from_row(row, kind):
    explicit = [
        (f"{kind}_x", f"{kind}_y", f"{kind}_z"),
        (f"{kind}_x_m", f"{kind}_y_m", f"{kind}_z_m"),
    ]
    for names in explicit:
        if all(n in row and row[n] not in ("", None) for n in names):
            return np.array([float(row[n]) for n in names], dtype=float)

    raise KeyError(f"Could not parse {kind} from row keys: {list(row.keys())}")


def T_camera_board_from_row(row):
    rvec = parse_vec_from_row(row, "rvec")
    tvec = parse_vec_from_row(row, "tvec")
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return make_T(R, tvec)


def find_station_obs_csv(raw_station_dir, station):
    meta = STATION_META[station]
    matches = sorted(Path(raw_station_dir).glob(meta["raw_glob"]))
    if not matches:
        raise FileNotFoundError(
            f"No raw station folder for {station} using glob {meta['raw_glob']} in {raw_station_dir}"
        )
    csv_path = matches[0] / "aruco_board_pose_observations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path


def read_static_board_pose(raw_station_dir, station):
    csv_path = find_station_obs_csv(raw_station_dir, station)
    camera = STATION_META[station]["static_camera"]

    rows = read_csv(csv_path)
    for r in rows:
        if r.get("camera") == camera and r.get("status") == "pose_valid":
            return T_camera_board_from_row(r), csv_path, r

    raise RuntimeError(f"No pose_valid {camera} row in {csv_path}")


def read_detection_row_for_image(det_csv, image_name):
    rows = read_csv(det_csv)
    for r in rows:
        if r.get("image_name") == image_name:
            if r.get("status") != "pose_valid":
                raise RuntimeError(
                    f"Row for {image_name} in {det_csv} is not pose_valid: {r.get('status')}"
                )
            return r
    raise RuntimeError(f"No row for image {image_name} in {det_csv}")


def route_pose_to_T(route_row):
    return T_optical_world_from_gazebo_pose(
        float(route_row["x"]),
        float(route_row["y"]),
        float(route_row["z"]),
        float(route_row["roll"]),
        float(route_row["pitch"]),
        float(route_row["yaw"]),
    )


def eval_T(T_est, T_gt):
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    baseline_est = float(np.linalg.norm(t_est))
    baseline_gt = float(np.linalg.norm(t_gt))
    baseline_error_cm = (baseline_est - baseline_gt) * 100.0
    translation_error_cm = float(np.linalg.norm(t_est - t_gt) * 100.0)
    rot_error = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])

    return {
        "baseline_est_m": baseline_est,
        "baseline_gt_m": baseline_gt,
        "baseline_error_cm": baseline_error_cm,
        "translation_error_cm": translation_error_cm,
        "rotation_error_deg": rot_error,
    }


def save_matrix(path, T):
    np.savetxt(path, T, delimiter=",", fmt="%.10f")


def compute_pair(pair, args):
    front_station, rear_station = pair.split("_")

    pair_result_dir = Path(args.chain_results_dir) / f"{pair}_board_scaled_colmap"
    summary_path = pair_result_dir / "summary_no_gt.txt"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = summary_path.read_text()
    front_image = parse_value(summary, "front image")
    rear_image = parse_value(summary, "rear image")

    if not front_image or not rear_image:
        raise RuntimeError(f"Could not parse front/rear images from {summary_path}")

    route = read_route(Path(args.sequence_dir) / "route_commanded.csv")

    if front_image not in route:
        raise RuntimeError(f"Front image missing in route_commanded.csv: {front_image}")
    if rear_image not in route:
        raise RuntimeError(f"Rear image missing in route_commanded.csv: {rear_image}")

    det_dir = Path(args.sequence_dir) / "aruco_no_gt_detections"

    front_det_csv = det_dir / STATION_META[front_station]["det_csv"]
    rear_det_csv = det_dir / STATION_META[rear_station]["det_csv"]

    front_moving_row = read_detection_row_for_image(front_det_csv, front_image)
    rear_moving_row = read_detection_row_for_image(rear_det_csv, rear_image)

    T_front_boardF, front_static_csv, _ = read_static_board_pose(args.raw_station_dir, front_station)
    T_rear_boardR, rear_static_csv, _ = read_static_board_pose(args.raw_station_dir, rear_station)

    T_movingF_boardF = T_camera_board_from_row(front_moving_row)
    T_movingR_boardR = T_camera_board_from_row(rear_moving_row)

    T_front_movingF = T_front_boardF @ inv_T(T_movingF_boardF)
    T_movingR_rear = T_movingR_boardR @ inv_T(T_rear_boardR)

    T_movingF_world_gt = route_pose_to_T(route[front_image])
    T_movingR_world_gt = route_pose_to_T(route[rear_image])

    # Exact GT moving-camera transform at the exact selected frames.
    T_movingF_movingR_gt = T_movingF_world_gt @ inv_T(T_movingR_world_gt)

    T_front_rear = T_front_movingF @ T_movingF_movingR_gt @ T_movingR_rear

    T_gt = static_gt_T_front_rear()
    metrics = eval_T(T_front_rear, T_gt)

    return {
        "pair": pair,
        "front_station": front_station,
        "rear_station": rear_station,
        "front_image": front_image,
        "rear_image": rear_image,
        "front_route": route[front_image],
        "rear_route": route[rear_image],
        "front_static_csv": str(front_static_csv),
        "rear_static_csv": str(rear_static_csv),
        "front_moving_used_ids": front_moving_row.get("used_ids", ""),
        "rear_moving_used_ids": rear_moving_row.get("used_ids", ""),
        "front_moving_rmse_px": front_moving_row.get("reprojection_rmse_px", ""),
        "rear_moving_rmse_px": rear_moving_row.get("reprojection_rmse_px", ""),
        "T_front_rear": T_front_rear,
        "T_gt": T_gt,
        "T_movingF_movingR_gt": T_movingF_movingR_gt,
        **metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--chain_results_dir", required=True)
    ap.add_argument("--raw_station_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pair_dirs = sorted(Path(args.chain_results_dir).glob("*_board_scaled_colmap"))
    pairs = [p.name.replace("_board_scaled_colmap", "") for p in pair_dirs]

    if not pairs:
        raise RuntimeError(f"No pair result folders found in {args.chain_results_dir}")

    all_rows = []

    for pair in pairs:
        result = compute_pair(pair, args)

        pair_dir = out / pair
        pair_dir.mkdir(parents=True, exist_ok=True)

        save_matrix(pair_dir / "T_front_rear_gt_moving_exact_frame.csv", result["T_front_rear"])
        save_matrix(pair_dir / "T_front_rear_static_gt.csv", result["T_gt"])
        save_matrix(pair_dir / "T_movingF_movingR_from_exact_gt_frames.csv", result["T_movingF_movingR_gt"])

        front_route = result["front_route"]
        rear_route = result["rear_route"]

        row = {
            "pair": result["pair"],
            "front_image": result["front_image"],
            "rear_image": result["rear_image"],
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
            "front_moving_used_ids": result["front_moving_used_ids"],
            "rear_moving_used_ids": result["rear_moving_used_ids"],
            "front_moving_rmse_px": result["front_moving_rmse_px"],
            "rear_moving_rmse_px": result["rear_moving_rmse_px"],
            "baseline_est_m": f"{result['baseline_est_m']:.6f}",
            "baseline_gt_m": f"{result['baseline_gt_m']:.6f}",
            "baseline_error_cm": f"{result['baseline_error_cm']:.2f}",
            "translation_error_cm": f"{result['translation_error_cm']:.2f}",
            "rotation_error_deg": f"{result['rotation_error_deg']:.2f}",
        }
        all_rows.append(row)

        text = f"""GT-MOVING EXACT-FRAME CHAIN
===========================

No COLMAP trajectory is used here.
No trajectory alignment is used here.

This run uses the exact moving-camera simulation pose from route_commanded.csv
for the exact front/rear image frames selected by the no-GT chain.

pair:                  {result['pair']}

front image:           {result['front_image']}
front GT pose:
  x y z:               {front_route['x']}, {front_route['y']}, {front_route['z']}
  roll pitch yaw:      {front_route['roll']}, {front_route['pitch']}, {front_route['yaw']}
front moving used ids: {result['front_moving_used_ids']}
front moving rmse px:  {result['front_moving_rmse_px']}

rear image:            {result['rear_image']}
rear GT pose:
  x y z:               {rear_route['x']}, {rear_route['y']}, {rear_route['z']}
  roll pitch yaw:      {rear_route['roll']}, {rear_route['pitch']}, {rear_route['yaw']}
rear moving used ids:  {result['rear_moving_used_ids']}
rear moving rmse px:   {result['rear_moving_rmse_px']}

Metrics against static-camera GT:
  baseline_est_m:       {result['baseline_est_m']:.6f}
  baseline_gt_m:        {result['baseline_gt_m']:.6f}
  baseline_error_cm:    {result['baseline_error_cm']:.2f}
  translation_error_cm: {result['translation_error_cm']:.2f}
  rotation_error_deg:   {result['rotation_error_deg']:.2f}
"""
        (pair_dir / "summary_gt_moving_exact_frame_chain.txt").write_text(text)

    eval_csv = out / "gt_moving_exact_frame_chain_evaluation.csv"
    with eval_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    md = out / "gt_moving_exact_frame_chain_evaluation.md"
    with md.open("w") as f:
        f.write("# GT-moving exact-frame chain evaluation\n\n")
        f.write("No COLMAP trajectory and no trajectory alignment are used here.\n\n")
        f.write("| Pair | Front frame | Rear frame | Baseline error [cm] | Translation error [cm] | Rotation error [deg] |\n")
        f.write("|---|---|---|---:|---:|---:|\n")
        for r in all_rows:
            f.write(
                f"| {r['pair']} | {r['front_image']} | {r['rear_image']} | "
                f"{r['baseline_error_cm']} | {r['translation_error_cm']} | {r['rotation_error_deg']} |\n"
            )

    print("")
    print("[OK] wrote:", eval_csv)
    print("[OK] wrote:", md)
    print("")
    print("Quick view:")
    for r in all_rows:
        print(
            f"{r['pair']:6s} "
            f"baseline_error_cm={r['baseline_error_cm']:>8s} "
            f"translation_error_cm={r['translation_error_cm']:>8s} "
            f"rotation_error_deg={r['rotation_error_deg']:>8s}"
        )


if __name__ == "__main__":
    main()
