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


STATIC_CAMERA_POSES = {
    "front_static_camera": (-3.90, 0.0, 2.85, 0.0, 0.69813170, 0.0),
    "rear_static_camera":  ( 5.70, 0.0, 2.85, 0.0, 0.69813170, math.pi),
}


BOARD_GT_POSES = {
    "F3": (-3.25,  0.71, 1.50, 0.0,        0.35,       0.0),
    "F4": (-2.46, -0.75, 1.62, 1.57079633, 1.57079633, 0.0),
    "R1": ( 3.02, -0.75, 1.55, 0.0,        0.30,       math.pi),
    "R3": ( 4.27,  0.93, 1.55, 0.0,        0.30,       1.57079633),
}


STATION_META = {
    "F3": {"det_csv": "F3_ids_00_05_moving_images.csv", "raw_glob": "F3_*_ids_00_05", "static_camera": "front_static_camera"},
    "F4": {"det_csv": "F4_ids_06_11_moving_images.csv", "raw_glob": "F4_*_ids_06_11", "static_camera": "front_static_camera"},
    "R1": {"det_csv": "R1_ids_24_29_moving_images.csv", "raw_glob": "R1_*_ids_24_29", "static_camera": "rear_static_camera"},
    "R3": {"det_csv": "R3_ids_12_17_moving_images.csv", "raw_glob": "R3_*_ids_12_17", "static_camera": "rear_static_camera"},
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


def T_world_link_from_gazebo_pose(x, y, z, roll, pitch, yaw):
    return make_T(rpy_to_R_world_link(roll, pitch, yaw), np.array([x, y, z], dtype=float))


def T_optical_world_from_gazebo_camera_pose(x, y, z, roll, pitch, yaw):
    R_world_link = rpy_to_R_world_link(roll, pitch, yaw)
    R_link_world = R_world_link.T
    R_opt_world = R_OPT_LINK @ R_link_world
    t_opt_world = -R_opt_world @ np.array([x, y, z], dtype=float)
    return make_T(R_opt_world, t_opt_world)


def T_static_world(camera_name):
    return T_optical_world_from_gazebo_camera_pose(*STATIC_CAMERA_POSES[camera_name])


def T_world_board(station):
    return T_world_link_from_gazebo_pose(*BOARD_GT_POSES[station])


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
                image_id = int(parts[0])
                qvec = np.array([float(v) for v in parts[1:5]], dtype=float)
                tvec = np.array([float(v) for v in parts[5:8]], dtype=float)
                name = parts[9]
            except Exception:
                i += 1
                continue
            R_cw = qvec_to_R(qvec)
            T_cw = make_T(R_cw, tvec)
            poses[name] = T_cw
            i += 2
        else:
            i += 1
    return poses


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def read_route(path):
    return {r["image_name"]: r for r in read_csv(path)}


def parse_value(text, key):
    m = re.search(rf"{re.escape(key)}:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def parse_scale(text):
    matches = re.findall(r"\bscale:\s*([0-9eE+\-.]+)", text)
    if not matches:
        raise RuntimeError("Could not parse scale from summary_no_gt.txt")
    return float(matches[0])


def parse_vec_from_row(row, kind):
    candidates = [
        (f"{kind}_x", f"{kind}_y", f"{kind}_z"),
        (f"{kind}_x_m", f"{kind}_y_m", f"{kind}_z_m"),
    ]
    for names in candidates:
        if all(n in row and row[n] not in ("", None) for n in names):
            return np.array([float(row[n]) for n in names], dtype=float)
    raise KeyError(f"Could not parse {kind} from row keys: {list(row.keys())}")


def T_camera_board_from_row(row):
    rvec = parse_vec_from_row(row, "rvec")
    tvec = parse_vec_from_row(row, "tvec")
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return make_T(R, tvec)


def find_station_obs_csv(raw_station_dir, station):
    matches = sorted(Path(raw_station_dir).glob(STATION_META[station]["raw_glob"]))
    if not matches:
        raise FileNotFoundError(f"No raw station folder for {station} in {raw_station_dir}")
    csv_path = matches[0] / "aruco_board_pose_observations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return csv_path


def read_static_pnp_board_pose(raw_station_dir, station):
    csv_path = find_station_obs_csv(raw_station_dir, station)
    camera = STATION_META[station]["static_camera"]
    for row in read_csv(csv_path):
        if row.get("camera") == camera and row.get("status") == "pose_valid":
            return T_camera_board_from_row(row)
    raise RuntimeError(f"No pose_valid {camera} row in {csv_path}")


def read_moving_pnp_board_pose(sequence_dir, station, image_name):
    det_csv = Path(sequence_dir) / "aruco_no_gt_detections" / STATION_META[station]["det_csv"]
    if not det_csv.exists():
        raise FileNotFoundError(det_csv)
    for row in read_csv(det_csv):
        if row.get("image_name") == image_name:
            if row.get("status") != "pose_valid":
                raise RuntimeError(f"{det_csv}: {image_name} is not pose_valid")
            return T_camera_board_from_row(row)
    raise RuntimeError(f"No row for {image_name} in {det_csv}")


def route_row_to_T_moving_world(row):
    return T_optical_world_from_gazebo_camera_pose(
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
        float(row["roll"]),
        float(row["pitch"]),
        float(row["yaw"]),
    )


def colmap_relative_scaled(colmap_poses, front_image, rear_image, scale):
    if front_image not in colmap_poses:
        raise RuntimeError(f"COLMAP image missing: {front_image}")
    if rear_image not in colmap_poses:
        raise RuntimeError(f"COLMAP image missing: {rear_image}")

    T_F_W = colmap_poses[front_image]
    T_R_W = colmap_poses[rear_image]

    T_F_R = T_F_W @ inv_T(T_R_W)
    T_F_R[:3, 3] *= scale
    return T_F_R


def rotation_error_deg(R_est, R_gt):
    R_delta = R_est @ R_gt.T
    val = (np.trace(R_delta) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return math.degrees(math.acos(val))


def eval_T(T_est, T_gt):
    t_est = T_est[:3, 3]
    t_gt = T_gt[:3, 3]

    return {
        "baseline_est_m": float(np.linalg.norm(t_est)),
        "baseline_gt_m": float(np.linalg.norm(t_gt)),
        "baseline_error_cm": float((np.linalg.norm(t_est) - np.linalg.norm(t_gt)) * 100.0),
        "translation_error_cm": float(np.linalg.norm(t_est - t_gt) * 100.0),
        "rotation_error_deg": float(rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])),
    }


def save_matrix(path, T):
    np.savetxt(path, T, delimiter=",", fmt="%.10f")


def make_chain(front_anchor, middle, rear_anchor):
    return front_anchor @ middle @ rear_anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--chain_results_dir", required=True)
    ap.add_argument("--raw_station_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    chain_results_dir = Path(args.chain_results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    route = read_route(sequence_dir / "route_commanded.csv")
    colmap_poses = read_colmap_images(sequence_dir / "sparse_txt" / "images.txt")

    T_front_world_gt = T_static_world("front_static_camera")
    T_rear_world_gt = T_static_world("rear_static_camera")
    T_front_rear_gt = T_front_world_gt @ inv_T(T_rear_world_gt)

    rows = []

    for pair_dir in sorted(chain_results_dir.glob("*_board_scaled_colmap")):
        pair = pair_dir.name.replace("_board_scaled_colmap", "")
        front_station, rear_station = pair.split("_")

        summary_path = pair_dir / "summary_no_gt.txt"
        summary_text = summary_path.read_text()
        front_image = parse_value(summary_text, "front image")
        rear_image = parse_value(summary_text, "rear image")
        scale = parse_scale(summary_text)

        T_movingF_world_gt = route_row_to_T_moving_world(route[front_image])
        T_movingR_world_gt = route_row_to_T_moving_world(route[rear_image])

        T_world_boardF_gt = T_world_board(front_station)
        T_world_boardR_gt = T_world_board(rear_station)

        # GT camera-board observations.
        T_front_boardF_gt = T_front_world_gt @ T_world_boardF_gt
        T_movingF_boardF_gt = T_movingF_world_gt @ T_world_boardF_gt
        T_rear_boardR_gt = T_rear_world_gt @ T_world_boardR_gt
        T_movingR_boardR_gt = T_movingR_world_gt @ T_world_boardR_gt

        # PnP camera-board observations.
        T_front_boardF_pnp = read_static_pnp_board_pose(args.raw_station_dir, front_station)
        T_rear_boardR_pnp = read_static_pnp_board_pose(args.raw_station_dir, rear_station)
        T_movingF_boardF_pnp = read_moving_pnp_board_pose(sequence_dir, front_station, front_image)
        T_movingR_boardR_pnp = read_moving_pnp_board_pose(sequence_dir, rear_station, rear_image)

        # Anchor links.
        gt_front_anchor = T_front_boardF_gt @ inv_T(T_movingF_boardF_gt)
        gt_rear_anchor = T_movingR_boardR_gt @ inv_T(T_rear_boardR_gt)

        pnp_front_anchor = T_front_boardF_pnp @ inv_T(T_movingF_boardF_pnp)
        pnp_rear_anchor = T_movingR_boardR_pnp @ inv_T(T_rear_boardR_pnp)

        # Middle moving-camera transform.
        gt_middle = T_movingF_world_gt @ inv_T(T_movingR_world_gt)
        colmap_middle = colmap_relative_scaled(colmap_poses, front_image, rear_image, scale)

        cases = {
            "all_gt": make_chain(gt_front_anchor, gt_middle, gt_rear_anchor),
            "gt_anchors_gt_moving": make_chain(gt_front_anchor, gt_middle, gt_rear_anchor),
            "gt_anchors_colmap_moving": make_chain(gt_front_anchor, colmap_middle, gt_rear_anchor),
            "pnp_anchors_gt_moving": make_chain(pnp_front_anchor, gt_middle, pnp_rear_anchor),
            "pnp_anchors_colmap_moving": make_chain(pnp_front_anchor, colmap_middle, pnp_rear_anchor),
        }

        for case_name, T_est in cases.items():
            case_dir = out / case_name / pair
            case_dir.mkdir(parents=True, exist_ok=True)

            save_matrix(case_dir / "T_front_rear_est.csv", T_est)
            save_matrix(case_dir / "T_front_rear_gt.csv", T_front_rear_gt)

            metrics = eval_T(T_est, T_front_rear_gt)

            row = {
                "case": case_name,
                "pair": pair,
                "front_image": front_image,
                "rear_image": rear_image,
                "scale_from_no_gt_summary": f"{scale:.10f}",
                "baseline_est_m": f"{metrics['baseline_est_m']:.6f}",
                "baseline_gt_m": f"{metrics['baseline_gt_m']:.6f}",
                "baseline_error_cm": f"{metrics['baseline_error_cm']:.2f}",
                "translation_error_cm": f"{metrics['translation_error_cm']:.2f}",
                "rotation_error_deg": f"{metrics['rotation_error_deg']:.2f}",
            }
            rows.append(row)

            (case_dir / "summary.txt").write_text(
                f"""CHAIN COMPONENT ABLATION
========================

case:        {case_name}
pair:        {pair}
front image: {front_image}
rear image:  {rear_image}

scale used for COLMAP middle:
  {scale:.10f}

Metrics against static-camera GT:
  baseline_est_m:       {metrics['baseline_est_m']:.6f}
  baseline_gt_m:        {metrics['baseline_gt_m']:.6f}
  baseline_error_cm:    {metrics['baseline_error_cm']:.2f}
  translation_error_cm: {metrics['translation_error_cm']:.2f}
  rotation_error_deg:   {metrics['rotation_error_deg']:.2f}
"""
            )

    csv_path = out / "component_ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out / "component_ablation_summary.md"
    with md_path.open("w") as f:
        f.write("# Chain Component Ablation Summary\n\n")
        f.write("| Case | Pair | Baseline error [cm] | Translation error [cm] | Rotation error [deg] |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['case']} | {r['pair']} | {r['baseline_error_cm']} | "
                f"{r['translation_error_cm']} | {r['rotation_error_deg']} |\n"
            )

    (out / "README.md").write_text("""# 03 Chain Component Ablation

This folder compares different levels of ground-truth and estimated components.

Cases:

## all_gt / gt_anchors_gt_moving
Uses:
- GT static camera poses
- GT board poses
- GT moving-camera pose at the exact selected frames

Expected:
- zero error
- validates transform algebra and frame conventions

## gt_anchors_colmap_moving
Uses:
- GT static camera poses
- GT board poses
- COLMAP moving-camera relative motion

Purpose:
- isolates the moving-camera trajectory error from COLMAP

## pnp_anchors_gt_moving
Uses:
- PnP-estimated board-camera anchor links
- GT moving-camera relative motion

Purpose:
- isolates ArUco/PnP anchor errors

## pnp_anchors_colmap_moving
Uses:
- PnP-estimated board-camera anchor links
- COLMAP moving-camera relative motion

Purpose:
- closest controlled reproduction of the no-GT chain

Important:
GT moving should be best when all other components are held constant.
If mixed cases appear better or worse, this can be caused by error compensation between PnP anchor errors and COLMAP motion errors.
""")

    print("[OK] wrote", csv_path)
    print("[OK] wrote", md_path)
    print("")
    print("Quick grouped view:")
    for r in rows:
        print(
            f"{r['case']:28s} {r['pair']:6s} "
            f"trans_cm={r['translation_error_cm']:>8s} "
            f"rot_deg={r['rotation_error_deg']:>6s}"
        )


if __name__ == "__main__":
    main()
