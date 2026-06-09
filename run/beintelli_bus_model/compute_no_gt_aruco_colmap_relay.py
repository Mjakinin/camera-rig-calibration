#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def T_from_R_t(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def read_colmap_images(images_txt: Path):
    images = {}

    with images_txt.open() as f:
        lines = list(f)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]

            R_cw = qvec_to_rotmat(qvec)
            T_cw = T_from_R_t(R_cw, tvec)
            C_w = -R_cw.T @ tvec

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "R_cw": R_cw,
                "T_cw": T_cw,
                "C_w": C_w,
            }

            i += 2
        else:
            i += 1

    return images


def read_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def valid_detection_rows(path: Path, min_markers: int, max_rmse: float):
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

        if n < min_markers:
            continue
        if rmse > max_rmse:
            continue

        out.append(r)

    return out


def best_detection_row(rows, colmap_images):
    usable = [r for r in rows if r["image_name"] in colmap_images]
    if not usable:
        raise RuntimeError("No valid detection row is also registered in COLMAP images.txt")

    return sorted(
        usable,
        key=lambda r: (-int(r["num_used_markers"]), float(r["reprojection_rmse_px"]))
    )[0]


def rvec_tvec_from_row(row):
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=np.float64)
    tvec = np.array([float(row["tvec_x_m"]), float(row["tvec_y_m"]), float(row["tvec_z_m"])], dtype=np.float64)
    return rvec, tvec


def T_camera_board_from_row(row):
    rvec, tvec = rvec_tvec_from_row(row)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return T_from_R_t(R, tvec)


def camera_center_board_from_row(row):
    return np.array([
        float(row["camera_center_board_x_m"]),
        float(row["camera_center_board_y_m"]),
        float(row["camera_center_board_z_m"]),
    ], dtype=np.float64)


def read_static_observation_pose(path: Path, camera_name: str):
    rows = read_csv(path)
    matches = [r for r in rows if r.get("camera") == camera_name and r.get("status") == "pose_valid"]
    if not matches:
        raise RuntimeError(f"No pose_valid row for camera={camera_name} in {path}")
    return T_camera_board_from_row(matches[0]), matches[0]


def umeyama_sim3(src, dst):
    """
    Estimate dst ~= s * R * src + t.
    src: Nx3 COLMAP camera centers
    dst: Nx3 metric ArUco board-frame camera centers
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 arrays")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 center pairs for Sim(3)")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    var_src = np.mean(np.sum(X * X, axis=1))
    if var_src < 1e-12:
        raise ValueError("Degenerate src centers: variance too small")

    cov = (Y.T @ X) / src.shape[0]
    U, Svals, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1

    R = U @ D @ Vt
    scale = np.trace(np.diag(Svals) @ D) / var_src
    t = mu_dst - scale * R @ mu_src

    pred = scale * (R @ src.T).T + t
    err = pred - dst
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    return scale, R, t, rmse


def align_colmap_T_cw_to_metric(T_cw_colmap, scale, R_sim, t_sim):
    """
    X_metric = scale * R_sim * X_colmap + t_sim
    COLMAP:  X_cam_colmap = R_cw * X_colmap + t_cw

    Metric camera coordinates use metric units:
    X_cam_metric = scale * X_cam_colmap

    Therefore:
    X_cam_metric = R_cw * R_sim.T * X_metric + scale*t_cw - R_cw*R_sim.T*t_sim
    """
    R_cw = T_cw_colmap[:3, :3]
    t_cw = T_cw_colmap[:3, 3]

    R_cm = R_cw @ R_sim.T
    t_cm = scale * t_cw - R_cm @ t_sim

    return T_from_R_t(R_cm, t_cm)


def rotation_angle_deg(R):
    val = (np.trace(R) - 1.0) / 2.0
    val = max(-1.0, min(1.0, float(val)))
    return math.degrees(math.acos(val))


def write_matrix_csv(path: Path, T):
    with path.open("w") as f:
        for row in T:
            f.write(",".join(f"{x:.10f}" for x in row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--scale_board_detections_csv", required=True)
    parser.add_argument("--front_board_detections_csv", required=True)
    parser.add_argument("--rear_board_detections_csv", required=True)

    parser.add_argument("--front_static_obs_csv", required=True)
    parser.add_argument("--rear_static_obs_csv", required=True)

    parser.add_argument("--front_static_camera", default="front_static_camera")
    parser.add_argument("--rear_static_camera", default="rear_static_camera")

    parser.add_argument("--min_scale_markers", type=int, default=2)
    parser.add_argument("--min_anchor_markers", type=int, default=2)
    parser.add_argument("--max_rmse_px", type=float, default=2.0)

    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    sparse_txt = dataset / "sparse_txt"
    images_txt = sparse_txt / "images.txt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colmap = read_colmap_images(images_txt)

    scale_rows = valid_detection_rows(Path(args.scale_board_detections_csv), args.min_scale_markers, args.max_rmse_px)
    scale_rows = [r for r in scale_rows if r["image_name"] in colmap]

    if len(scale_rows) < 3:
        raise RuntimeError(f"Need at least 3 valid scale rows registered in COLMAP. Got {len(scale_rows)}")

    src = []
    dst = []

    for r in scale_rows:
        name = r["image_name"]
        src.append(colmap[name]["C_w"])
        dst.append(camera_center_board_from_row(r))

    src = np.asarray(src)
    dst = np.asarray(dst)

    scale, R_sim, t_sim, sim3_rmse = umeyama_sim3(src, dst)

    front_rows = valid_detection_rows(Path(args.front_board_detections_csv), args.min_anchor_markers, args.max_rmse_px)
    rear_rows = valid_detection_rows(Path(args.rear_board_detections_csv), args.min_anchor_markers, args.max_rmse_px)

    front_row = best_detection_row(front_rows, colmap)
    rear_row = best_detection_row(rear_rows, colmap)

    front_image = front_row["image_name"]
    rear_image = rear_row["image_name"]

    T_frontStatic_boardF, front_static_row = read_static_observation_pose(
        Path(args.front_static_obs_csv),
        args.front_static_camera,
    )
    T_rearStatic_boardR, rear_static_row = read_static_observation_pose(
        Path(args.rear_static_obs_csv),
        args.rear_static_camera,
    )

    T_movingF_boardF = T_camera_board_from_row(front_row)
    T_movingR_boardR = T_camera_board_from_row(rear_row)

    T_frontStatic_movingF = T_frontStatic_boardF @ invT(T_movingF_boardF)
    T_movingR_rearStatic = T_movingR_boardR @ invT(T_rearStatic_boardR)

    T_movingF_metric = align_colmap_T_cw_to_metric(colmap[front_image]["T_cw"], scale, R_sim, t_sim)
    T_movingR_metric = align_colmap_T_cw_to_metric(colmap[rear_image]["T_cw"], scale, R_sim, t_sim)

    T_movingF_movingR = T_movingF_metric @ invT(T_movingR_metric)

    T_front_rear = T_frontStatic_movingF @ T_movingF_movingR @ T_movingR_rearStatic

    write_matrix_csv(output_dir / "T_front_rear_no_gt.csv", T_front_rear)
    write_matrix_csv(output_dir / "T_movingF_movingR_colmap_board_scaled.csv", T_movingF_movingR)

    sim3_mat = np.eye(4)
    sim3_mat[:3, :3] = scale * R_sim
    sim3_mat[:3, 3] = t_sim
    write_matrix_csv(output_dir / "Sim3_colmap_to_F3_board_metric.csv", sim3_mat)

    baseline = float(np.linalg.norm(T_front_rear[:3, 3]))
    rot_deg = rotation_angle_deg(T_front_rear[:3, :3])

    summary = []
    summary.append("NO-GT ARUCO + COLMAP RELAY RESULT")
    summary.append("=================================")
    summary.append("")
    summary.append("Inputs:")
    summary.append(f"dataset_dir:                  {dataset}")
    summary.append(f"scale_board_detections_csv:   {args.scale_board_detections_csv}")
    summary.append(f"front_board_detections_csv:   {args.front_board_detections_csv}")
    summary.append(f"rear_board_detections_csv:    {args.rear_board_detections_csv}")
    summary.append(f"front_static_obs_csv:         {args.front_static_obs_csv}")
    summary.append(f"rear_static_obs_csv:          {args.rear_static_obs_csv}")
    summary.append("")
    summary.append("Selected anchor frames:")
    summary.append(f"front image:                  {front_image}")
    summary.append(f"front used ids:               {front_row['used_ids']}")
    summary.append(f"front rmse px:                {front_row['reprojection_rmse_px']}")
    summary.append(f"rear image:                   {rear_image}")
    summary.append(f"rear used ids:                {rear_row['used_ids']}")
    summary.append(f"rear rmse px:                 {rear_row['reprojection_rmse_px']}")
    summary.append("")
    summary.append("Board-based Sim(3), no route_gt.csv:")
    summary.append(f"scale pairs:                  {len(scale_rows)}")
    summary.append(f"scale:                        {scale:.10f}")
    summary.append(f"sim3 center rmse m:           {sim3_rmse:.6f}")
    summary.append("")
    summary.append("Estimated T_front_rear:")
    summary.append(f"baseline norm m:              {baseline:.6f}")
    summary.append(f"rotation angle deg:           {rot_deg:.6f}")
    summary.append("")
    summary.append("T_front_rear_no_gt:")
    for row in T_front_rear:
        summary.append("  " + " ".join(f"{x: .10f}" for x in row))
    summary.append("")
    summary.append("Outputs:")
    summary.append(f"T_front_rear_no_gt.csv:       {output_dir / 'T_front_rear_no_gt.csv'}")
    summary.append(f"T_movingF_movingR csv:        {output_dir / 'T_movingF_movingR_colmap_board_scaled.csv'}")
    summary.append(f"Sim3 csv:                     {output_dir / 'Sim3_colmap_to_F3_board_metric.csv'}")
    summary.append("")
    summary.append("IMPORTANT:")
    summary.append("This computation does not use route_gt.csv.")
    summary.append("Ground truth may be used later only for evaluation, not for estimating the transform.")

    text = "\n".join(summary) + "\n"
    (output_dir / "summary_no_gt.txt").write_text(text)

    print(text)


if __name__ == "__main__":
    main()
