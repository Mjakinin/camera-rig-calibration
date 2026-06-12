#!/usr/bin/env python3
import csv
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


STATIC_DET = Path("results/bus_real_data/02_a4_marker_detection_static/detections.csv")
MOVING_DET = Path("results/bus_real_data/03_moving_camera_sequence/moving_detections.csv")
BEST_MOVING = Path("results/bus_real_data/03_moving_camera_sequence/best_marker_frames/best_registered_moving_frame_by_marker.csv")
WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

COLMAP_IMAGES = Path("results/bus_real_data/04_colmap_moving_sequence/sparse_txt_best/images.txt")
ARUCO_SCALE_FILE = Path("results/bus_real_data/04_colmap_moving_sequence/aruco_metric_scale/metric_scale.txt")

MOVING_DEBUG = Path("results/bus_real_data/03_moving_camera_sequence/debug_images")
OUT = Path("results/bus_real_data/06_moving_relay_chain_eval")
OUT.mkdir(parents=True, exist_ok=True)

ROOT_CAM = "cam_edge_3"

# User decision:
# Only cam3 -> cam1 is direct.
# cam3 -> cam0 and cam3 -> cam5 must use moving-camera relay.
TARGETS = {
    "cam3_to_cam0": {
        "target_cam": "cam_edge_0",
        # force true relay: marker 4 is seen by cam0 but not by cam3
        "target_markers": [4],
    },
    "cam3_to_cam5": {
        "target_cam": "cam_edge_5",
        "target_markers": [0, 9, 10, 11, 12, 13],
    },
}

ROOT_MARKERS = [1, 2, 3, 5, 6, 7, 8]

# OpenCV optical frame -> Gazebo camera link frame correction
# From previous direct-static convention fix:
# T_W_optical = T_W_link * inverse(T_optical_to_link)
OPTICAL_TO_LINK_RPY = (0.0, -math.pi / 2.0, math.pi / 2.0)


def get_float(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    raise KeyError(f"none of columns exist: {names}")


def pnp_ok(row):
    val = str(row.get("pnp_success", "True")).strip().lower()
    return val not in ("false", "0", "no", "none", "nan")


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_R(roll, pitch, yaw):
    # SDF/Gazebo convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return rz(yaw) @ ry(pitch) @ rx(roll)


def T_from_R_t(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def T_from_pose_xyz_rpy(x, y, z, roll, pitch, yaw):
    return T_from_R_t(rpy_to_R(roll, pitch, yaw), [x, y, z])


def T_from_detection(row):
    rvec = np.array([
        get_float(row, "rvec_x", "rvec_x_rad"),
        get_float(row, "rvec_y", "rvec_y_rad"),
        get_float(row, "rvec_z", "rvec_z_rad"),
    ], dtype=float)

    tvec = np.array([
        get_float(row, "tvec_x_m", "tvec_x"),
        get_float(row, "tvec_y_m", "tvec_y"),
        get_float(row, "tvec_z_m", "tvec_z"),
    ], dtype=float)

    R, _ = cv2.Rodrigues(rvec)
    return T_from_R_t(R, tvec)


def trans_error(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def rot_error_deg(T_est, T_gt):
    R = T_est[:3, :3].T @ T_gt[:3, :3]
    x = (np.trace(R) - 1.0) / 2.0
    x = max(-1.0, min(1.0, float(x)))
    return math.degrees(math.acos(x))


def load_static_detections():
    data = {}
    with STATIC_DET.open() as fp:
        for r in csv.DictReader(fp):
            if not pnp_ok(r):
                continue
            cam = r["camera"]
            mid = int(r["marker_id"])
            data[(cam, mid)] = r
    return data


def load_moving_detections():
    data = {}
    with MOVING_DET.open() as fp:
        for r in csv.DictReader(fp):
            if not pnp_ok(r):
                continue
            frame = int(r["frame"])
            mid = int(r["marker_id"])
            data[(frame, mid)] = r
    return data


def load_best_registered():
    data = {}
    with BEST_MOVING.open() as fp:
        for r in csv.DictReader(fp):
            mid = int(r["marker_id"])
            data[mid] = r
    return data


def parse_model_pose_from_sdf(model_name):
    root = ET.parse(WORLD_SDF).getroot()
    for model in root.iter():
        if not model.tag.endswith("model"):
            continue
        if model.attrib.get("name") != model_name:
            continue
        pose_el = None
        for child in model:
            if child.tag.endswith("pose"):
                pose_el = child
                break
        if pose_el is None:
            raise RuntimeError(f"no pose found for model {model_name}")
        vals = [float(x) for x in pose_el.text.split()]
        if len(vals) != 6:
            raise RuntimeError(f"bad pose for {model_name}: {pose_el.text}")
        return vals
    raise RuntimeError(f"model not found in SDF: {model_name}")


def T_W_optical_from_sdf_model(model_name):
    x, y, z, r, p, yaw = parse_model_pose_from_sdf(model_name)
    T_W_link = T_from_pose_xyz_rpy(x, y, z, r, p, yaw)
    T_opt_to_link = T_from_pose_xyz_rpy(0, 0, 0, *OPTICAL_TO_LINK_RPY)
    return T_W_link @ invT(T_opt_to_link)


def T_W_optical_from_best_row(row):
    x = get_float(row, "route_x", "x")
    y = get_float(row, "route_y", "y")
    z = get_float(row, "route_z", "z")
    r = get_float(row, "route_roll", "roll")
    p = get_float(row, "route_pitch", "pitch")
    yaw = get_float(row, "route_yaw", "yaw")

    T_W_link = T_from_pose_xyz_rpy(x, y, z, r, p, yaw)
    T_opt_to_link = T_from_pose_xyz_rpy(0, 0, 0, *OPTICAL_TO_LINK_RPY)
    return T_W_link @ invT(T_opt_to_link)


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ], dtype=float)


def load_colmap_poses():
    poses = {}
    if not COLMAP_IMAGES.exists():
        return poses

    for line in COLMAP_IMAGES.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "frame_" not in line or ".png" not in line:
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        name = parts[-1]
        m = re.search(r"frame_(\d+)\.png", name)
        if not m:
            continue

        frame = int(m.group(1))
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]

        R = qvec_to_rotmat(qvec)
        T_cw = T_from_R_t(R, tvec)  # COLMAP world -> camera
        poses[frame] = T_cw

    return poses


def load_colmap_scale():
    """
    No-GT metric scale for COLMAP.

    The scale is estimated from pairs of moving-camera frames that observe the
    same known-size ArUco marker. This is real-life applicable because it uses
    only image detections, COLMAP poses, and the known 0.17 m marker size.
    """
    if not ARUCO_SCALE_FILE.exists():
        raise FileNotFoundError(
            f"Missing no-GT ArUco metric scale file: {ARUCO_SCALE_FILE}\n"
            "Run: python3 run/bus_real_data/16_estimate_colmap_scale_from_aruco_overlap.py"
        )
    return float(ARUCO_SCALE_FILE.read_text().strip().split()[0])


def dist(row):
    try:
        return get_float(row, "distance_m", "distance")
    except Exception:
        return float("nan")


def score(row):
    try:
        return get_float(row, "score")
    except Exception:
        return float("nan")


def evaluate_chain(pair_name, target_cam, root_marker, target_marker,
                   static_det, moving_det, best, colmap_poses, colmap_scale):
    root_frame = int(best[root_marker]["frame"])
    target_frame = int(best[target_marker]["frame"])

    root_static_row = static_det[(ROOT_CAM, root_marker)]
    target_static_row = static_det[(target_cam, target_marker)]
    root_moving_row = moving_det[(root_frame, root_marker)]
    target_moving_row = moving_det[(target_frame, target_marker)]

    T_root_marker = T_from_detection(root_static_row)
    T_movi_marker = T_from_detection(root_moving_row)
    T_target_marker = T_from_detection(target_static_row)
    T_movj_marker = T_from_detection(target_moving_row)

    T_root_movi = T_root_marker @ invT(T_movi_marker)
    T_target_movj = T_target_marker @ invT(T_movj_marker)

    T_W_root = T_W_optical_from_sdf_model(ROOT_CAM)
    T_W_target = T_W_optical_from_sdf_model(target_cam)
    T_root_target_gt = invT(T_W_root) @ T_W_target

    T_W_movi = T_W_optical_from_best_row(best[root_marker])
    T_W_movj = T_W_optical_from_best_row(best[target_marker])
    T_movi_movj_gtmotion = invT(T_W_movi) @ T_W_movj

    out_rows = []

    for method, T_movi_movj in [("GT_motion", T_movi_movj_gtmotion)]:
        T_root_target_est = T_root_movi @ T_movi_movj @ invT(T_target_movj)
        out_rows.append({
            "pair": pair_name,
            "method": method,
            "root_cam": ROOT_CAM,
            "target_cam": target_cam,
            "root_marker": root_marker,
            "target_marker": target_marker,
            "root_frame": root_frame,
            "target_frame": target_frame,
            "translation_error_m": trans_error(T_root_target_est, T_root_target_gt),
            "rotation_error_deg": rot_error_deg(T_root_target_est, T_root_target_gt),
            "est_x": T_root_target_est[0, 3],
            "est_y": T_root_target_est[1, 3],
            "est_z": T_root_target_est[2, 3],
            "gt_x": T_root_target_gt[0, 3],
            "gt_y": T_root_target_gt[1, 3],
            "gt_z": T_root_target_gt[2, 3],
            "root_static_dist_m": dist(root_static_row),
            "target_static_dist_m": dist(target_static_row),
            "root_moving_dist_m": dist(root_moving_row),
            "target_moving_dist_m": dist(target_moving_row),
            "root_moving_score": score(best[root_marker]),
            "target_moving_score": score(best[target_marker]),
        })

    if root_frame in colmap_poses and target_frame in colmap_poses:
        T_cw_i = colmap_poses[root_frame]
        T_cw_j = colmap_poses[target_frame]

        # relative transform camera_j -> camera_i in COLMAP/OpenCV camera coordinates
        T_movi_movj_colmap = T_cw_i @ invT(T_cw_j)
        T_movi_movj_colmap[:3, 3] *= colmap_scale

        T_root_target_est = T_root_movi @ T_movi_movj_colmap @ invT(T_target_movj)
        out_rows.append({
            "pair": pair_name,
            "method": "COLMAP_motion",
            "root_cam": ROOT_CAM,
            "target_cam": target_cam,
            "root_marker": root_marker,
            "target_marker": target_marker,
            "root_frame": root_frame,
            "target_frame": target_frame,
            "translation_error_m": trans_error(T_root_target_est, T_root_target_gt),
            "rotation_error_deg": rot_error_deg(T_root_target_est, T_root_target_gt),
            "est_x": T_root_target_est[0, 3],
            "est_y": T_root_target_est[1, 3],
            "est_z": T_root_target_est[2, 3],
            "gt_x": T_root_target_gt[0, 3],
            "gt_y": T_root_target_gt[1, 3],
            "gt_z": T_root_target_gt[2, 3],
            "root_static_dist_m": dist(root_static_row),
            "target_static_dist_m": dist(target_static_row),
            "root_moving_dist_m": dist(root_moving_row),
            "target_moving_dist_m": dist(target_moving_row),
            "root_moving_score": score(best[root_marker]),
            "target_moving_score": score(best[target_marker]),
        })

    return out_rows


def copy_debug_frames(rows):
    dbg_out = OUT / "debug_selected_frames"
    dbg_out.mkdir(parents=True, exist_ok=True)

    selected = []
    for pair in ["cam3_to_cam0", "cam3_to_cam5"]:
        gt_rows = [
            r for r in rows
            if r["pair"] == pair and r["method"] == "GT_motion"
        ]
        gt_rows.sort(key=lambda r: r["translation_error_m"])
        selected.extend(gt_rows[:3])

    copied = set()
    for r in selected:
        for label, frame in [("root", int(r["root_frame"])), ("target", int(r["target_frame"]))]:
            src = MOVING_DEBUG / f"frame_{frame:04d}_debug.png"
            if not src.exists():
                continue
            dst = dbg_out / (
                f"{r['pair']}_{r['method']}_"
                f"m{int(r['root_marker']):02d}_to_m{int(r['target_marker']):02d}_"
                f"{label}_frame_{frame:04d}_debug.png"
            )
            key = str(dst)
            if key not in copied:
                shutil.copy2(src, dst)
                copied.add(key)


def main():
    static_det = load_static_detections()
    moving_det = load_moving_detections()
    best = load_best_registered()
    colmap_poses = load_colmap_poses()
    colmap_scale = load_colmap_scale()

    rows = []

    for pair_name, cfg in TARGETS.items():
        target_cam = cfg["target_cam"]
        for root_marker in ROOT_MARKERS:
            if (ROOT_CAM, root_marker) not in static_det:
                continue
            if root_marker not in best:
                continue

            for target_marker in cfg["target_markers"]:
                if (target_cam, target_marker) not in static_det:
                    continue
                if target_marker not in best:
                    continue

                root_frame = int(best[root_marker]["frame"])
                target_frame = int(best[target_marker]["frame"])

                if (root_frame, root_marker) not in moving_det:
                    continue
                if (target_frame, target_marker) not in moving_det:
                    continue

                try:
                    rows.extend(evaluate_chain(
                        pair_name, target_cam, root_marker, target_marker,
                        static_det, moving_det, best, colmap_poses, colmap_scale
                    ))
                except Exception as e:
                    print(f"[WARN] skipped {pair_name} root_marker={root_marker}, target_marker={target_marker}: {e}")

    rows.sort(key=lambda r: (r["pair"], r["method"], r["translation_error_m"]))

    out_csv = OUT / "relay_chain_results.csv"
    fields = [
        "pair", "method", "root_cam", "target_cam",
        "root_marker", "target_marker",
        "root_frame", "target_frame",
        "translation_error_m", "rotation_error_deg",
        "est_x", "est_y", "est_z",
        "gt_x", "gt_y", "gt_z",
        "root_static_dist_m", "target_static_dist_m",
        "root_moving_dist_m", "target_moving_dist_m",
        "root_moving_score", "target_moving_score",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    copy_debug_frames(rows)

    report = []
    report.append("Moving-camera relay chain evaluation")
    report.append("====================================")
    report.append("")
    report.append(f"Root camera: {ROOT_CAM}")
    report.append(f"COLMAP poses loaded: {len(colmap_poses)}")
    report.append(f"COLMAP scale used from ArUco metric overlap, no GT: {colmap_scale}")
    report.append("")
    report.append("Interpretation:")
    report.append("- GT_motion uses commanded moving-camera poses.")
    report.append("- COLMAP_motion uses COLMAP relative motion scaled from known-size ArUco marker overlap, without Gazebo GT.")
    report.append("- Both still use ArUco/PnP anchors at the two chain ends.")
    report.append("")

    for pair in ["cam3_to_cam0", "cam3_to_cam5"]:
        for method in ["GT_motion", "COLMAP_motion"]:
            rr = [r for r in rows if r["pair"] == pair and r["method"] == method]
            rr.sort(key=lambda r: r["translation_error_m"])

            report.append(f"{pair} | {method}")
            report.append("-" * (len(pair) + len(method) + 3))

            if not rr:
                report.append("no valid chains")
                report.append("")
                continue

            for r in rr[:10]:
                report.append(
                    f"m{int(r['root_marker']):02d}@f{int(r['root_frame']):04d} "
                    f"-> m{int(r['target_marker']):02d}@f{int(r['target_frame']):04d} | "
                    f"trans={100.0 * float(r['translation_error_m']):7.2f} cm | "
                    f"rot={float(r['rotation_error_deg']):6.2f} deg | "
                    f"root_static_dist={float(r['root_static_dist_m']):.2f} m | "
                    f"target_static_dist={float(r['target_static_dist_m']):.2f} m"
                )
            report.append("")

    report_txt = OUT / "relay_chain_report.txt"
    report_txt.write_text("\n".join(report) + "\n")

    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", report_txt)
    print("[OK] debug frames:", OUT / "debug_selected_frames")
    print()
    print(report_txt.read_text())


if __name__ == "__main__":
    main()
