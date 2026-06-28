#!/usr/bin/env python3

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT_CAM = "cam_edge_3"

STATIC_DET_CSV = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/.ap01_compat_cache/static_observations/detections.csv")
MOVING_DET_CSV = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/.ap01_compat_cache/moving_observations/moving_detections.csv")
COLMAP_DIR = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/04_moving_camera_colmap_trajectory")
WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
ROUTE_JSON = Path("src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json")

OUT = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/06_moving_relay_chain_eval")
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = {
    "cam3_to_cam0": "cam_edge_0",
    "cam3_to_cam5": "cam_edge_5",
}

R_OPT_TO_LINK = None


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def get_R_opt_to_link():
    global R_OPT_TO_LINK
    if R_OPT_TO_LINK is None:
        R_OPT_TO_LINK = rpy_to_R(0.0, -math.pi / 2.0, math.pi / 2.0)
    return R_OPT_TO_LINK


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rvec_to_R(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def R_to_rpy_rad(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def R_to_rpy_deg(R):
    r, p, y = R_to_rpy_rad(R)
    return [math.degrees(r), math.degrees(p), math.degrees(y)]


def qvec_to_R(qvec):
    qw, qx, qy, qz = [float(x) for x in qvec]
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(1e-15, float(np.linalg.norm(q)))
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def R_to_quat_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(1e-15, float(np.linalg.norm(q)))
    return q


def quat_wxyz_to_R(q):
    q = np.asarray(q, dtype=np.float64)
    q /= max(1e-15, float(np.linalg.norm(q)))
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def weighted_R_mean(rotations, weights):
    qs = [R_to_quat_wxyz(R) for R in rotations]
    ref = qs[0]
    A = np.zeros((4, 4), dtype=np.float64)

    for q, w in zip(qs, weights):
        if float(np.dot(q, ref)) < 0.0:
            q = -q
        A += float(w) * np.outer(q, q)

    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, int(np.argmax(vals))]
    if q[0] < 0:
        q = -q
    return quat_wxyz_to_R(q)


def weighted_T_mean(Ts, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(1e-15, float(np.sum(weights)))

    t = np.zeros(3, dtype=np.float64)
    for T, w in zip(Ts, weights):
        t += float(w) * T[:3, 3]

    R = weighted_R_mean([T[:3, :3] for T in Ts], weights)
    return make_T(R, t)


def trans_error(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    return float(np.linalg.norm(dT[:3, 3]))


def rot_error_deg(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    arg = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    return math.degrees(math.acos(clamp(float(arg))))


def transform_error(T_est, T_ref):
    return trans_error(T_est, T_ref), rot_error_deg(T_est, T_ref)


def safe_float(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def marker_area_px(row):
    keys = [
        ("corner0_u", "corner0_v"),
        ("corner1_u", "corner1_v"),
        ("corner2_u", "corner2_v"),
        ("corner3_u", "corner3_v"),
    ]

    if all(u in row and v in row for u, v in keys):
        pts = [[safe_float(row, u), safe_float(row, v)] for u, v in keys]
        if all(math.isfinite(x) and math.isfinite(y) for x, y in pts):
            return polygon_area(pts)

    return float("nan")


def center_norm(row, width=1280.0, height=720.0):
    cu = safe_float(row, "center_u")
    cv = safe_float(row, "center_v")
    if not math.isfinite(cu) or not math.isfinite(cv):
        return 1.0

    cx, cy = width / 2.0, height / 2.0
    half_diag = math.sqrt(cx * cx + cy * cy)
    return math.sqrt((cu - cx) ** 2 + (cv - cy) ** 2) / half_diag


def detection_quality(row):
    dist = safe_float(row, "distance_m", 99.0)
    area = marker_area_px(row)
    cn = center_norm(row)

    if not math.isfinite(dist) or dist <= 0:
        dist = 99.0
    if not math.isfinite(area) or area <= 0:
        area = 1.0
    if not math.isfinite(cn):
        cn = 1.0

    return area / ((dist * dist) * (1.0 + cn))


def frame_idx_from_row(row):
    for k in ["frame_idx", "frame", "image_idx", "idx"]:
        if k in row and str(row[k]).strip() != "":
            try:
                return int(float(row[k]))
            except Exception:
                pass

    for k in ["image_name", "filename", "path"]:
        if k in row:
            m = re.search(r"(\d+)", str(row[k]))
            if m:
                return int(m.group(1))

    raise RuntimeError(f"Cannot infer frame index from row keys: {list(row.keys())}")


def T_from_detection(row):
    rvec = np.array([
        float(row["rvec_x"]),
        float(row["rvec_y"]),
        float(row["rvec_z"]),
    ], dtype=np.float64)

    tvec = np.array([
        float(row["tvec_x_m"]),
        float(row["tvec_y_m"]),
        float(row["tvec_z_m"]),
    ], dtype=np.float64)

    return make_T(rvec_to_R(rvec), tvec)


def load_static_detections():
    if not STATIC_DET_CSV.exists():
        raise RuntimeError(f"Missing static detections: {STATIC_DET_CSV}")

    out = {}
    with STATIC_DET_CSV.open() as fp:
        for row in csv.DictReader(fp):
            if str(row.get("pnp_success", "True")).lower() in ["false", "0", "no"]:
                continue
            row = dict(row)
            row["marker_id"] = int(row["marker_id"])
            row["quality"] = detection_quality(row)
            key = (row["camera"], row["marker_id"])
            if key not in out or row["quality"] > out[key]["quality"]:
                out[key] = row
    return out


def load_moving_detections():
    if not MOVING_DET_CSV.exists():
        raise RuntimeError(f"Missing moving detections: {MOVING_DET_CSV}")

    out = {}
    by_marker = {}

    with MOVING_DET_CSV.open() as fp:
        for row in csv.DictReader(fp):
            if str(row.get("pnp_success", "True")).lower() in ["false", "0", "no"]:
                continue

            row = dict(row)
            frame = frame_idx_from_row(row)
            marker = int(row["marker_id"])
            row["frame_idx"] = frame
            row["marker_id"] = marker
            row["quality"] = detection_quality(row)

            key = (frame, marker)
            if key not in out or row["quality"] > out[key]["quality"]:
                out[key] = row

    for (frame, marker), row in out.items():
        by_marker.setdefault(marker, []).append(row)

    for marker in by_marker:
        by_marker[marker].sort(key=lambda r: int(r["frame_idx"]))

    return out, by_marker


def find_colmap_images_file():
    candidates = [
        COLMAP_DIR / "sparse/0/images.txt",
        COLMAP_DIR / "sparse_text/images.txt",
        COLMAP_DIR / "images.txt",
    ]
    for p in candidates:
        if p.exists():
            return p

    found = sorted(COLMAP_DIR.rglob("images.txt"))
    if found:
        return found[0]

    raise RuntimeError(f"Could not find COLMAP images.txt under {COLMAP_DIR}")


def frame_idx_from_image_name(name):
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        raise RuntimeError(f"Cannot infer frame index from COLMAP image name: {name}")
    return int(m.group(1))


def load_colmap_poses():
    p = find_colmap_images_file()
    poses = {}

    with p.open() as fp:
        lines = [line.strip() for line in fp if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            qvec = [float(x) for x in parts[1:5]]
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            name = parts[9]
            frame = frame_idx_from_image_name(name)
            poses[frame] = make_T(qvec_to_R(qvec), tvec)
            i += 2
        else:
            i += 1

    if not poses:
        raise RuntimeError(f"No COLMAP poses parsed from {p}")

    return poses


def load_colmap_scale():
    seq_root = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/04_moving_camera_colmap_trajectory")

    # Canonical AP01 scale source written by 12_estimate_colmap_scale_from_aruco.py.
    json_path = seq_root / "metric_colmap_scale.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        for key in ["metric_colmap_scale", "colmap_metric_scale", "metric_scale", "scale"]:
            if key in data:
                return float(data[key])

    # Fallback: direct text scale file.
    txt_path = seq_root / "aruco_metric_scale" / "metric_scale.txt"
    if txt_path.exists():
        return float(txt_path.read_text().strip())

    raise RuntimeError(
        "No AP01 COLMAP metric scale found. Run: "
        "python3 run/bus_real_data/12_estimate_colmap_scale_from_aruco.py"
    )

def parse_sdf_model_poses():
    if not WORLD_SDF.exists():
        raise RuntimeError(f"World SDF missing: {WORLD_SDF}")

    tree = ET.parse(WORLD_SDF)
    root = tree.getroot()
    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "")
        pose_el = model.find("pose")
        if pose_el is None or not pose_el.text:
            continue

        vals = [float(x) for x in pose_el.text.split()]
        if len(vals) < 6:
            continue

        x, y, z, roll, pitch, yaw = vals[:6]
        T = make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64))
        poses[name] = T

        if "cam_edge_" in name:
            short = name[name.index("cam_edge_"):]
            poses[short] = T

    return poses


def sdf_model_pose_to_optical(T_W_model):
    T_opt_to_link = make_T(get_R_opt_to_link(), np.zeros(3))
    return T_W_model @ invT(T_opt_to_link)


def T_W_optical_from_sdf_model(camera_name):
    poses = parse_sdf_model_poses()
    if camera_name not in poses:
        raise RuntimeError(f"Cannot find camera model pose in SDF: {camera_name}. Found keys: {sorted(poses)[:20]}")
    return sdf_model_pose_to_optical(poses[camera_name])


def extract_pose_from_dict(d):
    frame = None
    for k in ["frame_idx", "frame", "idx", "image_idx"]:
        if k in d:
            try:
                frame = int(float(d[k]))
                break
            except Exception:
                pass

    pose = None
    if "pose" in d and isinstance(d["pose"], list) and len(d["pose"]) >= 6:
        pose = [float(x) for x in d["pose"][:6]]
    elif all(k in d for k in ["x", "y", "z", "roll", "pitch", "yaw"]):
        pose = [float(d[k]) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    elif all(k in d for k in ["pose_x", "pose_y", "pose_z", "pose_roll", "pose_pitch", "pose_yaw"]):
        pose = [float(d[k]) for k in ["pose_x", "pose_y", "pose_z", "pose_roll", "pose_pitch", "pose_yaw"]]

    return frame, pose


def load_moving_gt_poses():
    if not ROUTE_JSON.exists():
        return {}

    data = json.loads(ROUTE_JSON.read_text())
    records = []

    def rec(obj):
        if isinstance(obj, dict):
            frame, pose = extract_pose_from_dict(obj)
            if pose is not None:
                records.append((frame, pose))
            for v in obj.values():
                rec(v)
        elif isinstance(obj, list):
            for v in obj:
                rec(v)

    rec(data)

    out = {}
    auto_idx = 0
    for frame, pose in records:
        if frame is None:
            frame = auto_idx
            auto_idx += 1
        x, y, z, roll, pitch, yaw = pose
        T_W_model = make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64))
        out[int(frame)] = sdf_model_pose_to_optical(T_W_model)

    return out


def load_best_registered():
    # Kept for backward compatibility with older export scripts.
    p = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/06_moving_relay_chain_eval/best_marker_frames/best_registered_moving_frame_by_marker.csv")
    if not p.exists():
        return {}
    out = {}
    with p.open() as fp:
        for row in csv.DictReader(fp):
            marker = int(row["marker_id"])
            out[marker] = row
    return out


def T_W_optical_from_best_row(row):
    for prefix in ["", "gt_", "pose_"]:
        keys = [prefix + k for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        if all(k in row for k in keys):
            vals = [float(row[k]) for k in keys]
            T_W_model = make_T(rpy_to_R(vals[3], vals[4], vals[5]), np.array(vals[:3], dtype=np.float64))
            return sdf_model_pose_to_optical(T_W_model)
    raise RuntimeError(f"Cannot parse GT pose from best row keys: {list(row.keys())}")


def pose_fields(prefix, T):
    rpy = R_to_rpy_deg(T[:3, :3])
    return {
        f"{prefix}_x": float(T[0, 3]),
        f"{prefix}_y": float(T[1, 3]),
        f"{prefix}_z": float(T[2, 3]),
        f"{prefix}_roll_deg": float(rpy[0]),
        f"{prefix}_pitch_deg": float(rpy[1]),
        f"{prefix}_yaw_deg": float(rpy[2]),
    }


def T_from_pose_fields(row, prefix="estimated_target_in_root"):
    x = float(row[f"{prefix}_x"])
    y = float(row[f"{prefix}_y"])
    z = float(row[f"{prefix}_z"])
    roll = math.radians(float(row[f"{prefix}_roll_deg"]))
    pitch = math.radians(float(row[f"{prefix}_pitch_deg"]))
    yaw = math.radians(float(row[f"{prefix}_yaw_deg"]))
    return make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64))


def robust_aggregate(candidates):
    if not candidates:
        raise RuntimeError("No candidates for aggregation")

    Ts = [c["T_est"] for c in candidates]
    weights = np.array([max(1e-12, c["combined_quality"]) for c in candidates], dtype=np.float64)

    # Robust initial center: component-wise translation median + quality-weighted quaternion mean.
    translations = np.array([T[:3, 3] for T in Ts], dtype=np.float64)
    t0 = np.median(translations, axis=0)

    R0 = weighted_R_mean([T[:3, :3] for T in Ts], weights)
    T0 = make_T(R0, t0)

    tdev = []
    rdev = []
    for c in candidates:
        te, re = transform_error(c["T_est"], T0)
        c["deviation_to_consensus_m"] = te
        c["deviation_to_consensus_cm"] = 100.0 * te
        c["deviation_to_consensus_rot_deg"] = re
        tdev.append(te)
        rdev.append(re)

    tdev = np.asarray(tdev, dtype=np.float64)
    rdev = np.asarray(rdev, dtype=np.float64)

    t_med = float(np.median(tdev))
    r_med = float(np.median(rdev))
    t_mad = 1.4826 * float(np.median(np.abs(tdev - t_med)))
    r_mad = 1.4826 * float(np.median(np.abs(rdev - r_med)))

    t_thresh = max(0.30, t_med + 3.0 * t_mad)
    r_thresh = max(7.0, r_med + 3.0 * r_mad)

    inliers = []
    outliers = []
    for c in candidates:
        ok = c["deviation_to_consensus_m"] <= t_thresh and c["deviation_to_consensus_rot_deg"] <= r_thresh
        c["inlier"] = bool(ok)
        if ok:
            inliers.append(c)
        else:
            outliers.append(c)

    if len(inliers) < 3:
        print(f"[WARN] Only {len(inliers)} inliers. Falling back to top 50% by quality.")
        sorted_by_q = sorted(candidates, key=lambda c: c["combined_quality"], reverse=True)
        keep_n = max(3, len(sorted_by_q) // 2)
        inliers = sorted_by_q[:keep_n]
        outliers = sorted_by_q[keep_n:]
        for c in inliers:
            c["inlier"] = True
        for c in outliers:
            c["inlier"] = False

    inlier_weights = np.array([max(1e-12, c["combined_quality"]) for c in inliers], dtype=np.float64)
    T_weighted = weighted_T_mean([c["T_est"] for c in inliers], inlier_weights)
    T_unweighted = weighted_T_mean([c["T_est"] for c in inliers], np.ones(len(inliers), dtype=np.float64))

    return {
        "T_weighted": T_weighted,
        "T_unweighted": T_unweighted,
        "inliers": inliers,
        "outliers": outliers,
        "stats": {
            "num_candidates": len(candidates),
            "num_inliers": len(inliers),
            "num_outliers": len(outliers),
            "translation_deviation_median_m": t_med,
            "translation_deviation_mad_scaled_m": t_mad,
            "translation_inlier_threshold_m": t_thresh,
            "rotation_deviation_median_deg": r_med,
            "rotation_deviation_mad_scaled_deg": r_mad,
            "rotation_inlier_threshold_deg": r_thresh,
            "inlier_rule": "candidate deviation to robust consensus <= median + 3*MAD, with floors 0.30 m and 7 deg",
        },
    }


def compute_pair_candidates(pair_name, target_cam, method, static_det, moving_det_by_marker, colmap_poses, colmap_scale, moving_gt_poses):
    root_markers = sorted([
        marker for (cam, marker) in static_det.keys()
        if cam == ROOT_CAM and marker in moving_det_by_marker
    ])
    target_markers = sorted([
        marker for (cam, marker) in static_det.keys()
        if cam == target_cam and marker in moving_det_by_marker
    ])

    T_W_root = T_W_optical_from_sdf_model(ROOT_CAM)
    T_W_target = T_W_optical_from_sdf_model(target_cam)
    T_gt = invT(T_W_root) @ T_W_target

    candidates = []

    for root_marker in root_markers:
        for target_marker in target_markers:
            root_static = static_det[(ROOT_CAM, root_marker)]
            target_static = static_det[(target_cam, target_marker)]

            T_root_marker = T_from_detection(root_static)
            T_target_marker = T_from_detection(target_static)

            root_mov_rows = moving_det_by_marker.get(root_marker, [])
            target_mov_rows = moving_det_by_marker.get(target_marker, [])

            for root_mov in root_mov_rows:
                root_frame = int(root_mov["frame_idx"])
                if root_frame not in colmap_poses:
                    continue
                if method == "GT_motion" and root_frame not in moving_gt_poses:
                    continue

                T_movi_marker = T_from_detection(root_mov)
                T_root_movi = T_root_marker @ invT(T_movi_marker)

                for target_mov in target_mov_rows:
                    target_frame = int(target_mov["frame_idx"])
                    if target_frame not in colmap_poses:
                        continue
                    if method == "GT_motion" and target_frame not in moving_gt_poses:
                        continue

                    # Skip only exact self-canceling relay candidate.
                    if root_marker == target_marker and root_frame == target_frame:
                        continue

                    T_movj_marker = T_from_detection(target_mov)
                    T_target_movj = T_target_marker @ invT(T_movj_marker)

                    if method == "COLMAP_motion":
                        Tcw_i = colmap_poses[root_frame]
                        Tcw_j = colmap_poses[target_frame]
                        T_movi_movj = Tcw_i @ invT(Tcw_j)
                        T_movi_movj[:3, 3] *= colmap_scale
                    elif method == "GT_motion":
                        T_W_movi = moving_gt_poses[root_frame]
                        T_W_movj = moving_gt_poses[target_frame]
                        T_movi_movj = invT(T_W_movi) @ T_W_movj
                    else:
                        raise RuntimeError(f"Unknown method: {method}")

                    T_est = T_root_movi @ T_movi_movj @ invT(T_target_movj)

                    te = trans_error(T_est, T_gt)
                    re_ = rot_error_deg(T_est, T_gt)

                    qs = [
                        float(root_static["quality"]),
                        float(target_static["quality"]),
                        float(root_mov["quality"]),
                        float(target_mov["quality"]),
                    ]
                    combined_quality = float(np.prod([max(1e-12, q) for q in qs]) ** 0.25)

                    c = {
                        "pair": pair_name,
                        "root_camera": ROOT_CAM,
                        "target_camera": target_cam,
                        "method": method,
                        "root_marker": root_marker,
                        "target_marker": target_marker,
                        "root_frame": root_frame,
                        "target_frame": target_frame,
                        "root_static_quality": qs[0],
                        "target_static_quality": qs[1],
                        "root_moving_quality": qs[2],
                        "target_moving_quality": qs[3],
                        "combined_quality": combined_quality,
                        "T_est": T_est,
                        "T_gt": T_gt,
                        "translation_error_m": te,
                        "translation_error_cm": 100.0 * te,
                        "rotation_error_deg": re_,
                    }
                    c.update(pose_fields("estimated_target_in_root", T_est))
                    c.update(pose_fields("gt_target_in_root", T_gt))
                    candidates.append(c)

    return candidates


def serializable(row):
    out = {}
    for k, v in row.items():
        if k in ["T_est", "T_gt"]:
            continue
        if isinstance(v, bool):
            out[k] = "1" if v else "0"
        elif isinstance(v, (float, np.floating)):
            if math.isfinite(float(v)):
                out[k] = f"{float(v):.9f}"
            else:
                out[k] = ""
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = [serializable(r) for r in rows]

    fields = []
    for r in serial:
        for k in r.keys():
            if k not in fields:
                fields.append(k)

    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in serial:
            w.writerow(r)

    print("[OK] wrote:", path)


def aggregate_rows_for(pair_name, target_cam, method, candidates):
    agg = robust_aggregate(candidates)

    rows = []

    for aggregate_type, T_est in [
        ("weighted_mean_of_mad_inliers_no_gt_selection", agg["T_weighted"]),
        ("unweighted_mean_of_mad_inliers_no_gt_selection", agg["T_unweighted"]),
    ]:
        T_gt = candidates[0]["T_gt"]
        te = trans_error(T_est, T_gt)
        re_ = rot_error_deg(T_est, T_gt)

        best_oracle = min(candidates, key=lambda c: c["translation_error_m"] + 0.02 * c["rotation_error_deg"])

        row = {
            "pair": pair_name,
            "root_camera": ROOT_CAM,
            "target_camera": target_cam,
            "method": method,
            "aggregate_type": aggregate_type,
            "num_candidates": agg["stats"]["num_candidates"],
            "num_inliers": agg["stats"]["num_inliers"],
            "num_outliers": agg["stats"]["num_outliers"],
            "translation_error_m": te,
            "translation_error_cm": 100.0 * te,
            "rotation_error_deg": re_,
            "best_oracle_root_marker": best_oracle["root_marker"],
            "best_oracle_target_marker": best_oracle["target_marker"],
            "best_oracle_root_frame": best_oracle["root_frame"],
            "best_oracle_target_frame": best_oracle["target_frame"],
            "best_oracle_translation_error_cm": best_oracle["translation_error_cm"],
            "best_oracle_rotation_error_deg": best_oracle["rotation_error_deg"],
            "inlier_rule": agg["stats"]["inlier_rule"],
            **agg["stats"],
        }
        row.update(pose_fields("estimated_target_in_root", T_est))
        row.update(pose_fields("gt_target_in_root", T_gt))
        rows.append(row)

    return rows, agg


def main():
    print("=== Loading inputs ===")
    static_det = load_static_detections()
    moving_det, moving_by_marker = load_moving_detections()
    colmap_poses = load_colmap_poses()
    colmap_scale = load_colmap_scale()
    moving_gt_poses = load_moving_gt_poses()

    print(f"static detections: {len(static_det)} best camera-marker observations")
    print(f"moving detections: {len(moving_det)} frame-marker observations")
    print(f"moving markers: {sorted(moving_by_marker)}")
    print(f"COLMAP registered frames: {len(colmap_poses)}")
    print(f"COLMAP metric scale: {colmap_scale}")
    print(f"moving GT frames parsed: {len(moving_gt_poses)}")

    all_candidates = []
    all_aggregate_rows = []
    all_inliers = []
    all_outliers = []
    all_stats = []

    methods = ["COLMAP_motion"]
    if moving_gt_poses:
        methods.append("GT_motion")
    else:
        print("[WARN] No moving GT poses parsed; GT_motion oracle multichain will be skipped.")

    for pair_name, target_cam in PAIRS.items():
        for method in methods:
            print()
            print(f"=== Computing all combinations: {pair_name} / {method} ===")
            cands = compute_pair_candidates(
                pair_name=pair_name,
                target_cam=target_cam,
                method=method,
                static_det=static_det,
                moving_det_by_marker=moving_by_marker,
                colmap_poses=colmap_poses,
                colmap_scale=colmap_scale,
                moving_gt_poses=moving_gt_poses,
            )

            if not cands:
                print(f"[WARN] No candidates for {pair_name} / {method}")
                continue

            agg_rows, agg = aggregate_rows_for(pair_name, target_cam, method, cands)

            for c in cands:
                all_candidates.append(c)
            for c in agg["inliers"]:
                all_inliers.append(c)
            for c in agg["outliers"]:
                all_outliers.append(c)
            for row in agg_rows:
                all_aggregate_rows.append(row)

            stat = {
                "pair": pair_name,
                "target_camera": target_cam,
                "method": method,
                **agg["stats"],
            }
            all_stats.append(stat)

            main_row = [r for r in agg_rows if r["aggregate_type"] == "weighted_mean_of_mad_inliers_no_gt_selection"][0]
            print(
                f"[RESULT] {pair_name} / {method}: "
                f"candidates={main_row['num_candidates']}, "
                f"inliers={main_row['num_inliers']}, "
                f"outliers={main_row['num_outliers']}, "
                f"error={main_row['translation_error_cm']:.3f} cm, "
                f"{main_row['rotation_error_deg']:.3f} deg"
            )

    all_candidates_sorted = sorted(
        all_candidates,
        key=lambda c: (c["pair"], c["method"], -c["combined_quality"])
    )
    all_inliers_sorted = sorted(
        all_inliers,
        key=lambda c: (c["pair"], c["method"], -c["combined_quality"])
    )
    all_outliers_sorted = sorted(
        all_outliers,
        key=lambda c: (c["pair"], c["method"], -c["deviation_to_consensus_m"])
    )

    write_csv(OUT / "relay_chain_results.csv", all_candidates_sorted)
    write_csv(OUT / "relay_chain_all_candidates.csv", all_candidates_sorted)
    write_csv(OUT / "relay_chain_multichain_aggregate.csv", all_aggregate_rows)
    write_csv(OUT / "relay_chain_multichain_inliers.csv", all_inliers_sorted)
    write_csv(OUT / "relay_chain_multichain_outliers.csv", all_outliers_sorted)
    write_csv(OUT / "relay_chain_multichain_stats.csv", all_stats)

    summary = []
    summary.append("Moving relay multichain aggregation")
    summary.append("=" * 80)
    summary.append("")
    summary.append("Candidate generation:")
    summary.append("  All valid combinations are evaluated:")
    summary.append("  root_marker × root_moving_frame × target_marker × target_moving_frame")
    summary.append("  A candidate is valid if both moving frames are registered in COLMAP and all PnP observations exist.")
    summary.append("")
    summary.append("Outlier rule, no GT:")
    summary.append("  Compute robust consensus from all candidates.")
    summary.append("  Candidate is inlier if translation and rotation deviation are <= median + 3*MAD.")
    summary.append("  Floors: 0.30 m and 7 deg.")
    summary.append("")
    summary.append("Final estimate:")
    summary.append("  weighted mean of MAD inlier transforms, weighted by observable marker quality.")
    summary.append("  GT is used only after aggregation for evaluation.")
    summary.append("")

    for row in all_aggregate_rows:
        if row["aggregate_type"] != "weighted_mean_of_mad_inliers_no_gt_selection":
            continue
        summary.append(
            f"{row['pair']} / {row['method']}: "
            f"candidates={row['num_candidates']}, "
            f"inliers={row['num_inliers']}, "
            f"outliers={row['num_outliers']}, "
            f"error={row['translation_error_cm']:.3f} cm, "
            f"{row['rotation_error_deg']:.3f} deg"
        )
        summary.append(
            f"  oracle best individual candidate, eval-only: "
            f"m{row['best_oracle_root_marker']}@f{row['best_oracle_root_frame']} -> "
            f"m{row['best_oracle_target_marker']}@f{row['best_oracle_target_frame']}, "
            f"{row['best_oracle_translation_error_cm']:.3f} cm, "
            f"{row['best_oracle_rotation_error_deg']:.3f} deg"
        )

    (OUT / "summary.txt").write_text("\n".join(summary) + "\n")
    print("[OK] wrote:", OUT / "summary.txt")
    print()
    print("\n".join(summary))


if __name__ == "__main__":
    main()
