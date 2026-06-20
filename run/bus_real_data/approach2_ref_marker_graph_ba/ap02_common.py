#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

import numpy as np


AP02_ROOT = Path("results/bus_real_data/02_ref_marker_graph_ba")
SHARED_RAW_ROOT = Path("results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1")

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
MOVING_CAMERA = "moving_calib_camera"

IMAGE_TOPICS = {
    "cam_edge_0": "/bus_real_data/cam_edge_0/image",
    "cam_edge_1": "/bus_real_data/cam_edge_1/image",
    "cam_edge_3": "/bus_real_data/cam_edge_3/image",
    "cam_edge_5": "/bus_real_data/cam_edge_5/image",
    "moving_calib_camera": "/bus_real_data/moving_calib_camera/image",
}

CAMERA_INFO_TOPICS = {
    "cam_edge_0": "/bus_real_data/cam_edge_0/camera_info",
    "cam_edge_1": "/bus_real_data/cam_edge_1/camera_info",
    "cam_edge_3": "/bus_real_data/cam_edge_3/camera_info",
    "cam_edge_5": "/bus_real_data/cam_edge_5/camera_info",
    "moving_calib_camera": "/bus_real_data/moving_calib_camera/camera_info",
}

DEFAULT_MARKER_LENGTH_M = 0.170
DEFAULT_REF_MARKER_ID = 14


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows, fields):
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing file: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def safe_float(row, key, default=float("nan")):
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def safe_int(row, key, default=None):
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def camera_info_to_dict(msg):
    return {
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
        "k": [float(x) for x in msg.k],
        "r": [float(x) for x in msg.r],
        "p": [float(x) for x in msg.p],
        "fx": float(msg.k[0]),
        "fy": float(msg.k[4]),
        "cx": float(msg.k[2]),
        "cy": float(msg.k[5]),
    }


def load_camera_info_json(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing camera_info JSON: {path}")
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data):
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def rvec_to_R(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)

    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def R_to_rvec(R):
    R = np.asarray(R, dtype=np.float64)
    arg = (float(np.trace(R)) - 1.0) / 2.0
    arg = max(-1.0, min(1.0, arg))
    theta = math.acos(arg)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    denom = 2.0 * math.sin(theta)
    axis = np.array([
        (R[2, 1] - R[1, 2]) / denom,
        (R[0, 2] - R[2, 0]) / denom,
        (R[1, 0] - R[0, 1]) / denom,
    ], dtype=np.float64)

    return axis * theta


def R_to_rpy_deg(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose_row(entity_type, entity_id, T, source=""):
    rvec = R_to_rvec(T[:3, :3])
    rpy = R_to_rpy_deg(T[:3, :3])
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source": source,
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
        "roll_deg": float(rpy[0]),
        "pitch_deg": float(rpy[1]),
        "yaw_deg": float(rpy[2]),
        "rvec_x": float(rvec[0]),
        "rvec_y": float(rvec[1]),
        "rvec_z": float(rvec[2]),
    }


def pose_fields():
    return [
        "entity_type", "entity_id", "source",
        "x_m", "y_m", "z_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rvec_x", "rvec_y", "rvec_z",
    ]


def T_from_detection_row(row):
    rvec = np.array([
        safe_float(row, "rvec_x"),
        safe_float(row, "rvec_y"),
        safe_float(row, "rvec_z"),
    ], dtype=np.float64)

    tvec = np.array([
        safe_float(row, "tvec_x_m"),
        safe_float(row, "tvec_y_m"),
        safe_float(row, "tvec_z_m"),
    ], dtype=np.float64)

    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None

    return make_T(rvec_to_R(rvec), tvec)


def make_observer_known_from_marker(T_ref_marker, T_observer_marker):
    # T_observer_marker maps marker coordinates into observer/camera optical coordinates.
    # T_ref_marker maps marker coordinates into reference-marker coordinates.
    # T_ref_marker = T_ref_observer @ T_observer_marker
    return T_ref_marker @ invT(T_observer_marker)


def make_marker_known_from_observer(T_ref_observer, T_observer_marker):
    # T_ref_marker = T_ref_observer @ T_observer_marker
    return T_ref_observer @ T_observer_marker
