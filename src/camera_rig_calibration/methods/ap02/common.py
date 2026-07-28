"""Shared AP02 geometry and artifact helpers."""

import csv
import math
from pathlib import Path
import json
import os

import cv2
import numpy as np

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
SHARED_RAW_ROOT = Path(
    "results/simulation/baseline/route2/raw_images"
)
MARKER_LENGTH_M = 0.170
REF_MARKER_ID = 14


AP02_ROOT = Path(
    os.environ.get(
        "AP02_ROOT",
        "workspace/standalone_methods/ap02",
    )
)

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

DEFAULT_MARKER_LENGTH_M = MARKER_LENGTH_M
DEFAULT_REF_MARKER_ID = REF_MARKER_ID


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows, fields):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def read_csv(path: Path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_T(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invT(transform):
    transform = np.asarray(transform, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def R_to_rpy_deg(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = math.atan2(
        -rotation[2, 0],
        math.hypot(rotation[0, 0], rotation[1, 0]),
    )
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return [math.degrees(value) for value in (roll, pitch, yaw)]


def R_to_rvec(rotation):
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return vector.reshape(3)


def rvec_to_R(vector):
    rotation, _ = cv2.Rodrigues(
        np.asarray(vector, dtype=np.float64).reshape(3, 1)
    )
    return rotation.astype(np.float64)


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
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing camera_info JSON: {path}")

    data = json.loads(path.read_text())

    # Compatibility with both AP02 raw JSON shape and normalized camera_io shape.
    if "k" not in data and "K" in data:
        data["k"] = data["K"]
    if "d" not in data and "D" in data:
        data["d"] = data["D"]
    if "fx" not in data and "k" in data:
        data["fx"] = float(data["k"][0])
        data["fy"] = float(data["k"][4])
        data["cx"] = float(data["k"][2])
        data["cy"] = float(data["k"][5])

    return data


def write_json(path: Path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


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
    # T_observer_marker maps marker coordinates into observer/camera optical coordinates.
    # T_ref_observer maps observer/camera optical coordinates into reference-marker coordinates.
    # Therefore: T_ref_marker = T_ref_observer @ T_observer_marker
    return T_ref_observer @ T_observer_marker
