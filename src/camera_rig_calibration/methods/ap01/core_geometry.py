"""AP01 scientific core.

The functions in this module preserve the established marker-direct and
moving-COLMAP-relay mathematics.  The v4 stage modules import these functions
directly; no path mutation or simulated command-line invocation is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .contracts import AP01MethodContract, resolve_ap01_method_contract


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]
ROOT_CAMERA = "cam_edge_3"



from .core_io import safe_float
def qvec_to_R(values: list[float]) -> np.ndarray:
    qw, qx, qy, qz = values
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def T_from_observation(row: dict[str, str]) -> np.ndarray:
    rvec = np.asarray([
        safe_float(row, "rvec_x"),
        safe_float(row, "rvec_y"),
        safe_float(row, "rvec_z"),
    ], dtype=np.float64)
    tvec = np.asarray([
        safe_float(row, "tvec_x_m"),
        safe_float(row, "tvec_y_m"),
        safe_float(row, "tvec_z_m"),
    ], dtype=np.float64)
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        raise RuntimeError("Non-finite ArUco PnP pose")
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


def parse_colmap_poses(images_txt: Path) -> dict[int, np.ndarray]:
    if not images_txt.is_file():
        raise RuntimeError(f"Missing COLMAP images.txt: {images_txt}")
    result = {}
    for raw in images_txt.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10 or not parts[9].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        matches = re.findall(r"(\d+)", Path(parts[9]).stem)
        if not matches:
            continue
        frame = int(matches[-1])
        R = qvec_to_R([float(v) for v in parts[1:5]])
        t = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        result[frame] = make_T(R, t)  # world -> camera
    if not result:
        raise RuntimeError(f"No AP01 moving poses parsed from {images_txt}")
    return result


def observation_quality(row: dict[str, str], width: float, height: float) -> float:
    area = max(safe_float(row, "area_px2", 0.0), 1.0)
    distance = max(safe_float(row, "distance_m", 99.0), 0.1)
    center_u = safe_float(row, "center_u", width / 2.0)
    center_v = safe_float(row, "center_v", height / 2.0)
    center_norm = math.hypot(center_u - width / 2.0, center_v - height / 2.0)
    center_norm /= max(math.hypot(width / 2.0, height / 2.0), 1.0)
    return math.sqrt(area) / (distance * (1.0 + center_norm))


def marker_area_from_corners(row: dict[str, str]) -> float:
    points = np.asarray(
        [
            [
                safe_float(row, f"corner{index}_u"),
                safe_float(row, f"corner{index}_v"),
            ]
            for index in range(4)
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(points)):
        return float("nan")
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def baseline_detection_quality(
    row: dict[str, str], width: float = 1280.0, height: float = 720.0
) -> tuple[float, dict[str, float]]:
    """Return area/(distance^2*(1+center_norm)) for baseline ranking."""

    distance = safe_float(row, "distance_m", 99.0)
    area = marker_area_from_corners(row)
    center_u = safe_float(row, "center_u")
    center_v = safe_float(row, "center_v")
    if math.isfinite(center_u) and math.isfinite(center_v):
        center_norm = math.hypot(
            center_u - width / 2.0, center_v - height / 2.0
        ) / max(math.hypot(width / 2.0, height / 2.0), 1.0)
    else:
        center_norm = 1.0
    if not math.isfinite(distance) or distance <= 0.0:
        distance = 99.0
    if not math.isfinite(area) or area <= 0.0:
        area = 1.0
    if not math.isfinite(center_norm):
        center_norm = 1.0
    return area / (distance * distance * (1.0 + center_norm)), {
        "area_px2_from_corners": area,
        "distance_m": distance,
        "center_norm": center_norm,
        "quality_image_width_px": float(width),
        "quality_image_height_px": float(height),
    }
