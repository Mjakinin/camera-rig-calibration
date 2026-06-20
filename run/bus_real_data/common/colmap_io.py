#!/usr/bin/env python3
from pathlib import Path
import numpy as np

from common.geometry import qvec_to_R, make_T, invT


def parse_cameras_txt(path):
    path = Path(path)
    cams = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]

        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.zeros(5, dtype=np.float64)
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.zeros(5, dtype=np.float64)
        elif model == "SIMPLE_RADIAL":
            f, cx, cy, k = params[:4]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k, 0, 0, 0, 0], dtype=np.float64)
        elif model == "RADIAL":
            f, cx, cy, k1, k2 = params[:5]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
        elif model == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
        else:
            raise RuntimeError(f"Unsupported COLMAP camera model: {model}")

        cams[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "params": params,
            "K": K,
            "D": D,
        }

    return cams


def parse_images_txt(path):
    path = Path(path)
    images = {}

    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]

            R_cw = qvec_to_R(qvec)
            T_cam_col = make_T(R_cw, tvec)
            T_col_cam = invT(T_cam_col)

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "R_cw": R_cw,
                "T_cam_col": T_cam_col,
                "T_col_cam": T_col_cam,
            }
            i += 2
        else:
            i += 1

    return images


def parse_points3D_txt(path):
    path = Path(path)
    pts = []
    if not path.exists():
        return pts

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        pts.append({
            "point3d_id": int(parts[0]),
            "x_colmap": float(parts[1]),
            "y_colmap": float(parts[2]),
            "z_colmap": float(parts[3]),
            "r": int(parts[4]),
            "g": int(parts[5]),
            "b": int(parts[6]),
            "error": float(parts[7]),
        })
    return pts
