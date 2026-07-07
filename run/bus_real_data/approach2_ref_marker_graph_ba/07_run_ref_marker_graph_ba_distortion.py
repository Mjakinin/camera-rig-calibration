#!/usr/bin/env python3
"""Distortion-aware entry point for the AP02 graph bundle adjustment.

The underlying parameterization and output contract remain unchanged. Only the
reprojection model is replaced so that real ROS camera_info distortion is not
silently discarded.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "07_run_ref_marker_graph_ba.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("ap02_ba_legacy", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load AP02 BA implementation: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def distortion_from_row(row: dict) -> tuple[str, np.ndarray]:
    model = str(row.get("distortion_model", "plumb_bob") or "plumb_bob")
    model = model.strip().lower()
    values = []
    for index in range(8):
        value = legacy.safe_float(row, f"d{index}", 0.0)
        values.append(value if np.isfinite(value) else 0.0)

    if model in {"equidistant", "fisheye"}:
        return "equidistant", np.asarray(values[:4], dtype=np.float64)
    if model in {"", "none", "plumb_bob", "rational_polynomial"}:
        # cv2.projectPoints accepts 4, 5, 8, 12 or 14 coefficients. Eight
        # coefficients preserve ROS plumb_bob/rational-polynomial values.
        return "plumb_bob", np.asarray(values, dtype=np.float64)
    raise RuntimeError(f"Unsupported AP02 distortion model: {model!r}")


def project_camera_point(
    point_camera: np.ndarray,
    K: np.ndarray,
    distortion_model: str,
    distortion: np.ndarray,
) -> np.ndarray | None:
    point_camera = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if point_camera[2] <= 1e-9:
        return None

    if distortion_model == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            point_camera.reshape(1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            distortion.reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            point_camera.reshape(1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            distortion.reshape(-1, 1),
        )
    return projected.reshape(2).astype(np.float64)


def observation_residuals(row, marker_poses, observer_poses):
    marker_id = int(float(row["marker_id"]))
    observer_id = row["observer_id"]

    T_ref_marker = marker_poses[marker_id]
    T_ref_observer = observer_poses[observer_id]
    T_observer_ref = legacy.invT(T_ref_observer)

    fx = legacy.safe_float(row, "fx")
    fy = legacy.safe_float(row, "fy")
    cx = legacy.safe_float(row, "cx")
    cy = legacy.safe_float(row, "cy")
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(K)):
        raise RuntimeError(f"Invalid intrinsics in AP02 observation: {row}")

    distortion_model, distortion = distortion_from_row(row)
    marker_length_m = legacy.safe_float(row, "marker_length_m", 0.170)
    object_points = legacy.marker_object_points(marker_length_m)

    residuals = []
    for index, point_marker in enumerate(object_points):
        observed = np.array(
            [
                legacy.safe_float(row, f"corner{index}_u"),
                legacy.safe_float(row, f"corner{index}_v"),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(observed)):
            continue

        point_marker_h = np.array(
            [point_marker[0], point_marker[1], point_marker[2], 1.0],
            dtype=np.float64,
        )
        point_ref = T_ref_marker @ point_marker_h
        point_observer = T_observer_ref @ point_ref
        projected = project_camera_point(
            point_observer[:3], K, distortion_model, distortion
        )
        if projected is None:
            residuals.extend([1000.0, 1000.0])
        else:
            residuals.extend((projected - observed).tolist())

    return residuals


# The existing BA uses this global function from all residual, diagnostic and
# reporting paths. Replacing it here keeps every output internally consistent.
legacy.observation_residuals = observation_residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["static_only", "with_moving"], required=True
    )
    parser.add_argument(
        "--ref-marker-id", type=int, default=legacy.DEFAULT_REF_MARKER_ID
    )
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--moving-stride", type=int, default=1)
    parser.add_argument("--max-moving-frames", type=int, default=0)
    parser.add_argument(
        "--moving-selection",
        choices=["smart", "all", "stride"],
        default="smart",
    )
    parser.add_argument("--top-per-marker", type=int, default=8)
    parser.add_argument("--top-per-pair", type=int, default=4)
    args = parser.parse_args()

    legacy.run_ba(
        args.mode,
        args.ref_marker_id,
        args.max_nfev,
        args.moving_stride,
        args.max_moving_frames,
        args.moving_selection,
        args.top_per_marker,
        args.top_per_pair,
    )


if __name__ == "__main__":
    main()
