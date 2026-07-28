"""Distortion-aware entry point for the AP02 graph bundle adjustment.

The underlying parameterization and output contract remain unchanged. Only the
reprojection model is replaced so that real ROS camera_info distortion is not
silently discarded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


from . import optimize_core as core


def distortion_from_row(row: dict) -> tuple[str, np.ndarray]:
    model = str(row.get("distortion_model", "plumb_bob") or "plumb_bob")
    model = model.strip().lower()
    values = []
    for index in range(8):
        value = core.safe_float(row, f"d{index}", 0.0)
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
    T_observer_ref = core.invT(T_ref_observer)

    fx = core.safe_float(row, "fx")
    fy = core.safe_float(row, "fy")
    cx = core.safe_float(row, "cx")
    cy = core.safe_float(row, "cy")
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(K)):
        raise RuntimeError(f"Invalid intrinsics in AP02 observation: {row}")

    distortion_model, distortion = distortion_from_row(row)
    marker_length_m = core.safe_float(row, "marker_length_m", 0.170)
    object_points = core.marker_object_points(marker_length_m)

    residuals = []
    for index, point_marker in enumerate(object_points):
        observed = np.array(
            [
                core.safe_float(row, f"corner{index}_u"),
                core.safe_float(row, f"corner{index}_v"),
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
core.observation_residuals = observation_residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["static_only", "with_moving"], required=True
    )
    parser.add_argument(
        "--ref-marker-id", type=int, default=core.DEFAULT_REF_MARKER_ID
    )
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--ap02-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--initialization-root", type=Path, required=True)
    parser.add_argument(
        "--robust-loss",
        choices=["soft_l1", "huber", "linear"],
        default="soft_l1",
    )
    parser.add_argument("--robust-loss-scale-px", type=float, default=3.0)
    args = parser.parse_args()
    if args.robust_loss_scale_px <= 0:
        parser.error("--robust-loss-scale-px must be greater than zero")
    ap02_root = args.ap02_root.resolve()
    observations = args.observations.resolve()
    initialization_root = args.initialization_root.resolve()

    scipy_least_squares = core.least_squares
    optimizer_report: dict[str, object] = {}

    def robust_cost(residuals: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=np.float64)
        if args.robust_loss == "linear":
            return 0.5 * float(np.sum(residuals * residuals))
        scale = float(args.robust_loss_scale_px)
        z = (residuals / scale) ** 2
        if args.robust_loss == "soft_l1":
            rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
        else:  # huber
            rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
        return 0.5 * scale * scale * float(np.sum(rho))

    def configured_least_squares(function, initial, **kwargs):
        kwargs["loss"] = args.robust_loss
        kwargs["f_scale"] = args.robust_loss_scale_px
        initial_residuals = np.asarray(function(initial), dtype=np.float64)
        started = time.monotonic()
        result = scipy_least_squares(function, initial, **kwargs)
        optimizer_report.update(
            {
                "mode": args.mode,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "maximum_function_evaluations": int(args.max_nfev),
                "initial_cost": robust_cost(initial_residuals),
                "final_cost": float(result.cost),
                "loss": args.robust_loss,
                "loss_scale_px": float(args.robust_loss_scale_px),
                "runtime_seconds": time.monotonic() - started,
            }
        )
        return result

    # The numerical core owns residuals, pose parameterization and optimizer
    # invocation. This adapter changes only the two explicitly configurable
    # SciPy loss arguments; defaults remain soft_l1 and 3.0 px.
    core.least_squares = configured_least_squares

    try:
        core.run_ba(
            args.mode,
            args.ref_marker_id,
            args.max_nfev,
            ap02_root=ap02_root,
            observations_csv=observations,
            initialization_root=initialization_root,
        )
        report_path = (
            ap02_root
            / "07_graph_ba"
            / args.mode
            / "optimizer_report.json"
        )
        report_path.write_text(
            json.dumps(optimizer_report, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        core.least_squares = scipy_least_squares


if __name__ == "__main__":
    main()
