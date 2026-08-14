#!/usr/bin/env python3
"""Calibrate a managed camera-intrinsics profile from video or images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from camera_rig_calibration.input.video_geometry import open_oriented_video
except ImportError:  # pragma: no cover - direct script fallback
    from video_geometry import open_oriented_video


def object_points(cols: int, rows: int) -> np.ndarray:
    points = np.zeros(
        (cols * rows, 3),
        dtype=np.float32,
    )

    points[:, :2] = np.mgrid[
        0:cols,
        0:rows,
    ].T.reshape(-1, 2)

    return points


def calibrate(
    detections: list[dict],
    indices: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    rational: bool,
) -> dict:
    objects = [
        object_template.copy()
        for _ in indices
    ]

    images = [
        detections[index]["corners"].astype(np.float32)
        for index in indices
    ]

    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0

    rms, matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            objects,
            images,
            image_size,
            None,
            None,
            flags=flags,
        )
    )

    per_view = []

    for obj, img, rvec, tvec in zip(
        objects,
        images,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            matrix,
            distortion,
        )

        difference = (
            img.reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        per_view.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            difference * difference,
                            axis=1,
                        )
                    )
                )
            )
        )

    return {
        "rms": float(rms),
        "K": matrix,
        "D": distortion.reshape(-1),
        "per_view": per_view,
        "rational": rational,
    }


def holdout_error(
    detections: list[dict],
    indices: list[int],
    object_template: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> list[float]:
    errors = []

    for index in indices:
        corners = detections[index]["corners"].astype(
            np.float32
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_template,
            corners,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            continue

        projected, _ = cv2.projectPoints(
            object_template,
            rvec,
            tvec,
            K,
            D,
        )

        difference = (
            corners.reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            difference * difference,
                            axis=1,
                        )
                    )
                )
            )
        )

    return errors


def model_comparison(
    detections: list[dict],
    selected: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[bool, dict]:
    holdout = selected[::5]
    holdout_set = set(holdout)

    training = [
        index
        for index in selected
        if index not in holdout_set
    ]

    if len(training) < 20 or len(holdout) < 5:
        training = selected
        holdout = selected

    standard = calibrate(
        detections,
        training,
        object_template,
        image_size,
        rational=False,
    )

    rational = calibrate(
        detections,
        training,
        object_template,
        image_size,
        rational=True,
    )

    standard_errors = holdout_error(
        detections,
        holdout,
        object_template,
        standard["K"],
        standard["D"],
    )

    rational_errors = holdout_error(
        detections,
        holdout,
        object_template,
        rational["K"],
        rational["D"],
    )

    standard_median = float(
        np.median(standard_errors)
    )

    rational_median = float(
        np.median(rational_errors)
    )

    use_rational = rational_median < 0.98 * standard_median

    comparison = {
        "training_views": len(training),
        "holdout_views": len(holdout),
        "standard_holdout_median_rmse_px": standard_median,
        "rational_holdout_median_rmse_px": rational_median,
        "selected_model": (
            "rational_polynomial"
            if use_rational
            else "plumb_bob"
        ),
    }

    return use_rational, comparison


def calibrate_with_outlier_filter(
    detections: list[dict],
    selected: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    rational: bool,
) -> tuple[list[int], dict, list[dict]]:
    active = list(selected)
    removed = []

    while True:
        result = calibrate(
            detections,
            active,
            object_template,
            image_size,
            rational=rational,
        )

        errors = np.asarray(
            result["per_view"],
            dtype=np.float64,
        )

        median = float(np.median(errors))
        mad = float(
            np.median(
                np.abs(errors - median)
            )
        )

        robust_limit = median + 3.0 * max(
            1.4826 * mad,
            0.05,
        )

        threshold = max(1.5, robust_limit)

        worst_position = int(np.argmax(errors))
        worst_error = float(errors[worst_position])

        if (
            worst_error <= threshold
            or len(active) <= 30
            or len(removed) >= 15
        ):
            result["median_view_error"] = median
            result["maximum_view_error"] = worst_error
            result["outlier_threshold"] = threshold
            return active, result, removed

        removed_index = active.pop(worst_position)

        removed.append(
            {
                "detection_index": removed_index,
                "frame_index": detections[
                    removed_index
                ]["frame_index"],
                "reprojection_rmse_px": worst_error,
            }
        )

        print(
            "[FILTER]",
            f"frame={detections[removed_index]['frame_index']}",
            f"rmse={worst_error:.4f}px",
        )


