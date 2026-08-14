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


def open_video_without_autorotation(path):
    """Compatibility name; returned frames now use the canonical display transform."""
    return open_oriented_video(path)


def balanced_candidate_indices(
    frame_count: int,
    source_fps: float,
    target_hz: float,
    *,
    tested: set[int] | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []
    step = max(1, int(round(max(source_fps, target_hz) / target_hz)))
    excluded = tested or set()
    return [
        frame_index
        for frame_index in range(0, frame_count, step)
        if frame_index not in excluded
    ]


def detect_checkerboard_balanced(
    gray: np.ndarray,
    pattern: tuple[int, int],
    preview_max_dimension: int,
) -> tuple[bool, np.ndarray | None]:
    height, width = gray.shape[:2]
    scale = min(
        1.0,
        float(preview_max_dimension) / float(max(width, height)),
    )
    preview = (
        cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else gray
    )
    found, corners = cv2.findChessboardCornersSB(
        preview,
        pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )
    if not found or corners is None:
        return False, None
    full_resolution = (corners / scale).astype(np.float32)
    cv2.cornerSubPix(
        gray,
        full_resolution,
        (7, 7),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        ),
    )
    return True, full_resolution


def _detect_candidates(
    video: Path,
    *,
    candidate_indices: list[int],
    pass_label: str,
    pattern: tuple[int, int],
    reported_frames: int,
    source_fps: float,
    columns: int,
    rows: int,
    preview_max_dimension: int,
    detections: list[dict],
) -> int:
    if not candidate_indices:
        return 0
    candidate_set = set(candidate_indices)
    capture = open_video_without_autorotation(video)
    tested = 0
    next_progress_frame = 0
    frame_index = 0
    while frame_index < reported_frames:
        ok = capture.grab()
        if not ok:
            break
        if frame_index not in candidate_set:
            frame_index += 1
            continue
        ok, frame = capture.retrieve()
        if not ok:
            frame_index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_checkerboard_balanced(
            gray, pattern, preview_max_dimension
        )
        if found and corners is not None:
            metrics = board_metrics(
                gray,
                corners,
                columns,
                rows,
                frame_index,
                max(reported_frames, frame_index + 1),
            )
            detections.append(
                {
                    "frame_index": frame_index,
                    "time_s": frame_index / max(source_fps, 1e-9),
                    "corners": corners.reshape(-1, 1, 2),
                    **metrics,
                }
            )
        tested += 1
        if frame_index >= next_progress_frame:
            print(
                "RIGCAL_PROGRESS "
                f"current={min(frame_index + 1, reported_frames)} "
                f"total={reported_frames} unit=frames "
                f"label=intrinsics_{pass_label}",
                flush=True,
            )
            next_progress_frame = frame_index + 50
        frame_index += 1
    capture.release()
    return tested



def board_metrics(
    gray: np.ndarray,
    corners: np.ndarray,
    cols: int,
    rows: int,
    frame_index: int,
    frame_count: int,
) -> dict:
    height, width = gray.shape[:2]
    points = corners.reshape(-1, 2).astype(np.float64)

    center = points.mean(axis=0)
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    area_fraction = area / float(width * height)

    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    padding = 30

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)

    roi = gray[y0:y1, x0:x1]

    sharpness = float(
        cv2.Laplacian(roi, cv2.CV_64F).var()
    )

    horizontal = points[cols - 1] - points[0]
    angle = math.atan2(horizontal[1], horizontal[0])

    top = np.linalg.norm(points[cols - 1] - points[0])
    bottom = np.linalg.norm(points[-1] - points[-cols])
    left = np.linalg.norm(points[(rows - 1) * cols] - points[0])
    right = np.linalg.norm(points[-1] - points[cols - 1])

    top_bottom = math.log(
        max(top, 1e-9) / max(bottom, 1e-9)
    )

    left_right = math.log(
        max(left, 1e-9) / max(right, 1e-9)
    )

    time_fraction = frame_index / max(frame_count - 1, 1)

    feature = np.array(
        [
            center[0] / width,
            center[1] / height,
            math.log(max(area_fraction, 1e-9)),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle),
            top_bottom,
            left_right,
            0.25 * time_fraction,
        ],
        dtype=np.float64,
    )

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "area_fraction": float(area_fraction),
        "angle_deg": float(math.degrees(angle)),
        "top_bottom_log_ratio": float(top_bottom),
        "left_right_log_ratio": float(left_right),
        "sharpness": sharpness,
        "feature": feature,
    }


def select_diverse(
    detections: list[dict],
    maximum: int,
    minimum_frame_gap: int,
) -> list[int]:
    if len(detections) <= maximum:
        return list(range(len(detections)))

    features = np.vstack(
        [item["feature"] for item in detections]
    )

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-9] = 1.0

    normalized = (features - mean) / std

    sharpness = np.asarray(
        [item["sharpness"] for item in detections],
        dtype=np.float64,
    )

    sharpness = (
        sharpness - sharpness.min()
    ) / max(float(np.ptp(sharpness)), 1e-9)

    selected = [int(np.argmax(sharpness))]

    while len(selected) < maximum:
        selected_features = normalized[selected]

        distances = np.linalg.norm(
            normalized[:, None, :]
            - selected_features[None, :, :],
            axis=2,
        )

        score = distances.min(axis=1) + 0.15 * sharpness
        score[selected] = -np.inf

        order = np.argsort(score)[::-1]
        chosen = None

        for candidate in order:
            candidate = int(candidate)
            candidate_frame = detections[candidate]["frame_index"]

            sufficiently_separated = all(
                abs(
                    candidate_frame
                    - detections[index]["frame_index"]
                ) >= minimum_frame_gap
                for index in selected
            )

            if sufficiently_separated:
                chosen = candidate
                break

        if chosen is None:
            chosen = int(order[0])

        selected.append(chosen)

    return sorted(
        selected,
        key=lambda index: detections[index]["frame_index"],
    )


