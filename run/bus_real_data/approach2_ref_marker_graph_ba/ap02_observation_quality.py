#!/usr/bin/env python3

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np


def safe_float(
    row: dict[str, str],
    key: str,
    default: float = float("nan"),
) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("pnp_success", "")).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0

    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def observed_corners(row: dict[str, str]) -> np.ndarray | None:
    corners = []

    for index in range(4):
        u = safe_float(row, f"corner{index}_u")
        v = safe_float(row, f"corner{index}_v")

        if not math.isfinite(u) or not math.isfinite(v):
            return None

        corners.append([u, v])

    return np.asarray(corners, dtype=np.float64)


def pnp_reprojection_rmse(row: dict[str, str]) -> float:
    if not is_success(row):
        return float("inf")

    rvec = np.array(
        [
            safe_float(row, "rvec_x"),
            safe_float(row, "rvec_y"),
            safe_float(row, "rvec_z"),
        ],
        dtype=np.float64,
    )

    tvec = np.array(
        [
            safe_float(row, "tvec_x_m"),
            safe_float(row, "tvec_y_m"),
            safe_float(row, "tvec_z_m"),
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(rvec)):
        return float("inf")

    if not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
        return float("inf")

    corners = observed_corners(row)

    if corners is None:
        return float("inf")

    fx = safe_float(row, "fx")
    fy = safe_float(row, "fy")
    cx = safe_float(row, "cx")
    cy = safe_float(row, "cy")

    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        return float("inf")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    marker_length = safe_float(
        row,
        "marker_length_m",
        0.170,
    )

    object_points = marker_object_points(marker_length)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        np.zeros((5, 1), dtype=np.float64),
    )

    projected = projected.reshape(-1, 2)

    error = projected - corners

    return float(
        math.sqrt(
            np.mean(
                np.sum(error * error, axis=1)
            )
        )
    )


def estimated_border_margin_px(row: dict[str, str]) -> float:
    corners = observed_corners(row)

    if corners is None:
        return 0.0

    cx = safe_float(row, "cx")
    cy = safe_float(row, "cy")

    if not math.isfinite(cx) or not math.isfinite(cy):
        return 0.0

    # The simulated cameras use approximately centred principal points.
    width = max(1.0, 2.0 * cx)
    height = max(1.0, 2.0 * cy)

    margins = []

    for u, v in corners:
        margins.extend(
            [
                u,
                width - 1.0 - u,
                v,
                height - 1.0 - v,
            ]
        )

    return max(0.0, float(min(margins)))


def observation_score(row: dict[str, str]) -> float:
    """
    Higher is better.

    The score penalizes:
    - poor PnP reprojection,
    - small marker area,
    - excessive distance,
    - marker corners near the image border,
    - invalid / negative depth.
    """

    if not is_success(row):
        return 0.0

    area = safe_float(row, "area_px2", 0.0)
    distance = safe_float(row, "distance_m", 99.0)
    depth = safe_float(row, "tvec_z_m", -1.0)
    rmse = pnp_reprojection_rmse(row)
    margin = estimated_border_margin_px(row)

    if area < 64.0:
        return 0.0

    if distance <= 0.0 or depth <= 0.0:
        return 0.0

    if not math.isfinite(rmse) or rmse > 25.0:
        return 0.0

    area_term = math.sqrt(area)
    distance_term = 1.0 / math.sqrt(max(distance, 0.25))
    reprojection_term = 1.0 / ((1.0 + rmse) ** 2)

    # Avoid eliminating valid observations merely because they are
    # relatively close to an image border.
    border_term = min(
        1.0,
        max(0.20, margin / 40.0),
    )

    return float(
        area_term
        * distance_term
        * reprojection_term
        * border_term
    )


def frame_number(observer_id: str) -> int:
    try:
        return int(observer_id.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 10**9


def select_smart_moving_observations(
    rows: list[dict[str, str]],
    *,
    ref_marker_id: int,
    top_per_marker: int,
    top_per_pair: int,
    max_frames: int = 0,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """
    Select moving-camera keyframes without regular stride sampling.

    Always retains:
    - every valid frame observing the reference marker,
    - strongest frames per marker ID,
    - strongest co-observation frames per marker pair.

    Marker-pair retention is important for bridge construction.
    """

    by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row.get("observer_type") != "moving":
            continue

        if observation_score(row) <= 0.0:
            continue

        by_frame[row["observer_id"]].append(row)

    marker_scores_by_frame: dict[str, dict[int, float]] = {}

    for observer_id, frame_rows in by_frame.items():
        marker_scores: dict[int, float] = {}

        for row in frame_rows:
            marker_id = int(float(row["marker_id"]))
            score = observation_score(row)

            marker_scores[marker_id] = max(
                score,
                marker_scores.get(marker_id, 0.0),
            )

        marker_scores_by_frame[observer_id] = marker_scores

    reasons: dict[str, set[str]] = defaultdict(set)

    # All reference-marker observations are mandatory.
    for observer_id, scores in marker_scores_by_frame.items():
        if ref_marker_id in scores:
            reasons[observer_id].add(
                f"reference_marker_{ref_marker_id}"
            )

    per_marker: dict[int, list[tuple[float, str]]] = defaultdict(list)

    for observer_id, scores in marker_scores_by_frame.items():
        for marker_id, score in scores.items():
            per_marker[marker_id].append(
                (score, observer_id)
            )

    for marker_id, candidates in per_marker.items():
        candidates.sort(
            key=lambda item: (
                item[0],
                -frame_number(item[1]),
            ),
            reverse=True,
        )

        for _, observer_id in candidates[:top_per_marker]:
            reasons[observer_id].add(
                f"top_marker_{marker_id}"
            )

    per_pair: dict[
        tuple[int, int],
        list[tuple[float, str]],
    ] = defaultdict(list)

    for observer_id, scores in marker_scores_by_frame.items():
        marker_ids = sorted(scores)

        for marker_a, marker_b in itertools.combinations(
            marker_ids,
            2,
        ):
            pair_score = min(
                scores[marker_a],
                scores[marker_b],
            )

            per_pair[(marker_a, marker_b)].append(
                (pair_score, observer_id)
            )

    for pair, candidates in per_pair.items():
        candidates.sort(
            key=lambda item: (
                item[0],
                -frame_number(item[1]),
            ),
            reverse=True,
        )

        for _, observer_id in candidates[:top_per_pair]:
            reasons[observer_id].add(
                f"top_pair_{pair[0]}_{pair[1]}"
            )

    selected = set(reasons)

    if max_frames > 0 and len(selected) > max_frames:
        mandatory = {
            observer_id
            for observer_id in selected
            if f"reference_marker_{ref_marker_id}"
            in reasons[observer_id]
        }

        optional = sorted(
            selected - mandatory,
            key=lambda observer_id: (
                len(reasons[observer_id]),
                sum(
                    marker_scores_by_frame[observer_id].values()
                ),
                -frame_number(observer_id),
            ),
            reverse=True,
        )

        remaining = max(
            0,
            max_frames - len(mandatory),
        )

        selected = mandatory | set(
            optional[:remaining]
        )

    selected_rows = [
        row
        for row in rows
        if row.get("observer_id") in selected
    ]

    report = []

    for observer_id in sorted(
        selected,
        key=frame_number,
    ):
        scores = marker_scores_by_frame[observer_id]

        report.append(
            {
                "observer_id": observer_id,
                "frame_number": frame_number(observer_id),
                "marker_ids": ";".join(
                    str(marker_id)
                    for marker_id in sorted(scores)
                ),
                "marker_count": len(scores),
                "frame_score": sum(scores.values()),
                "reference_marker_seen": (
                    ref_marker_id in scores
                ),
                "selection_reasons": ";".join(
                    sorted(reasons[observer_id])
                ),
            }
        )

    return selected_rows, report
