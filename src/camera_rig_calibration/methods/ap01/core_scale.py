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



from .core_geometry import (
    T_from_observation,
    invT,
    legacy_detection_quality,
    marker_area_from_corners,
    observation_quality,
)
from .core_io import frame_number, is_success, safe_float
def prepare_observations(
    static_rows: list[dict[str, str]],
    moving_rows: list[dict[str, str]],
    static_size: tuple[int, int],
    moving_size: tuple[int, int],
    *,
    contract: AP01MethodContract | None = None,
) -> tuple[list[dict], list[dict]]:
    contract = contract or resolve_ap01_method_contract()

    def prepare_quality(
        row: dict[str, str], image_size: tuple[int, int]
    ) -> tuple[float, dict[str, float | str]]:
        if contract.quality_model == "legacy_area_over_distance_squared_center_v1":
            return legacy_detection_quality(
                row,
                float(contract.quality_image_width_px or 1280),
                float(contract.quality_image_height_px or 720),
            )
        if contract.quality_model == "observation_quality_v2_selection_score":
            score = safe_float(
                row,
                "selection_score",
                observation_quality(row, *image_size),
            )
            return score, {
                "quality_model": contract.quality_model,
                "selection_score": score,
            }
        raise ValueError(f"Unknown AP01 quality model: {contract.quality_model}")

    prepared_static = []
    for row in static_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_camera"] = row["camera_name"]
        item["_quality"], item["_quality_components"] = prepare_quality(
            row, static_size
        )
        item["_area_px2"] = marker_area_from_corners(row)
        item["_distance_m"] = safe_float(row, "distance_m", 99.0)
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_static.append(item)

    prepared_moving = []
    for row in moving_rows:
        if not is_success(row):
            continue
        item = dict(row)
        item["_marker"] = int(float(row["marker_id"]))
        item["_frame"] = frame_number(row)
        item["_quality"], item["_quality_components"] = prepare_quality(
            row, moving_size
        )
        item["_area_px2"] = marker_area_from_corners(row)
        item["_distance_m"] = safe_float(row, "distance_m", 99.0)
        item["_T_cam_marker"] = T_from_observation(row)
        prepared_moving.append(item)

    return prepared_static, prepared_moving


def robust_scale(
    moving_rows: list[dict],
    colmap_poses: dict[int, np.ndarray],
    maximum_observations_per_marker: int | None = None,
    contract: AP01MethodContract | None = None,
) -> tuple[float, dict, list[dict]]:
    contract = contract or resolve_ap01_method_contract(
        scale_top_per_marker=maximum_observations_per_marker,
    )
    by_marker = defaultdict(list)
    rejected_observations: Counter[str] = Counter()
    for row in moving_rows:
        if contract.scale_pnp_success_only and not is_success(row):
            rejected_observations["pnp_unsuccessful"] += 1
            continue
        if contract.scale_registered_frames_only and row["_frame"] not in colmap_poses:
            rejected_observations["unregistered_frame"] += 1
            continue
        if (
            contract.scale_minimum_marker_area_px2 is not None
            and float(row["_area_px2"])
            < contract.scale_minimum_marker_area_px2
        ):
            rejected_observations["marker_area_below_minimum"] += 1
            continue
        if (
            contract.scale_maximum_marker_distance_m is not None
            and float(row["_distance_m"])
            > contract.scale_maximum_marker_distance_m
        ):
            rejected_observations["marker_distance_above_maximum"] += 1
            continue
        if contract.scale_maximum_center_norm is not None:
            width = float(contract.quality_image_width_px or 1280)
            height = float(contract.quality_image_height_px or 720)
            center_u = safe_float(row, "center_u")
            center_v = safe_float(row, "center_v")
            center_norm = math.hypot(
                center_u - width / 2.0,
                center_v - height / 2.0,
            ) / math.hypot(width / 2.0, height / 2.0)
            if center_norm > contract.scale_maximum_center_norm:
                rejected_observations["marker_center_norm_above_maximum"] += 1
                continue
        by_marker[row["_marker"]].append(row)

    registered_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }
    for marker, rows in by_marker.items():
        if contract.scale_observation_construction_policy == (
            "quality_ranked_per_marker_before_pairing_v1"
        ):
            selected = sorted(
                rows,
                key=lambda row: (
                    -float(row["_quality"]),
                    int(row["_frame"]),
                ),
            )
        elif contract.scale_observation_construction_policy == (
            "legacy_registered_quality_filters_then_all_pairs_v1"
        ):
            selected = list(rows)
        else:
            raise ValueError(
                "Unknown AP01 scale observation policy: "
                f"{contract.scale_observation_construction_policy}"
            )
        if contract.scale_observation_limit_per_marker is not None:
            truncated = max(
                0,
                len(selected) - contract.scale_observation_limit_per_marker,
            )
            rejected_observations["per_marker_limit"] += truncated
            selected = selected[: contract.scale_observation_limit_per_marker]
        by_marker[marker] = selected
    selected_counts = {
        int(marker): len(rows) for marker, rows in sorted(by_marker.items())
    }

    pairs = []
    if contract.scale_sample_multiplicity_policy != (
        "all_within_marker_unordered_frame_pairs"
    ):
        raise ValueError(
            "Unknown AP01 scale sample multiplicity policy: "
            f"{contract.scale_sample_multiplicity_policy}"
        )
    for marker, rows in by_marker.items():
        rows = sorted(rows, key=lambda r: r["_frame"])
        for first, second in combinations(rows, 2):
            gap = abs(first["_frame"] - second["_frame"])
            if (
                gap < contract.scale_frame_gap_minimum
                or gap > contract.scale_frame_gap_maximum
            ):
                continue

            if contract.scale_pnp_quantity_policy != (
                "relative_camera_translation_norm_from_T_cam_marker_v1"
            ):
                raise ValueError(
                    "Unknown AP01 scale PnP quantity policy: "
                    f"{contract.scale_pnp_quantity_policy}"
                )
            T_metric = first["_T_cam_marker"] @ invT(second["_T_cam_marker"])
            metric_distance = float(np.linalg.norm(T_metric[:3, 3]))
            if not (
                contract.scale_metric_translation_minimum_m
                <= metric_distance
                <= contract.scale_metric_translation_maximum_m
            ):
                continue

            T_colmap = colmap_poses[first["_frame"]] @ invT(colmap_poses[second["_frame"]])
            colmap_distance = float(np.linalg.norm(T_colmap[:3, 3]))
            if contract.scale_colmap_translation_rejection_policy == "less_than":
                colmap_rejected = (
                    colmap_distance
                    < contract.scale_colmap_translation_minimum_units
                )
            elif contract.scale_colmap_translation_rejection_policy == (
                "less_than_or_equal"
            ):
                colmap_rejected = (
                    colmap_distance
                    <= contract.scale_colmap_translation_minimum_units
                )
            else:
                raise ValueError(
                    "Unknown COLMAP scale displacement rejection policy: "
                    f"{contract.scale_colmap_translation_rejection_policy}"
                )
            if colmap_rejected:
                continue

            ratio = metric_distance / colmap_distance
            if not math.isfinite(ratio) or ratio <= 0:
                continue

            if contract.scale_pair_quality_policy == "sqrt_marker_area_product":
                pair_quality = math.sqrt(
                    float(first["_area_px2"]) * float(second["_area_px2"])
                )
            elif contract.scale_pair_quality_policy == (
                "sqrt_observation_quality_product"
            ):
                pair_quality = math.sqrt(
                    float(first["_quality"]) * float(second["_quality"])
                )
            else:
                raise ValueError(
                    "Unknown AP01 scale pair-quality policy: "
                    f"{contract.scale_pair_quality_policy}"
                )
            pairs.append({
                "marker_id": marker,
                "frame_i": first["_frame"],
                "frame_j": second["_frame"],
                "frame_gap": gap,
                "metric_translation_m": metric_distance,
                "colmap_translation_units": colmap_distance,
                "scale_m_per_colmap_unit": ratio,
                "quality": pair_quality,
            })

    if len(pairs) < contract.scale_minimum_pair_count:
        raise RuntimeError(f"Too few AP01 metric-scale pairs: {len(pairs)}")

    values = np.asarray([row["scale_m_per_colmap_unit"] for row in pairs], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = contract.scale_mad_sigma_factor * mad
    threshold = contract.scale_mad_multiplier * sigma
    if contract.scale_relative_deviation_floor_fraction is not None:
        threshold = max(
            threshold,
            contract.scale_relative_deviation_floor_fraction * median,
        )
    if (
        contract.scale_aggregation_policy
        == "legacy_median_three_sigma_mad_v1"
        and mad <= 1e-12
    ):
        kept = pairs
    elif contract.scale_aggregation_policy in {
        "legacy_median_three_sigma_mad_v1",
        "wizard_median_mad_relative_floor_v1",
    }:
        kept = [
            row
            for row in pairs
            if abs(row["scale_m_per_colmap_unit"] - median) <= threshold
        ]
    else:
        raise ValueError(
            "Unknown AP01 scale aggregation policy: "
            f"{contract.scale_aggregation_policy}"
        )
    fallback_threshold: float | int
    if contract.scale_aggregation_policy == "legacy_median_three_sigma_mad_v1":
        fallback_threshold = max(
            contract.scale_fallback_minimum_count,
            contract.scale_fallback_fraction * len(pairs),
        )
    else:
        fallback_threshold = max(
            contract.scale_fallback_minimum_count,
            int(contract.scale_fallback_fraction * len(pairs)),
        )
    fallback_used = len(kept) < fallback_threshold
    if fallback_used:
        kept = pairs
    kept_values = np.asarray([row["scale_m_per_colmap_unit"] for row in kept], dtype=np.float64)
    if contract.scale_final_statistic != "median":
        raise ValueError(
            "Unknown AP01 final scale statistic: "
            f"{contract.scale_final_statistic}"
        )
    scale = float(np.median(kept_values))
    kept_ids = {id(row) for row in kept}
    for row in pairs:
        row["used_for_scale"] = id(row) in kept_ids

    marker_pair_counts = Counter(
        int(row["marker_id"]) for row in pairs
    )
    marker_inlier_counts = Counter(
        int(row["marker_id"]) for row in kept
    )
    stats = {
        "scale_m_per_colmap_unit": scale,
        "raw_pairs": len(pairs),
        "used_pairs": len(kept),
        "raw_median": median,
        "raw_mad": mad,
        "used_mean": float(np.mean(kept_values)),
        "used_std": float(np.std(kept_values)),
        "used_relative_std": float(np.std(kept_values) / scale),
        "markers_with_registered_observations": sorted(by_marker),
        "maximum_observations_per_marker": (
            contract.scale_observation_limit_per_marker
        ),
        "rejected_observations_by_reason": dict(
            sorted(rejected_observations.items())
        ),
        "scale_contract": contract.fingerprint_payload(),
        "scale_contract_sha256": contract.scientific_fingerprint(),
        "aggregation_threshold": threshold,
        "aggregation_fallback_threshold": fallback_threshold,
        "aggregation_fallback_used": fallback_used,
        "registered_observations_per_marker": registered_counts,
        "selected_observations_per_marker": selected_counts,
        "candidate_pairs_per_marker": dict(sorted(marker_pair_counts.items())),
        "inlier_pairs_per_marker": dict(sorted(marker_inlier_counts.items())),
        "outlier_pairs_per_marker": {
            marker: marker_pair_counts[marker] - marker_inlier_counts[marker]
            for marker in sorted(marker_pair_counts)
        },
    }
    return scale, stats, pairs


