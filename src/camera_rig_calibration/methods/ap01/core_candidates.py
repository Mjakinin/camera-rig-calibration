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



from .core_geometry import invT, make_T
def R_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1e-15, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        q = np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    else:
        s = math.sqrt(max(1e-15, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        q = np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ])
    q /= max(float(np.linalg.norm(q)), 1e-15)
    return q


def quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def weighted_rotation_mean(rotations: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    quaternions = [R_to_quat(R) for R in rotations]
    reference = quaternions[0]
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion, weight in zip(quaternions, weights):
        if float(np.dot(quaternion, reference)) < 0:
            quaternion = -quaternion
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    _, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, -1]
    if quaternion[0] < 0:
        quaternion = -quaternion
    return quat_to_R(quaternion)


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    value = max(-1.0, min(1.0, float((np.trace(relative) - 1.0) / 2.0)))
    return math.degrees(math.acos(value))


def aggregate_candidates(candidates: list[dict], translation_floor: float = 0.20, rotation_floor: float = 5.0) -> tuple[np.ndarray, dict]:
    if not candidates:
        raise RuntimeError("No AP01 transform candidates")

    translations = np.asarray([row["T"][:3, 3] for row in candidates], dtype=np.float64)
    weights = np.asarray([max(float(row["quality"]), 1e-12) for row in candidates], dtype=np.float64)
    weights /= weights.sum()

    initial_translation = np.median(translations, axis=0)
    initial_rotation = weighted_rotation_mean([row["T"][:3, :3] for row in candidates], weights)

    translation_deviation = np.linalg.norm(translations - initial_translation[None, :], axis=1)
    rotation_deviation = np.asarray(
        [rotation_difference_deg(initial_rotation, row["T"][:3, :3]) for row in candidates],
        dtype=np.float64,
    )

    t_median = float(np.median(translation_deviation))
    r_median = float(np.median(rotation_deviation))
    t_mad = 1.4826 * float(np.median(np.abs(translation_deviation - t_median)))
    r_mad = 1.4826 * float(np.median(np.abs(rotation_deviation - r_median)))

    t_threshold = max(translation_floor, t_median + 3.0 * t_mad)
    r_threshold = max(rotation_floor, r_median + 3.0 * r_mad)

    robust_inlier_indices = [
        index
        for index, (t_dev, r_dev) in enumerate(zip(translation_deviation, rotation_deviation))
        if t_dev <= t_threshold and r_dev <= r_threshold
    ]
    inlier_indices = list(robust_inlier_indices)
    pose_fallback_used = False
    if len(inlier_indices) < min(3, len(candidates)):
        pose_fallback_used = True
        inlier_indices = list(np.argsort(weights)[::-1][:max(1, min(3, len(candidates)))])

    inlier_weights = weights[inlier_indices]
    inlier_weights /= inlier_weights.sum()
    translation = np.sum(translations[inlier_indices] * inlier_weights[:, None], axis=0)
    rotation = weighted_rotation_mean(
        [candidates[index]["T"][:3, :3] for index in inlier_indices],
        inlier_weights,
    )
    transform = make_T(rotation, translation)

    robust_inlier_set = set(robust_inlier_indices)
    pose_support_set = set(inlier_indices)
    final_translation_deviation = np.linalg.norm(
        translations - translation[None, :], axis=1
    )
    final_rotation_deviation = np.asarray(
        [
            rotation_difference_deg(rotation, row["T"][:3, :3])
            for row in candidates
        ],
        dtype=np.float64,
    )
    robust_translation_deviation = np.asarray([], dtype=np.float64)
    robust_rotation_deviation = np.asarray([], dtype=np.float64)
    if robust_inlier_indices:
        robust_weights = weights[robust_inlier_indices]
        robust_weights /= robust_weights.sum()
        robust_translation = np.sum(
            translations[robust_inlier_indices]
            * robust_weights[:, None],
            axis=0,
        )
        robust_rotation = weighted_rotation_mean(
            [
                candidates[index]["T"][:3, :3]
                for index in robust_inlier_indices
            ],
            robust_weights,
        )
        robust_translation_deviation = np.linalg.norm(
            translations[robust_inlier_indices]
            - robust_translation[None, :],
            axis=1,
        )
        robust_rotation_deviation = np.asarray(
            [
                rotation_difference_deg(
                    robust_rotation,
                    candidates[index]["T"][:3, :3],
                )
                for index in robust_inlier_indices
            ],
            dtype=np.float64,
        )
    for index, row in enumerate(candidates):
        row["translation_deviation_m"] = float(
            final_translation_deviation[index]
        )
        row["rotation_deviation_deg"] = float(
            final_rotation_deviation[index]
        )
        row["inlier"] = index in robust_inlier_set
        row["pose_support"] = index in pose_support_set

    stats = {
        "candidates": len(candidates),
        "inliers": len(robust_inlier_indices),
        "robust_inliers": len(robust_inlier_indices),
        "pose_support_count": len(inlier_indices),
        "pose_fallback_used": pose_fallback_used,
        "inlier_ratio": (
            len(robust_inlier_indices) / len(candidates)
            if candidates
            else 0.0
        ),
        "maximum_inlier_translation_dispersion_m": (
            float(np.max(robust_translation_deviation))
            if robust_inlier_indices
            else None
        ),
        "maximum_inlier_rotation_dispersion_deg": (
            float(np.max(robust_rotation_deviation))
            if robust_inlier_indices
            else None
        ),
        "translation_deviation_median_m": t_median,
        "rotation_deviation_median_deg": r_median,
        "translation_deviation_p90_m": (
            float(np.percentile(robust_translation_deviation, 90))
            if robust_inlier_indices
            else None
        ),
        "rotation_deviation_p90_deg": (
            float(np.percentile(robust_rotation_deviation, 90))
            if robust_inlier_indices
            else None
        ),
        "translation_robust_rms_m": (
            float(np.sqrt(np.mean(robust_translation_deviation**2)))
            if robust_inlier_indices
            else None
        ),
        "rotation_robust_rms_deg": (
            float(np.sqrt(np.mean(robust_rotation_deviation**2)))
            if robust_inlier_indices
            else None
        ),
        "translation_threshold_m": t_threshold,
        "rotation_threshold_deg": r_threshold,
    }
    return transform, stats


def aggregate_direct_marker_estimates(
    candidates: list[dict],
) -> tuple[np.ndarray, dict]:
    """Aggregate one GT-free relation per independent shared marker."""

    transform, stats = aggregate_candidates(
        candidates, translation_floor=0.12, rotation_floor=4.0
    )
    return transform, {
        **stats,
        "aggregate_type": (
            "quality_filtered_weighted_mean_of_mad_inliers_"
            "no_gt_selection"
        ),
        "raw_candidate_count": len(candidates),
        "independent_marker_count": len(
            {
                int(item["root_marker"])
                for item in candidates
                if item.get("root_marker") is not None
            }
        ),
        "ground_truth_used": False,
    }


def aggregate_relay_marker_chains(
    candidates: list[dict],
) -> tuple[np.ndarray, dict, list[dict]]:
    """Aggregate correlated relay samples in two GT-free hierarchy levels.

    Samples sharing a root/target marker pair form one correlated chain.
    Stage one estimates each chain.  Stage two robustly combines only those
    independent chain estimates, so thousands of Cartesian frame pairs can no
    longer masquerade as thousands of independent observations.
    """

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[
            (int(item["root_marker"]), int(item["target_marker"]))
        ].append(item)
    chain_candidates: list[dict] = []
    chain_reports: list[dict] = []
    for (root_marker, target_marker), group in sorted(grouped.items()):
        pose, stats = aggregate_candidates(
            group, translation_floor=0.30, rotation_floor=7.0
        )
        inlier_quality = [
            max(float(item.get("quality", 0.0)), 1e-12)
            for item in group
            if item.get("inlier")
        ]
        quality = (
            float(np.mean(inlier_quality))
            if inlier_quality
            else max(float(item.get("quality", 0.0)) for item in group)
        )
        chain_id = f"{root_marker}->{target_marker}"
        chain_candidate = {
            "mode": "relay_chain",
            "chain_id": chain_id,
            "root_marker": root_marker,
            "target_marker": target_marker,
            "quality": quality,
            "T": pose,
            "raw_candidate_count": len(group),
        }
        chain_candidates.append(chain_candidate)
        chain_reports.append(
            {
                "chain_id": chain_id,
                "root_marker": root_marker,
                "target_marker": target_marker,
                "raw_candidate_count": len(group),
                "robust_inlier_count": stats.get("robust_inliers", 0),
                "quality_weight": quality,
                "translation_dispersion_m": stats.get(
                    "maximum_inlier_translation_dispersion_m"
                ),
                "rotation_dispersion_deg": stats.get(
                    "maximum_inlier_rotation_dispersion_deg"
                ),
                "translation_robust_rms_m": stats.get(
                    "translation_robust_rms_m"
                ),
                "rotation_robust_rms_deg": stats.get(
                    "rotation_robust_rms_deg"
                ),
                "estimate": pose.tolist(),
            }
        )
    if not chain_candidates:
        raise RuntimeError("No AP01 relay marker-chain estimate")
    transform, final_stats = aggregate_candidates(
        chain_candidates, translation_floor=0.30, rotation_floor=7.0
    )
    final_stats.update(
        {
            "aggregate_type": (
                "hierarchical_weighted_mean_of_mad_inliers_"
                "no_gt_selection"
            ),
            "raw_candidate_count": len(candidates),
            "chain_count": len(chain_candidates),
            "independent_marker_chain_count": len(chain_candidates),
            "effective_support": int(
                final_stats.get("robust_inliers", 0)
            ),
            "chain_reports": chain_reports,
            "ground_truth_used": False,
        }
    )
    return transform, final_stats, chain_candidates


def best_static_by_camera_marker(rows: list[dict]) -> dict[tuple[str, int], dict]:
    result = {}
    for row in rows:
        key = (row["_camera"], row["_marker"])
        if key not in result or row["_quality"] > result[key]["_quality"]:
            result[key] = row
    return result


def moving_by_marker(
    rows: list[dict],
    registered_frames: set[int],
    top_per_marker: int | None = None,
) -> dict[int, list[dict]]:
    """Rank registered moving observations and apply the AP01 relay cap."""
    grouped = defaultdict(list)
    for row in rows:
        if row["_frame"] in registered_frames:
            grouped[row["_marker"]].append(row)
    for marker in grouped:
        ranked = sorted(
            grouped[marker],
            key=lambda row: (
                -float(row["_quality"]),
                int(row["_frame"]),
            ),
        )
        if top_per_marker is not None:
            ranked = ranked[:top_per_marker]
        grouped[marker] = sorted(
            ranked, key=lambda row: int(row["_frame"])
        )
    return grouped


def moving_by_contract(
    rows: list[dict],
    registered_frames: set[int],
    contract: AP01MethodContract,
) -> dict[int, list[dict]]:
    """Select moving supports using the resolved AP01 scientific contract."""

    if contract.moving_support_policy == (
        "quality_ranked_registered_then_frame_ascending"
    ):
        return moving_by_marker(
            rows,
            registered_frames,
            top_per_marker=contract.relay_input_limit,
        )
    if contract.moving_support_policy == (
        "best_quality_per_frame_marker_first_tie_registered_only_frame_ascending"
    ):
        best: dict[tuple[int, int], dict] = {}
        for row in rows:
            frame = int(row["_frame"])
            marker = int(row["_marker"])
            if frame not in registered_frames:
                continue
            key = (frame, marker)
            if key not in best or float(row["_quality"]) > float(
                best[key]["_quality"]
            ):
                best[key] = row
        grouped: defaultdict[int, list[dict]] = defaultdict(list)
        for (_, marker), row in best.items():
            grouped[marker].append(row)
        for marker in grouped:
            grouped[marker].sort(key=lambda row: int(row["_frame"]))
        return dict(grouped)
    raise ValueError(
        f"Unknown AP01 moving-support policy: {contract.moving_support_policy}"
    )


def direct_candidates(
    root: str,
    target: str,
    static_best: dict[tuple[str, int], dict],
) -> list[dict]:
    root_markers = {marker for camera, marker in static_best if camera == root}
    target_markers = {marker for camera, marker in static_best if camera == target}
    result = []
    for marker in sorted(root_markers & target_markers):
        root_row = static_best[(root, marker)]
        target_row = static_best[(target, marker)]
        transform = root_row["_T_cam_marker"] @ invT(target_row["_T_cam_marker"])
        result.append({
            "mode": "direct",
            "root_camera": root,
            "target_camera": target,
            "root_marker": marker,
            "target_marker": marker,
            "root_frame": "",
            "target_frame": "",
            "quality": math.sqrt(root_row["_quality"] * target_row["_quality"]),
            "root_area_px2": float(root_row.get("_area_px2", float("nan"))),
            "target_area_px2": float(target_row.get("_area_px2", float("nan"))),
            "root_distance_m": float(root_row.get("_distance_m", float("nan"))),
            "target_distance_m": float(target_row.get("_distance_m", float("nan"))),
            "root_support_key": root_row.get("observation_key"),
            "target_support_key": target_row.get("observation_key"),
            "T": transform,
        })
    return result


def relay_candidates(
    root: str,
    target: str,
    static_best: dict[tuple[str, int], dict],
    moving_by_marker: dict[int, list[dict]],
    colmap_poses: dict[int, np.ndarray],
    scale: float,
) -> list[dict]:
    root_markers = sorted(
        marker
        for camera, marker in static_best
        if camera == root and marker in moving_by_marker
    )
    target_markers = sorted(
        marker
        for camera, marker in static_best
        if camera == target and marker in moving_by_marker
    )

    result = []
    for root_marker in root_markers:
        root_static = static_best[(root, root_marker)]
        T_root_marker = root_static["_T_cam_marker"]
        for target_marker in target_markers:
            target_static = static_best[(target, target_marker)]
            T_target_marker = target_static["_T_cam_marker"]

            for root_moving in moving_by_marker[root_marker]:
                frame_i = root_moving["_frame"]
                T_root_moving_i = T_root_marker @ invT(root_moving["_T_cam_marker"])

                for target_moving in moving_by_marker[target_marker]:
                    frame_j = target_moving["_frame"]
                    if frame_i == frame_j and root_marker == target_marker:
                        continue

                    T_target_moving_j = T_target_marker @ invT(target_moving["_T_cam_marker"])
                    T_moving_i_moving_j = colmap_poses[frame_i] @ invT(colmap_poses[frame_j])
                    T_moving_i_moving_j = T_moving_i_moving_j.copy()
                    T_moving_i_moving_j[:3, 3] *= scale

                    transform = (
                        T_root_moving_i
                        @ T_moving_i_moving_j
                        @ invT(T_target_moving_j)
                    )
                    quality = (
                        root_static["_quality"]
                        * target_static["_quality"]
                        * root_moving["_quality"]
                        * target_moving["_quality"]
                    ) ** 0.25
                    result.append({
                        "mode": "relay",
                        "root_camera": root,
                        "target_camera": target,
                        "root_marker": root_marker,
                        "target_marker": target_marker,
                        "root_frame": frame_i,
                        "target_frame": frame_j,
                        "quality": quality,
                        "support_keys": [
                            root_static.get("observation_key"),
                            target_static.get("observation_key"),
                            root_moving.get("observation_key"),
                            target_moving.get("observation_key"),
                        ],
                        "T": transform,
                    })
    return result


