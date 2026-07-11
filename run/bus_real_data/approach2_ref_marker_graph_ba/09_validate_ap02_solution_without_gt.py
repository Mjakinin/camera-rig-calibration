#!/usr/bin/env python3
"""Validate the AP02 solution without using simulation ground truth.

This gate is intended for real-data acceptance. It checks camera coverage,
reprojection quality, cheirality, graph connectivity and a configurable rig
size envelope. It deliberately does not read the Gazebo world or GT files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(
    os.environ.get(
        "AP02_ROOT",
        "results/bus_real_data/02_ref_marker_graph_ba",
    )
)
BA_ROOT = ROOT / "07_graph_ba" / "with_moving"
OBS_CSV = ROOT / "02_aruco_observations" / "ap02_all_aruco_observations.csv"
OUT_ROOT = ROOT / "08_final_results"
EXPECTED_CAMERAS = {"cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]


def pose_from_row(row: dict[str, str]) -> np.ndarray:
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])])
    translation = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rodrigues(rvec)
    transform[:3, 3] = translation
    return transform


def inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def marker_points(length: float) -> np.ndarray:
    half = length / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def connectivity_ok(observations: list[dict], cameras: set[str], marker_ids: set[int], ref_marker: int) -> tuple[bool, list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        if str(row.get("pnp_success", "")).lower() not in {"true", "1", "yes"}:
            continue
        observer = row.get("observer_id", "")
        try:
            marker_id = int(float(row.get("marker_id", "nan")))
        except ValueError:
            continue
        if marker_id not in marker_ids:
            continue
        observer_node = f"observer:{observer}"
        marker_node = f"marker:{marker_id}"
        graph[observer_node].add(marker_node)
        graph[marker_node].add(observer_node)

    start = f"marker:{ref_marker}"
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    disconnected = sorted(camera for camera in cameras if f"observer:{camera}" not in visited)
    return not disconnected, disconnected


def cheirality_fraction(observations: list[dict], camera_poses: dict[str, np.ndarray], marker_poses: dict[int, np.ndarray]) -> float:
    positive = 0
    total = 0
    for row in observations:
        observer = row.get("observer_id", "")
        if observer not in camera_poses:
            continue
        try:
            marker_id = int(float(row.get("marker_id", "nan")))
            length = float(row.get("marker_length_m", 0.170) or 0.170)
        except ValueError:
            continue
        if marker_id not in marker_poses:
            continue
        observer_from_ref = inverse(camera_poses[observer])
        for point in marker_points(length):
            point_h = np.array([point[0], point[1], point[2], 1.0])
            depth = float((observer_from_ref @ marker_poses[marker_id] @ point_h)[2])
            total += 1
            if depth > 1e-6:
                positive += 1
    return positive / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-median-reproj-px", type=float, default=2.0)
    parser.add_argument("--min-positive-depth-fraction", type=float, default=0.99)
    parser.add_argument(
        "--max-rig-diameter-m",
        type=float,
        default=float(os.environ.get("AP02_MAX_RIG_DIAMETER_M", "12.0")),
        help=(
            "Maximum plausible distance between static cameras. The bus "
            "benchmark spans about ten metres, so the default is 12 m. "
            "Override with AP02_MAX_RIG_DIAMETER_M or this argument for "
            "other physical rigs."
        ),
    )
    parser.add_argument("--min-pair-baseline-m", type=float, default=0.02)
    parser.add_argument("--ref-marker-id", type=int, default=14)
    args = parser.parse_args()

    static_rows = read_csv(BA_ROOT / "optimized_static_camera_poses_ref_marker.csv")
    marker_rows = read_csv(BA_ROOT / "optimized_marker_poses_ref_marker.csv")
    error_rows = read_csv(BA_ROOT / "reprojection_errors_by_observation.csv")
    observations = read_csv(OBS_CSV)

    camera_poses = {row["entity_id"]: pose_from_row(row) for row in static_rows}
    marker_poses = {int(row["entity_id"]): pose_from_row(row) for row in marker_rows}
    cameras = set(camera_poses)

    reprojection = [float(row["median_reproj_px"]) for row in error_rows if row.get("median_reproj_px", "")]
    median_reprojection = float(np.median(reprojection)) if reprojection else math.inf

    pair_distances = []
    camera_names = sorted(camera_poses)
    for index, first in enumerate(camera_names):
        for second in camera_names[index + 1 :]:
            pair_distances.append(
                float(np.linalg.norm(camera_poses[first][:3, 3] - camera_poses[second][:3, 3]))
            )
    max_diameter = max(pair_distances, default=math.inf)
    min_baseline = min(pair_distances, default=0.0)

    connected, disconnected = connectivity_ok(
        observations, cameras, set(marker_poses), args.ref_marker_id
    )
    positive_depth_fraction = cheirality_fraction(
        observations, camera_poses, marker_poses
    )

    checks = {
        "complete_static_camera_coverage": cameras == EXPECTED_CAMERAS,
        "connected_to_reference_marker": connected,
        "median_reprojection_within_limit": median_reprojection <= args.max_median_reproj_px,
        "positive_depth_fraction_within_limit": positive_depth_fraction >= args.min_positive_depth_fraction,
        "rig_diameter_within_limit": max_diameter <= args.max_rig_diameter_m,
        "pair_baseline_above_limit": min_baseline >= args.min_pair_baseline_m,
    }
    accepted = all(checks.values())

    result = {
        "accepted_without_ground_truth": accepted,
        "checks": checks,
        "metrics": {
            "static_cameras_found": sorted(cameras),
            "static_cameras_expected": sorted(EXPECTED_CAMERAS),
            "disconnected_cameras": disconnected,
            "median_reprojection_px": median_reprojection,
            "positive_depth_fraction": positive_depth_fraction,
            "max_rig_diameter_m": max_diameter,
            "min_pair_baseline_m": min_baseline,
        },
        "thresholds": {
            "max_median_reproj_px": args.max_median_reproj_px,
            "min_positive_depth_fraction": args.min_positive_depth_fraction,
            "max_rig_diameter_m": args.max_rig_diameter_m,
            "min_pair_baseline_m": args.min_pair_baseline_m,
        },
        "ground_truth_used": False,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "AP02_GT_FREE_VALIDITY_GATE.json"
    text_path = OUT_ROOT / "AP02_GT_FREE_VALIDITY_GATE.txt"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    text_lines = [
        "AP02 GT-free validity gate",
        "===========================",
        "",
        f"accepted_without_ground_truth: {accepted}",
        f"median_reprojection_px: {median_reprojection:.6f}",
        f"positive_depth_fraction: {positive_depth_fraction:.6f}",
        f"max_rig_diameter_m: {max_diameter:.6f}",
        f"min_pair_baseline_m: {min_baseline:.6f}",
        f"disconnected_cameras: {disconnected}",
        "",
        "Checks:",
    ]
    text_lines.extend(f"- {name}: {value}" for name, value in checks.items())
    text_path.write_text("\n".join(text_lines) + "\n")

    print(text_path.read_text())
    if not accepted:
        raise SystemExit("AP02 rejected by GT-free validity gate")


if __name__ == "__main__":
    main()
