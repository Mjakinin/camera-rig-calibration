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



from .core_bindings import AP01CoreBindings
from .core_candidates import (
    aggregate_candidates,
    best_static_by_camera_marker,
    direct_candidates,
    moving_by_marker,
    relay_candidates,
)
from .core_colmap import run_colmap
from .core_geometry import parse_colmap_poses
from .core_io import (
    load_camera_info,
    parse_args,
    read_csv,
    write_csv,
    write_status,
)
from .core_scale import prepare_observations, robust_scale


def R_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def rpy_deg_to_R(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> np.ndarray:
    """Reconstruct the ZYX rotation used by the Legacy aggregate CSV."""

    roll, pitch, yaw = (
        math.radians(float(value))
        for value in (roll_deg, pitch_deg, yaw_deg)
    )
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def serialize_final_pose(
    transform: np.ndarray, contract: AP01MethodContract
) -> np.ndarray:
    """Apply the contract-scoped final numeric serialization adapter."""

    if contract.final_pose_serialization_policy == "native_full_precision_v1":
        return np.asarray(transform, dtype=np.float64).copy()
    if (
        contract.final_pose_serialization_policy
        != "legacy_aggregate_csv_rpy_roundtrip_v1"
    ):
        raise ValueError(
            "Unknown AP01 final-pose serialization policy: "
            f"{contract.final_pose_serialization_policy}"
        )
    places = contract.final_pose_serialization_decimal_places
    if places is None:
        raise ValueError("Legacy AP01 final-pose serialization needs precision")
    result = np.eye(4, dtype=np.float64)
    rpy = R_to_rpy_deg(np.asarray(transform, dtype=np.float64)[:3, :3])
    serialized_rpy = tuple(float(f"{value:.{places}f}") for value in rpy)
    result[:3, :3] = rpy_deg_to_R(*serialized_rpy)
    result[:3, 3] = [
        float(f"{float(value):.{places}f}")
        for value in np.asarray(transform, dtype=np.float64)[:3, 3]
    ]
    return result


def pose_row(camera: str, T: np.ndarray, source: str) -> dict:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    roll, pitch, yaw = R_to_rpy_deg(T[:3, :3])
    return {
        "entity_type": "static_camera",
        "entity_id": camera,
        "source": source,
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "rvec_x": float(rvec[0, 0]),
        "rvec_y": float(rvec[1, 0]),
        "rvec_z": float(rvec[2, 0]),
    }


def serializable_candidate(row: dict) -> dict:
    result = {key: value for key, value in row.items() if key != "T"}
    T = row["T"]
    result.update({
        "x_m": float(T[0, 3]),
        "y_m": float(T[1, 3]),
        "z_m": float(T[2, 3]),
    })
    return result


def pairwise_rows(poses: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for first, second in combinations(CAMERAS, 2):
        if first not in poses or second not in poses:
            continue
        distance = float(np.linalg.norm(poses[first][:3, 3] - poses[second][:3, 3]))
        rows.append({
            "camera_a": first,
            "camera_b": second,
            "distance_m": distance,
        })
    return rows


def main() -> None:
    global CAMERAS, ROOT_CAMERA
    bindings = AP01CoreBindings.current()
    parse_args = bindings.parse_args
    load_camera_info = bindings.load_camera_info
    run_colmap = bindings.run_colmap
    parse_colmap_poses = bindings.parse_colmap_poses
    read_csv = bindings.read_csv
    prepare_observations = bindings.prepare_observations
    robust_scale = bindings.robust_scale
    write_csv = bindings.write_csv
    best_static_by_camera_marker = bindings.best_static_by_camera_marker
    moving_by_marker = bindings.moving_by_marker
    direct_candidates = bindings.direct_candidates
    relay_candidates = bindings.relay_candidates
    aggregate_candidates = bindings.aggregate_candidates
    write_status = bindings.write_status
    args = parse_args()
    CAMERAS = [value.strip() for value in args.cameras.split(",") if value.strip()]
    ROOT_CAMERA = args.root_camera.strip()
    if not CAMERAS:
        raise RuntimeError("--cameras must contain at least one camera ID")
    if ROOT_CAMERA not in CAMERAS:
        raise RuntimeError(f"Root camera '{ROOT_CAMERA}' is not in --cameras")
    started = time.time()

    dataset = Path(args.dataset).resolve()
    observations_root = Path(args.observations_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        moving_dir = dataset / "raw_images" / "moving"
        moving_info_path = (
            dataset
            / "raw_images"
            / "camera_info"
            / f"{args.moving_camera_id}.json"
        )
        static_csv = observations_root / "shared_static_aruco_observations.csv"
        moving_csv = observations_root / "shared_moving_aruco_observations.csv"

        if not moving_dir.is_dir():
            raise RuntimeError(f"Missing moving images: {moving_dir}")
        if not moving_info_path.is_file():
            raise RuntimeError(f"Missing moving camera info: {moving_info_path}")

        moving_info = load_camera_info(moving_info_path)
        static_info = load_camera_info(
            dataset / "raw_images" / "camera_info" / f"{ROOT_CAMERA}.json"
        )

        images_txt = run_colmap(
            image_dir=moving_dir,
            camera_info=moving_info,
            out_dir=out / "01_moving_colmap",
            matcher=args.matcher,
            use_gpu=args.use_gpu,
            max_image_size=args.max_image_size,
            max_features=args.max_features,
            sequential_overlap=args.sequential_overlap,
            loop_detection=args.loop_detection,
            mapper_min_matches=args.mapper_min_matches,
            colmap_executable=args.colmap_executable,
            reuse=args.reuse_colmap,
        )

        colmap_poses = parse_colmap_poses(images_txt)
        static_rows_raw = read_csv(static_csv)
        moving_rows_raw = read_csv(moving_csv)

        static_rows, moving_rows = prepare_observations(
            static_rows_raw,
            moving_rows_raw,
            (static_info["width"], static_info["height"]),
            (moving_info["width"], moving_info["height"]),
        )

        scale, scale_stats, scale_pairs = robust_scale(
            moving_rows,
            colmap_poses,
        )

        scale_dir = out / "02_metric_scale"
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "metric_scale.txt").write_text(f"{scale:.12g}\n")
        (scale_dir / "SCALE_DIAGNOSTICS.json").write_text(
            json.dumps(scale_stats, indent=2) + "\n"
        )
        write_csv(scale_dir / "scale_pairs.csv", scale_pairs)

        static_best = best_static_by_camera_marker(static_rows)
        moving_by_marker_rows = moving_by_marker(
            moving_rows,
            set(colmap_poses),
        )

        poses = {ROOT_CAMERA: np.eye(4, dtype=np.float64)}
        method_by_camera = {ROOT_CAMERA: "gauge_identity"}
        method_diagnostics = {}
        all_candidate_rows = []

        for target in CAMERAS:
            if target == ROOT_CAMERA:
                continue

            direct = direct_candidates(ROOT_CAMERA, target, static_best)
            relay = relay_candidates(
                ROOT_CAMERA,
                target,
                static_best,
                moving_by_marker_rows,
                colmap_poses,
                scale,
            )

            direct_transform = None
            direct_stats = None
            if direct:
                direct_transform, direct_stats = aggregate_candidates(
                    direct,
                    translation_floor=0.12,
                    rotation_floor=4.0,
                )

            relay_transform = None
            relay_stats = None
            if relay:
                relay_transform, relay_stats = aggregate_candidates(
                    relay,
                    translation_floor=0.30,
                    rotation_floor=7.0,
                )

            if len(direct) >= 2:
                selected = direct_transform
                selected_method = "direct_multimarker"
            elif relay_transform is not None:
                selected = relay_transform
                selected_method = "moving_colmap_relay"
            elif direct_transform is not None:
                selected = direct_transform
                selected_method = "direct_single_marker"
            else:
                selected = None
                selected_method = "unavailable"

            if selected is not None:
                poses[target] = selected
                method_by_camera[target] = selected_method

            method_diagnostics[target] = {
                "selected_method": selected_method,
                "direct_common_markers": sorted(
                    {int(row["root_marker"]) for row in direct}
                ),
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_candidates": len(relay),
            }

            all_candidate_rows.extend(serializable_candidate(row) for row in direct)
            all_candidate_rows.extend(serializable_candidate(row) for row in relay)

        final_dir = out / "03_static_extrinsics"
        final_dir.mkdir(parents=True, exist_ok=True)

        pose_rows = [
            pose_row(camera, transform, method_by_camera[camera])
            for camera, transform in sorted(poses.items())
        ]
        pose_fields = [
            "entity_type", "entity_id", "source",
            "x_m", "y_m", "z_m",
            "roll_deg", "pitch_deg", "yaw_deg",
            "rvec_x", "rvec_y", "rvec_z",
        ]
        generic_pose_file = (
            final_dir / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        )
        write_csv(generic_pose_file, pose_rows, pose_fields)
        if ROOT_CAMERA == "cam_edge_3":
            write_csv(
                final_dir / "AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv",
                pose_rows,
                pose_fields,
            )
        write_csv(
            final_dir / "AP01_PAIRWISE_DISTANCES.csv",
            pairwise_rows(poses),
            ["camera_a", "camera_b", "distance_m"],
        )
        write_csv(final_dir / "AP01_TRANSFORM_CANDIDATES.csv", all_candidate_rows)

        diagnostics = {
            "approach": "AP01_marker_direct_and_moving_colmap_relay",
            "root_camera": ROOT_CAMERA,
            "registered_moving_frames": len(colmap_poses),
            "input_moving_frames": len(list(moving_dir.glob("frame_*.png"))),
            "metric_scale": scale_stats,
            "static_camera_methods": method_by_camera,
            "per_target_diagnostics": method_diagnostics,
            "available_static_cameras": sorted(poses),
            "missing_static_cameras": sorted(set(CAMERAS) - set(poses)),
            "runtime_seconds": time.time() - started,
            "ground_truth_used": False,
        }
        (final_dir / "AP01_DIAGNOSTICS.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )

        expected_count = len(CAMERAS)
        status = (
            "OK_FULL"
            if len(poses) == expected_count
            else f"PARTIAL_{len(poses)}_OF_{expected_count}"
        )
        write_status(out, {
            "method": "AP01",
            "status": status,
            "success": len(poses) == expected_count,
            "available_static_cameras": sorted(poses),
            "runtime_seconds": time.time() - started,
            "pose_file": str(generic_pose_file),
            "diagnostics_file": str(final_dir / "AP01_DIAGNOSTICS.json"),
        })

        print("\nAP01 REAL-DATA RESULT")
        print("=" * 72)
        print("status:", status)
        print("registered moving frames:", len(colmap_poses))
        print("metric scale:", scale)
        print("camera methods:", method_by_camera)
        print("pose file:", generic_pose_file)

        if len(poses) < expected_count:
            raise RuntimeError(
                f"AP01 produced only {len(poses)}/{expected_count} static camera poses"
            )

    except Exception as exc:
        failure = {
            "method": "AP01",
            "status": "FAILED",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_seconds": time.time() - started,
        }
        write_status(out, failure)
        print(failure["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
