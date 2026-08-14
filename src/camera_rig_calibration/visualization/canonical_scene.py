"""Pose-only visualization for method-independent canonical 6DOF results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..anchor_export.geometry import validate_transform


@dataclass(frozen=True)
class CanonicalSceneDependencies:
    """Small late-bound bridge to the established scene I/O helpers."""

    scene_contract: str
    write_ply: Callable[[Path, list], None]
    camera_info: Callable[[Path, str], dict[str, Any]]
    intrinsics: Callable[[dict[str, Any]], tuple | None]
    frustum: Callable[[np.ndarray, tuple, float], list[list[float]]]
    topic: Callable[[str, str, str], str]
    write_json: Callable[[Path, dict[str, Any]], None]
    atomic_text: Callable[[Path, str], None]
    rviz_config: Callable[[str, list[dict[str, Any]], bool], str]
    update_result_status: Callable[[Path, str], None]


def append_canonical_variants(
    experiment_root: Path,
    variants: list[dict[str, Any]],
    read_json: Callable[[Path], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add canonical results not already represented by anchor exports."""

    existing = {
        (str(item["method"]), str(item["label"])) for item in variants
    }
    for canonical_path in sorted(
        (experiment_root / "methods").glob(
            "*/*/canonical_method_result.json"
        )
    ):
        result_root = canonical_path.parent
        result = read_json(result_root / "RESULT.json")
        payload = read_json(canonical_path)
        method = str(
            payload.get("method_id") or canonical_path.parents[1].name
        )
        label = str(result.get("label") or result_root.name)
        if (method, label) in existing or payload.get("status") != "available":
            continue
        camera_poses = [
            {**pose, "parent_frame": pose.get("reference_frame")}
            for pose in payload.get("camera_poses", [])
            if isinstance(pose, dict) and pose.get("matrix") is not None
        ]
        if camera_poses:
            variants.append(
                {
                    "method": method,
                    "label": label,
                    "path": canonical_path,
                    "payload": {
                        "parent_frame": payload.get("reference_frame"),
                        "cameras": camera_poses,
                        "canonical_native": True,
                    },
                }
            )
    return variants


def create_canonical_native_scene(
    experiment_root: Path,
    output: Path,
    variants: list[dict[str, Any]],
    dependencies: CanonicalSceneDependencies,
) -> dict[str, Any] | None:
    """Create a pose-only scene when canonical results share one frame."""

    canonical = [
        item
        for item in variants
        if item.get("payload", {}).get("canonical_native")
    ]
    frames = {
        str(item["payload"].get("parent_frame") or "")
        for item in canonical
    }
    if not canonical or len(frames) != 1 or not next(iter(frames)):
        return None
    fixed_frame = next(iter(frames))
    fingerprint_hash = hashlib.sha256(
        (dependencies.scene_contract + ":native_canonical").encode("utf-8")
    )
    for item in canonical:
        fingerprint_hash.update(Path(item["path"]).read_bytes())
    dependencies.write_ply(output / "scene_anchor_frame.ply", [])
    pose_variants: list[dict[str, Any]] = []
    frustum_variants: list[dict[str, Any]] = []
    for item in canonical:
        cameras = list(item["payload"].get("cameras", []))
        pose_variants.append(
            {
                "method": item["method"],
                "label": item["label"],
                "camera_topic": dependencies.topic(
                    item["method"], item["label"], "cameras"
                ),
                "anchor_edges_topic": dependencies.topic(
                    item["method"], item["label"], "anchor_edges"
                ),
                "error_lines_topic": dependencies.topic(
                    item["method"], item["label"], "error_lines"
                ),
                "cameras": cameras,
            }
        )
        frustums = []
        for camera in cameras:
            camera_id = str(camera.get("camera_id", ""))
            info = dependencies.intrinsics(
                dependencies.camera_info(experiment_root, camera_id)
            )
            if info is None:
                continue
            transform = validate_transform(
                np.asarray(camera["matrix"], dtype=np.float64)
            )
            frustums.append(
                {
                    "camera_id": camera_id,
                    "points": dependencies.frustum(transform, info, 0.34),
                }
            )
        frustum_variants.append(
            {
                "method": item["method"],
                "label": item["label"],
                "frustums": frustums,
            }
        )
    dependencies.write_json(
        output / "poses_anchor_frame.json",
        {
            "schema_version": 2,
            "fixed_frame": fixed_frame,
            "variants": pose_variants,
            "ground_truth": {"cameras": []},
        },
    )
    dependencies.write_json(
        output / "camera_frustums.json",
        {
            "schema_version": 2,
            "display_depth_m": 0.34,
            "variants": frustum_variants,
        },
    )
    dependencies.atomic_text(
        output / "rigcal_result.rviz",
        dependencies.rviz_config(fixed_frame, canonical, False),
    )
    dependencies.atomic_text(
        output / "launch_rviz.py",
        (
            "from camera_rig_calibration.visualization.session import "
            "main_for_generated_scene\n\n"
            "if __name__ == '__main__':\n"
            "    main_for_generated_scene(__file__)\n"
        ),
    )
    manifest = {
        "schema_version": 2,
        "contract": dependencies.scene_contract,
        "fingerprint": fingerprint_hash.hexdigest(),
        "status": "OK_NATIVE_CANONICAL_6DOF",
        "available": True,
        "fixed_frame": fixed_frame,
        "point_cloud": "scene_anchor_frame.ply",
        "point_count": 0,
        "poses": "poses_anchor_frame.json",
        "frustums": "camera_frustums.json",
        "rviz_config": "rigcal_result.rviz",
        "publisher": "camera_rig_calibration.visualization.ros_scene",
        "variants": [
            {"method": item["method"], "label": item["label"]}
            for item in canonical
        ],
        "warnings": [
            "Pose-only canonical scene; no AP03 point cloud or ground truth was used."
        ],
    }
    dependencies.write_json(output / "visualization_manifest.json", manifest)
    dependencies.update_result_status(experiment_root, manifest["status"])
    return manifest


__all__ = [
    "CanonicalSceneDependencies",
    "append_canonical_variants",
    "create_canonical_native_scene",
]
