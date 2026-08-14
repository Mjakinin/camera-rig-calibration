"""Adapters from maintained AP artifacts to the canonical result contract."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..anchor_export.adapters import row_transform
from ..contracts import RunContext
from .results import CanonicalCameraPose, CanonicalMethodResult


def _csv_result(
    context: RunContext,
    status: dict[str, Any],
    *,
    method_id: str,
    algorithm_version: str,
    artifact_directory: str,
    pose_path: str,
    reference_frame: str,
) -> CanonicalMethodResult:
    source = context.run_directory / artifact_directory / pose_path
    poses: list[CanonicalCameraPose] = []
    warnings: list[str] = []
    if source.is_file():
        with source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                camera_id = str(
                    row.get("entity_id")
                    or row.get("camera_id")
                    or row.get("camera_name")
                    or ""
                ).strip()
                if not camera_id:
                    continue
                try:
                    poses.append(
                        CanonicalCameraPose.from_transform(
                            camera_id=camera_id,
                            reference_frame=reference_frame,
                            transform=row_transform(row),
                            source=pose_path,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {method_id} pose for '{camera_id}': {exc}"
                    ) from exc
    else:
        warnings.append(f"Native pose artifact is unavailable: {pose_path}")
    available = bool(poses)
    return CanonicalMethodResult(
        method_id=method_id,
        algorithm_version=algorithm_version,
        status="available" if available else "incomplete",
        reference_frame=reference_frame,
        camera_poses=poses,
        quality_status=str(status.get("quality_status", "unknown")),
        metrics={
            "available_static_cameras": [pose.camera_id for pose in poses],
            "expected_static_camera_count": len(context.config.static_cameras),
            "native_method_status": status.get("status"),
        },
        warnings=warnings,
        native_artifacts={"primary_camera_poses": pose_path},
    )


def ap01_result(
    context: RunContext, status: dict[str, Any]
) -> CanonicalMethodResult:
    reference = str(
        status.get("root_camera")
        or status.get("reference_camera")
        or context.resolved_root_camera
        or context.config.methods.ap01.root_camera
    )
    return _csv_result(
        context,
        status,
        method_id="ap01",
        algorithm_version="ap01_explicit_method_contract_v2",
        artifact_directory="02_AP01",
        pose_path=(
            "03_static_extrinsics/"
            "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        ),
        reference_frame=reference,
    )


def ap02_result(
    context: RunContext, status: dict[str, Any]
) -> CanonicalMethodResult:
    marker = status.get(
        "reference_marker_id", context.resolved_ap02_reference_marker_id
    )
    return _csv_result(
        context,
        status,
        method_id="ap02",
        algorithm_version="ap02_maximum_frontier_v1",
        artifact_directory="03_AP02",
        pose_path=(
            "07_graph_ba/with_moving/"
            "optimized_static_camera_poses_ref_marker.csv"
        ),
        reference_frame=f"ArUco marker {marker}",
    )


def ap03_result(
    context: RunContext, status: dict[str, Any]
) -> CanonicalMethodResult:
    return _csv_result(
        context,
        status,
        method_id="ap03",
        algorithm_version="ap03_baseline_method_contract_v1",
        artifact_directory="04_AP03",
        pose_path=(
            "scale_multi/"
            "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
        ),
        reference_frame="COLMAP gauge with metric marker scale",
    )


__all__ = ["ap01_result", "ap02_result", "ap03_result"]
