"""Experiment manifest creation and stage invalidation rules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.models import RigConfig
from ..storage_layout import storage_manifest
from .identity import ExperimentPaths, experiment_fingerprint, file_sha256


STAGE_ORDER = (
    "capture_import",
    "input_preparation",
    "marker_detection_pnp",
    "observation_quality",
    "colmap",
    "method_estimation",
    "evaluation",
    "comparison",
    "report",
)

PARAMETER_INVALIDATION: dict[str, str] = {
    "simulation.route": "capture_import",
    "simulation.route_name": "capture_import",
    "simulation.world": "capture_import",
    "simulation.world_id": "capture_import",
    "simulation.moving_width": "capture_import",
    "simulation.moving_height": "capture_import",
    "simulation.moving_hfov_deg": "capture_import",
    "simulation.lighting": "capture_import",
    "simulation.lighting_scale": "capture_import",
    "simulation.motion_blur_kernel": "capture_import",
    "simulation.motion_blur_angle_deg": "capture_import",
    "sampling.target_hz": "input_preparation",
    "markers.dictionary": "marker_detection_pnp",
    "markers.detection_mode": "marker_detection_pnp",
    "markers.accepted_ids": "observation_quality",
    "observation_quality": "observation_quality",
    "markers.length_m": "method_estimation",
    "methods.ap01.root_camera": "method_estimation",
    "methods.ap02.reference_marker_id": "method_estimation",
    "methods.ap02.reference_marker_selection_mode": "method_estimation",
    "methods.ap03.single.scale_marker_id": "method_estimation",
    "methods.ap03.multi.marker_ids": "method_estimation",
    "colmap": "colmap",
    "evaluation": "evaluation",
    "reporting": "report",
}


def experiment_manifest_payload(
    config: RigConfig,
    paths: ExperimentPaths,
    input_id: str,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    route = config.simulation.route
    world = config.simulation.world
    return {
        "schema_version": 5,
        "layout_version": 2,
        "id": paths.experiment_id,
        "category": paths.category,
        "scene_type": config.dataset.scene_type.value,
        "source_kind": config.dataset.source_kind.value,
        "storage": storage_manifest(config),
        "experiment_fingerprint": experiment_fingerprint(config),
        "input_fingerprint": input_id,
        "created_at": (
            created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        "static_cameras": [
            {"id": camera.id, "label": camera.label}
            for camera in config.static_cameras
        ],
        "moving_camera": {"id": config.moving_camera.id},
        "simulation_parameters": (
            {
                "route": config.simulation.route_name,
                "moving_width": config.simulation.moving_width,
                "moving_height": config.simulation.moving_height,
                "moving_hfov_deg": config.simulation.moving_hfov_deg,
                "lighting": config.simulation.lighting,
                "lighting_scale": config.simulation.lighting_scale,
                "motion_blur_kernel": config.simulation.motion_blur_kernel,
                "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
                "target_route_frames": config.simulation.target_route_frames,
                "route_sampling_strategy": (
                    config.simulation.route_sampling_strategy
                ),
                "settle_seconds": config.simulation.settle_seconds,
                "post_pose_skip": config.simulation.post_pose_skip,
                "frame_timeout_seconds": (
                    config.simulation.frame_timeout_seconds
                ),
                "startup_timeout_seconds": (
                    config.simulation.startup_timeout_seconds
                ),
                "route_sha256": (
                    file_sha256(route)
                    if route is not None and route.is_file()
                    else None
                ),
                "world_sha256": (
                    file_sha256(world)
                    if world is not None and world.is_file()
                    else None
                ),
            }
            if paths.category == "simulation"
            else None
        ),
    }


def write_experiment_manifest(
    config: RigConfig, paths: ExperimentPaths, input_id: str
) -> Path:
    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    destination = paths.dataset_root / "dataset.json"
    payload = experiment_manifest_payload(config, paths, input_id)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        previous_fingerprint = existing.get("experiment_fingerprint")
        if (
            previous_fingerprint
            and previous_fingerprint != payload["experiment_fingerprint"]
        ):
            from ..dataset_identity import build_dataset_identity

            previous_input = existing.get("input_fingerprint")
            if previous_input != input_id:
                raise RuntimeError(
                    f"Experiment ID '{paths.experiment_id}' already belongs "
                    "to a different rig or immutable dataset. Choose a new "
                    "dataset/experiment ID instead of mixing inputs."
                )
            identity = build_dataset_identity(paths.dataset_root)
            if not identity.get("fingerprint") or not identity.get(
                "content_files"
            ):
                raise RuntimeError(
                    f"Experiment ID '{paths.experiment_id}' has no verifiable "
                    "immutable dataset content contract."
                )
            payload["dataset_identity"] = identity
            payload["legacy_experiment_fingerprint"] = previous_fingerprint
            payload["identity_migrated_from_method_resolved_config"] = True
            payload["created_at"] = existing.get(
                "created_at", payload["created_at"]
            )
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
            return destination
        previous_input = existing.get("input_fingerprint")
        if previous_input and previous_input != input_id:
            raise RuntimeError(
                f"Experiment ID '{paths.experiment_id}' already contains a "
                "different immutable dataset. Choose a new experiment ID."
            )
        return destination
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def first_invalidated_stage(changed_paths: list[str]) -> str:
    if not changed_paths:
        return "report"
    stages: list[str] = []
    for changed in changed_paths:
        match = next(
            (
                stage
                for prefix, stage in PARAMETER_INVALIDATION.items()
                if changed == prefix or changed.startswith(prefix + ".")
            ),
            "method_estimation",
        )
        stages.append(match)
    return min(stages, key=STAGE_ORDER.index)


__all__ = [
    "PARAMETER_INVALIDATION",
    "STAGE_ORDER",
    "experiment_manifest_payload",
    "first_invalidated_stage",
    "write_experiment_manifest",
]
