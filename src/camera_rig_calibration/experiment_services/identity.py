"""Input and experiment identity independent from method configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.models import RigConfig
from ..dataset.manifest import DatasetManifest
from ..storage_layout import (
    canonical_dataset_root,
    canonical_result_root,
    storage_key,
)


@dataclass(frozen=True)
class ExperimentPaths:
    category: str
    experiment_id: str
    root: Path
    dataset_root: Path
    datasets: Path
    methods: Path
    evaluations: Path
    comparisons: Path
    attempts: Path
    artifacts: Path
    staging: Path

    @property
    def inputs(self) -> Path:
        return self.datasets


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[
        :length
    ]


def file_sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def input_fingerprint(
    manifest: DatasetManifest | None, dataset_root: Path
) -> str:
    provenance: list[dict[str, Any]] = [
        {
            "role": f"source:{item.role}",
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in (manifest.files if manifest is not None else [])
        if item.sha256
    ]
    raw_images = dataset_root / "raw_images"
    if raw_images.is_dir():
        for path in sorted(
            item for item in raw_images.rglob("*") if item.is_file()
        ):
            provenance.append(
                {
                    "role": (
                        "prepared:"
                        f"{path.relative_to(dataset_root).as_posix()}"
                    ),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not provenance:
        metadata = dataset_root / "metadata"
        if metadata.is_dir():
            for path in sorted(
                item for item in metadata.rglob("*") if item.is_file()
            ):
                provenance.append(
                    {
                        "role": (
                            "prepared:"
                            f"{path.relative_to(dataset_root).as_posix()}"
                        ),
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    payload = {
        "static_camera_ids": (
            [camera.id for camera in manifest.static_cameras]
            if manifest
            else []
        ),
        "moving_camera_id": manifest.moving_camera.id if manifest else None,
        "sampling_hz": manifest.sampling_hz if manifest else None,
        "files": sorted(
            provenance,
            key=lambda item: (
                str(item["role"]),
                str(item["sha256"]),
                int(item["size_bytes"] or 0),
            ),
        ),
    }
    return f"input_{digest(payload)}"


def result_category(config: RigConfig) -> str:
    return config.dataset.category.value


def experiment_paths(config: RigConfig) -> ExperimentPaths:
    key = storage_key(config)
    experiment_id = config.project.experiment_id or config.dataset.id
    root = canonical_result_root(config)
    dataset_root = canonical_dataset_root(config)
    staging = (
        config.project.workspace_root.resolve()
        / "temporary_runs"
        / f"standalone_{key.canonical_id}"
        / "jobs"
    )
    return ExperimentPaths(
        category=key.category,
        experiment_id=experiment_id,
        root=root,
        dataset_root=dataset_root,
        datasets=dataset_root,
        methods=root / "methods",
        evaluations=root / "evaluations",
        comparisons=root,
        attempts=root / "attempts",
        artifacts=(
            config.project.workspace_root.resolve()
            / "cache"
            / "colmap"
            / key.category
            / key.relative
        ),
        staging=staging,
    )


def colmap_artifact_fingerprint(
    config: RigConfig, method_id: str, input_id: str
) -> str:
    if method_id == "ap01":
        family = "ap01_moving"
        cameras: list[str] = []
    elif method_id == "ap03":
        family = "ap03_grouped"
        cameras = [camera.id for camera in config.static_cameras]
    else:
        raise ValueError(f"Method '{method_id}' has no COLMAP artifact")
    colmap = config.colmap.model_dump(mode="json")
    colmap.pop("reuse", None)
    return digest(
        {
            "family": family,
            "input_id": input_id,
            "static_cameras": cameras,
            "moving_camera": config.moving_camera.id,
            "colmap": colmap,
        },
        64,
    )


def experiment_fingerprint(config: RigConfig) -> str:
    def asset_hash(path: Path | None) -> str | None:
        return file_sha256(path) if path is not None and path.is_file() else None

    simulation = {
        "route_name": config.simulation.route_name,
        "moving_width": config.simulation.moving_width,
        "moving_height": config.simulation.moving_height,
        "moving_hfov_deg": config.simulation.moving_hfov_deg,
        "lighting": config.simulation.lighting,
        "lighting_scale": config.simulation.lighting_scale,
        "motion_blur_kernel": config.simulation.motion_blur_kernel,
        "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
        "target_route_frames": config.simulation.target_route_frames,
        "route_sampling_strategy": config.simulation.route_sampling_strategy,
        "world_id": config.simulation.world_id,
        "world_baseline": config.simulation.world_baseline,
        "route_sha256": asset_hash(config.simulation.route),
        "world_sha256": asset_hash(config.simulation.world),
    }
    payload = {
        "category": result_category(config),
        "experiment_id": config.project.experiment_id or config.dataset.id,
        "scene_type": config.dataset.scene_type.value,
        "static_cameras": [
            {"id": camera.id, "label": camera.label}
            for camera in config.static_cameras
        ],
        "moving_camera": config.moving_camera.id,
        "simulation": (
            simulation
            if result_category(config) == "simulation"
            else None
        ),
        "sampling": config.sampling.model_dump(mode="json"),
    }
    return digest(payload, 64)


__all__ = [
    "ExperimentPaths",
    "canonical_json",
    "colmap_artifact_fingerprint",
    "digest",
    "experiment_fingerprint",
    "experiment_paths",
    "file_sha256",
    "input_fingerprint",
    "result_category",
]
