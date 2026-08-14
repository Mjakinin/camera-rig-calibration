from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config.models import RigConfig, StaticCameraSettings
from ..contracts import CommandSpec
from ..dataset.manifest import (
    CameraManifest,
    DatasetManifest,
    FileProvenance,
    load_dataset_manifest,
    save_dataset_manifest,
)
from ..intrinsics_profiles import resolve_intrinsic_profile
from .video_geometry import open_oriented_video


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


from .preparation_files import PreparationPlan, _materialize_tree, _sha256


def finalize_dataset(config: RigConfig, plan: PreparationPlan) -> DatasetManifest:
    if config.dataset.scene_type.value == "simulation":
        from .simulation import capture_frame_diversity

        simulation_frames = sorted(
            (plan.dataset_root.resolve() / "raw_images" / "moving").glob(
                "frame_*.*"
            )
        )
        capture_frame_diversity(simulation_frames)
    if plan.existing_manifest is not None:
        return plan.existing_manifest.model_copy(
            update={
                "scene_type": config.dataset.scene_type,
                "sampling_hz": config.sampling.target_hz,
                "marker_dictionary": config.markers.dictionary,
                "marker_length_m": config.markers.length_m,
            },
            deep=True,
        )
    root = plan.dataset_root.resolve()
    if plan.acquisition_root is not None:
        acquisition = plan.acquisition_root.resolve()
        marker = acquisition / "ACQUISITION_COMPLETE.json"
        if not marker.is_file() and acquisition != root:
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "acquisition_fingerprint": plan.acquisition_fingerprint,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        moving_info = (
            acquisition
            / "raw_images"
            / "camera_info"
            / f"{config.moving_camera.id}.json"
        )
        _materialize_tree(
            acquisition / "raw_images",
            root / "raw_images",
            excluded=(
                {moving_info}
                if plan.moving_intrinsics_override
                else set()
            ),
        )
        _materialize_tree(
            acquisition / "metadata",
            root / "metadata" / "acquisition",
        )
    raw = root / "raw_images"
    static_manifests = []
    for camera in config.static_cameras:
        images = sorted((raw / "static_multi" / camera.id).glob("*.png"))
        if not images:
            images = sorted((raw / "static").glob(f"{camera.id}.*"))
        static_manifests.append(
            CameraManifest(
                id=camera.id,
                kind="static",
                image_count=len(images),
                images=[str(path.relative_to(root)) for path in images],
                intrinsics=str(Path("raw_images/camera_info") / f"{camera.id}.json"),
                source_topic=camera.image_topic,
            )
        )
    moving_frames = sorted((raw / "moving").glob("frame_*.*"))
    provenance = []
    for role, path in plan.source_files:
        if path.is_file():
            key = f"{role}\0{path}"
            provenance.append(
                FileProvenance(
                    role=role,
                    path=str(path),
                    sha256=plan.source_hashes.get(key) or _sha256(path),
                    size_bytes=path.stat().st_size,
                )
            )
    resolved_moving_info = (
        raw
        / "camera_info"
        / f"{config.moving_camera.id}.json"
    )
    if resolved_moving_info.is_file():
        provenance.append(
            FileProvenance(
                role="resolved_moving_intrinsics",
                path=str(resolved_moving_info),
                sha256=_sha256(resolved_moving_info),
                size_bytes=resolved_moving_info.stat().st_size,
            )
        )
    manifest = DatasetManifest(
        dataset_id=config.dataset.id,
        scene_type=config.dataset.scene_type,
        prepared_root=str(root),
        static_cameras=static_manifests,
        moving_camera=CameraManifest(
            id=config.moving_camera.id,
            kind="moving",
            image_count=len(moving_frames),
            images=[str(path.relative_to(root)) for path in moving_frames],
            intrinsics=str(
                Path("raw_images/camera_info") / f"{config.moving_camera.id}.json"
            ),
            source_topic=config.moving_camera.image_topic,
        ),
        sampling_hz=config.sampling.target_hz,
        marker_dictionary=config.markers.dictionary,
        marker_length_m=config.markers.length_m,
        simulation_parameters=(
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
                "route_sampling_strategy": config.simulation.route_sampling_strategy,
            }
            if config.dataset.scene_type.value == "simulation"
            else {}
        ),
        files=provenance,
        notes=[
            "scene_type is descriptive metadata and does not change method mathematics",
            *(
                [
                    "Simulation inputs were captured into a new immutable dataset cache; published results were not overwritten."
                ]
                if config.simulation.enabled
                else []
            ),
        ],
    )
    if not plan.prepared_input:
        save_dataset_manifest(manifest, root / "dataset_manifest.json")
    return manifest
