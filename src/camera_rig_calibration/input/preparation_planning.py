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


from .preparation_files import (
    PreparationPlan,
    _configured_source_files,
    _copy_moving_frames,
    _copy_static_camera,
    _hash_sources,
    _image_files,
    _normalize_intrinsics,
    _prepared_source_files,
    _sha256,
)


def _preparation_fingerprint(config: RigConfig) -> str:
    payload = {
        "dataset": config.dataset.model_dump(mode="json", exclude={"input_root"}),
        "static_cameras": [
            camera.model_dump(mode="json") for camera in config.static_cameras
        ],
        "moving_camera": config.moving_camera.model_dump(mode="json"),
        "mcap": config.mcap.model_dump(mode="json"),
        "simulation": config.simulation.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "marker_dictionary": config.markers.dictionary,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fingerprint_sources(
    payload: dict[str, object],
    sources: list[tuple[str, Path]],
) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for role, path in sorted(sources, key=lambda item: (item[0], str(item[1]))):
        if path.is_file():
            digest.update(role.encode("utf-8"))
            digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _resolved_moving_intrinsics(
    config: RigConfig, repository_root: Path
) -> tuple[RigConfig, Path | None]:
    moving = config.moving_camera
    if (
        moving.intrinsics_profile
        and moving.intrinsics is None
        and moving.intrinsic_calibration_video is None
        and moving.intrinsic_calibration_images is None
    ):
        profile = resolve_intrinsic_profile(
            repository_root, moving.intrinsics_profile
        )
        moving = moving.model_copy(update={"intrinsics": profile.intrinsics})
        return (
            config.model_copy(update={"moving_camera": moving}, deep=True),
            profile.intrinsics,
        )
    return config, moving.intrinsics


def _real_acquisition_payload(config: RigConfig) -> dict[str, object]:
    moving = config.moving_camera.model_dump(
        mode="json",
        exclude={
            "intrinsics",
            "intrinsics_profile",
            "intrinsic_calibration_video",
            "intrinsic_calibration_images",
            "intrinsic_scan",
            "checkerboard_columns",
            "checkerboard_rows",
            "intrinsic_maximum_views",
            "intrinsic_minimum_frame_gap",
            "intrinsic_minimum_detections",
        },
    )
    return {
        "static_cameras": [
            camera.model_dump(mode="json") for camera in config.static_cameras
        ],
        "moving_camera": moving,
        "mcap": config.mcap.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "marker_dictionary": config.markers.dictionary,
        "video_orientation_policy": "apply_ffprobe_display_rotation",
        "contract": "rigcal_real_acquisition_v2",
    }


def _real_acquisition_sources(
    config: RigConfig, prepared_root: Path | None
) -> list[tuple[str, Path]]:
    if prepared_root is not None:
        sources = _prepared_source_files(config, prepared_root)
        moving_info = (
            prepared_root
            / "raw_images"
            / "camera_info"
            / f"{config.moving_camera.id}.json"
        ).resolve()
        return [
            item
            for item in sources
            if not (
                item[0] == "prepared_moving_intrinsics"
                and item[1].resolve() == moving_info
            )
        ]
    return [
        item
        for item in _configured_source_files(config)
        if item[0]
        not in {
            "moving_intrinsics",
            "intrinsic_calibration_video",
            "intrinsic_calibration_image",
        }
    ]


def _build_real_preparation_plan(
    config: RigConfig,
    repository_root: Path,
) -> PreparationPlan:
    config, selected_intrinsics = _resolved_moving_intrinsics(
        config, repository_root
    )
    moving = config.moving_camera
    prepared_root = config.dataset.prepared_root
    if prepared_root is not None:
        prepared_root = prepared_root.resolve()
        if prepared_root.name == "raw_images":
            prepared_root = prepared_root.parent
    has_override = (
        selected_intrinsics is not None
        or moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
        or moving.intrinsics_profile is not None
    )
    if prepared_root is not None and not has_override:
        sources = _prepared_source_files(config, prepared_root)
        manifest_path = prepared_root / "dataset_manifest.json"
        if not manifest_path.is_file():
            manifest_path = (
                prepared_root / "metadata" / "dataset_manifest.json"
            )
        return PreparationPlan(
            prepared_root,
            source_files=sources,
            source_hashes=_hash_sources(sources),
            prepared_input=True,
            existing_manifest=(
                load_dataset_manifest(manifest_path)
                if manifest_path.is_file()
                else None
            ),
        )

    acquisition_sources = _real_acquisition_sources(config, prepared_root)
    acquisition_fingerprint = _fingerprint_sources(
        _real_acquisition_payload(config), acquisition_sources
    )
    acquisition_root = (
        prepared_root
        if prepared_root is not None
        else config.project.dataset_cache_root.resolve()
        / "_acquisitions"
        / acquisition_fingerprint[:12]
    )
    intrinsic_sources: list[tuple[str, Path]] = []
    if selected_intrinsics is not None:
        intrinsic_sources.append(
            ("moving_intrinsics", selected_intrinsics.resolve())
        )
    if moving.intrinsic_calibration_video is not None:
        intrinsic_sources.append(
            (
                "intrinsic_calibration_video",
                moving.intrinsic_calibration_video.resolve(),
            )
        )
    if moving.intrinsic_calibration_images is not None:
        intrinsic_sources.extend(
            ("intrinsic_calibration_image", path)
            for path in _image_files(moving.intrinsic_calibration_images)
        )
    if not intrinsic_sources and prepared_root is None:
        # A ROS recording may provide the moving CameraInfo together with its
        # moving image topic. Its MCAP hash is already part of the acquisition.
        intrinsic_identity: dict[str, object] = {
            "source": "acquisition_camera_info"
        }
    else:
        intrinsic_identity = {
            "sources": [
                {"role": role, "sha256": _sha256(path)}
                for role, path in intrinsic_sources
            ],
            "profile": moving.intrinsics_profile,
            "scan": moving.intrinsic_scan.model_dump(mode="json"),
            "checkerboard": {
                "columns": moving.checkerboard_columns,
                "rows": moving.checkerboard_rows,
                "maximum_views": moving.intrinsic_maximum_views,
                "minimum_frame_gap": moving.intrinsic_minimum_frame_gap,
                "minimum_detections": moving.intrinsic_minimum_detections,
            },
        }
    composition_payload = {
        "acquisition_fingerprint": acquisition_fingerprint,
        "moving_camera_id": moving.id,
        "intrinsics": intrinsic_identity,
        "contract": "rigcal_real_composition_v1",
    }
    composition_fingerprint = hashlib.sha256(
        json.dumps(
            composition_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    dataset_root = (
        config.project.dataset_cache_root.resolve()
        / config.dataset.id
        / composition_fingerprint[:12]
    )
    all_sources = [*acquisition_sources, *intrinsic_sources]
    all_hashes = _hash_sources(all_sources)
    existing_path = dataset_root / "dataset_manifest.json"
    if existing_path.is_file():
        return PreparationPlan(
            dataset_root,
            source_files=all_sources,
            source_hashes=all_hashes,
            existing_manifest=load_dataset_manifest(existing_path),
            acquisition_root=acquisition_root,
            acquisition_fingerprint=acquisition_fingerprint,
            moving_intrinsics_override=has_override,
        )

    for directory in (
        dataset_root / "raw_images" / "static",
        dataset_root / "raw_images" / "moving",
        dataset_root / "raw_images" / "camera_info",
        dataset_root / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    acquisition_ready = prepared_root is not None or (
        acquisition_root / "ACQUISITION_COMPLETE.json"
    ).is_file()
    commands: list[CommandSpec] = []
    if not acquisition_ready:
        for directory in (
            acquisition_root / "raw_images" / "static",
            acquisition_root / "raw_images" / "moving",
            acquisition_root / "raw_images" / "camera_info",
            acquisition_root / "metadata",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for camera in config.static_cameras:
            _copy_static_camera(camera, acquisition_root, [])
        _copy_moving_frames(config, acquisition_root, [])
        if config.mcap.path is not None:
            mapping_path = (
                acquisition_root / "metadata" / "mcap_topic_mapping.json"
            )
            mapping = {
                "marker_dictionary": config.markers.dictionary,
                "save_all_candidates": config.mcap.save_all_candidates,
                "moving_sampling_hz": config.sampling.target_hz,
                "cameras": [
                    {
                        "id": camera.id,
                        "image_topic": camera.image_topic,
                        "camera_info_topic": camera.camera_info_topic,
                    }
                    for camera in config.static_cameras
                ],
                "moving_camera": (
                    {
                        "id": moving.id,
                        "image_topic": moving.image_topic,
                        "camera_info_topic": moving.camera_info_topic,
                    }
                    if moving.video is None
                    and moving.frames is None
                    and moving.image_topic is not None
                    else None
                ),
            }
            mapping_path.write_text(
                json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
            )
            commands.append(
                CommandSpec(
                    "prepare_mcap",
                    "Extract selected camera inputs from MCAP",
                    (
                        sys.executable,
                        "-m",
                        "camera_rig_calibration.input.mcap",
                        "--mcap",
                        str(config.mcap.path.resolve()),
                        "--dataset",
                        str(acquisition_root),
                        "--mapping",
                        str(mapping_path),
                    ),
                    repository_root,
                    acquisition_root,
                )
            )
        if moving.video is not None:
            argv = [
                sys.executable,
                str(
                    repository_root
                    / "src/camera_rig_calibration/input/video.py"
                ),
                "--video",
                str(moving.video.resolve()),
                "--dataset",
                str(acquisition_root),
                "--target-fps",
                str(config.sampling.target_hz),
                "--start-s",
                str(config.sampling.start_seconds),
            ]
            if config.sampling.end_seconds is not None:
                argv += ["--end-s", str(config.sampling.end_seconds)]
            if config.sampling.maximum_frames is not None:
                argv += ["--max-frames", str(config.sampling.maximum_frames)]
            argv.append("--overwrite")
            commands.append(
                CommandSpec(
                    "prepare_moving_frames",
                    "Extract moving-camera video frames",
                    tuple(argv),
                    repository_root,
                    acquisition_root / "raw_images" / "moving",
                )
            )

    info_destination = (
        dataset_root / "raw_images" / "camera_info" / f"{moving.id}.json"
    )
    if selected_intrinsics is not None:
        _normalize_intrinsics(
            selected_intrinsics.resolve(), info_destination, moving.id
        )
    elif (
        moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
    ):
        calibration_source = (
            moving.intrinsic_calibration_video
            or moving.intrinsic_calibration_images
        )
        assert calibration_source is not None
        source_option = (
            "--video"
            if moving.intrinsic_calibration_video is not None
            else "--images"
        )
        profile_id = (
            moving.intrinsics_profile
            or calibration_source.stem
        ).split("@", 1)[0]
        commands.append(
            CommandSpec(
                "prepare_moving_intrinsics",
                "Calibrate moving-camera intrinsics",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.intrinsics",
                    "--script",
                    str(
                        repository_root
                        / "src/camera_rig_calibration/input/intrinsics_calibration.py"
                    ),
                    source_option,
                    str(calibration_source.resolve()),
                    "--work-directory",
                    str(dataset_root / "metadata" / "intrinsic_calibration"),
                    "--destination",
                    str(info_destination),
                    "--camera-id",
                    moving.id,
                    "--repository",
                    str(repository_root),
                    "--profile-id",
                    profile_id,
                    "--cols",
                    str(moving.checkerboard_columns),
                    "--rows",
                    str(moving.checkerboard_rows),
                    "--max-views",
                    str(moving.intrinsic_maximum_views),
                    "--minimum-frame-gap",
                    str(moving.intrinsic_minimum_frame_gap),
                    "--minimum-detections",
                    str(moving.intrinsic_minimum_detections),
                    "--scan-mode",
                    moving.intrinsic_scan.mode,
                    "--scan-target-hz",
                    str(moving.intrinsic_scan.target_hz),
                    "--preview-max-dimension",
                    str(moving.intrinsic_scan.preview_max_dimension),
                ),
                repository_root,
                info_destination.parent,
            )
        )
    return PreparationPlan(
        dataset_root,
        commands,
        all_sources,
        all_hashes,
        prepared_input=prepared_root is not None,
        acquisition_root=acquisition_root,
        acquisition_fingerprint=acquisition_fingerprint,
        moving_intrinsics_override=has_override,
    )


def build_preparation_plan(config: RigConfig, repository_root: Path) -> PreparationPlan:
    if not config.simulation.enabled:
        return _build_real_preparation_plan(config, repository_root)
    config, _ = _resolved_moving_intrinsics(config, repository_root)
    if config.dataset.prepared_root is not None:
        root = config.dataset.prepared_root.resolve()
        if root.name == "raw_images":
            root = root.parent
        sources = _prepared_source_files(config, root)
        manifest_path = root / "dataset_manifest.json"
        if not manifest_path.is_file():
            manifest_path = root / "metadata" / "dataset_manifest.json"
        return PreparationPlan(
            root,
            source_files=sources,
            source_hashes=_hash_sources(sources),
            prepared_input=True,
            existing_manifest=(
                load_dataset_manifest(manifest_path)
                if manifest_path.is_file()
                else None
            ),
        )

    sources = _configured_source_files(config)
    source_hashes = _hash_sources(sources)
    digest = hashlib.sha256(_preparation_fingerprint(config).encode("ascii"))
    for key, value in sorted(source_hashes.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.encode("ascii"))
    fingerprint = digest.hexdigest()[:12]
    dataset_root = (
        config.project.dataset_cache_root.resolve() / config.dataset.id / fingerprint
    )
    existing_path = dataset_root / "dataset_manifest.json"
    if existing_path.is_file():
        return PreparationPlan(
            dataset_root,
            source_files=sources,
            source_hashes=source_hashes,
            existing_manifest=load_dataset_manifest(existing_path),
        )
    for directory in (
        dataset_root / "raw_images" / "static",
        dataset_root / "raw_images" / "moving",
        dataset_root / "raw_images" / "camera_info",
        dataset_root / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for camera in config.static_cameras:
        _copy_static_camera(camera, dataset_root, [])
    _copy_moving_frames(config, dataset_root, [])

    commands: list[CommandSpec] = []
    if config.simulation.enabled:
        mapping_path = dataset_root / "metadata" / "simulation_capture.json"
        mapping = {
            "preset": config.simulation.preset,
            "capture_id": config.simulation.capture_id,
            "world": str(config.simulation.world.resolve()),
            "route": str(config.simulation.route.resolve()),
            "moving_model_name": config.simulation.moving_model_name,
            "moving_sensor_name": config.simulation.moving_sensor_name,
            "world_id": config.simulation.world_id,
            "world_baseline": config.simulation.world_baseline,
            "resource_paths": [
                str(path)
                for path in config.simulation.resource_paths
            ],
            "settle_seconds": config.simulation.settle_seconds,
            "post_pose_skip": config.simulation.post_pose_skip,
            "frame_timeout_seconds": config.simulation.frame_timeout_seconds,
            "startup_timeout_seconds": config.simulation.startup_timeout_seconds,
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
            "static_cameras": [
                {
                    "id": camera.id,
                    "image_topic": camera.image_topic,
                    "camera_info_topic": camera.camera_info_topic,
                    "intrinsics_source": (
                        "provided"
                        if camera.intrinsics is not None
                        else "gazebo_camera_info"
                    ),
                }
                for camera in config.static_cameras
            ],
            "moving_camera": {
                "id": config.moving_camera.id,
                "image_topic": config.moving_camera.image_topic,
                "camera_info_topic": config.moving_camera.camera_info_topic,
                "intrinsics_source": (
                    "provided"
                    if config.moving_camera.intrinsics is not None
                    else "gazebo_camera_info"
                ),
            },
        }
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        commands.append(
            CommandSpec(
                "prepare_simulation",
                "Launch Gazebo and capture simulation dataset",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.simulation",
                    "--repository",
                    str(repository_root),
                    "--dataset",
                    str(dataset_root),
                    "--mapping",
                    str(mapping_path),
                ),
                repository_root,
                dataset_root,
            )
        )
    if config.mcap.path is not None:
        mapping_path = dataset_root / "metadata" / "mcap_topic_mapping.json"
        mapping = {
            "marker_dictionary": config.markers.dictionary,
            "save_all_candidates": config.mcap.save_all_candidates,
            "moving_sampling_hz": config.sampling.target_hz,
            "cameras": [
                {
                    "id": camera.id,
                    "image_topic": camera.image_topic,
                    "camera_info_topic": camera.camera_info_topic,
                }
                for camera in config.static_cameras
            ],
            "moving_camera": (
                {
                    "id": config.moving_camera.id,
                    "image_topic": config.moving_camera.image_topic,
                    "camera_info_topic": config.moving_camera.camera_info_topic,
                }
                if config.moving_camera.video is None
                and config.moving_camera.frames is None
                and config.moving_camera.image_topic is not None
                else None
            ),
        }
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        commands.append(
            CommandSpec(
                "prepare_mcap",
                "Extract selected static images from MCAP",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.mcap",
                    "--mcap",
                    str(config.mcap.path.resolve()),
                    "--dataset",
                    str(dataset_root),
                    "--mapping",
                    str(mapping_path),
                ),
                repository_root,
                dataset_root,
            )
        )

    moving = config.moving_camera
    if moving.video is not None:
        argv = [
            sys.executable,
            str(repository_root / "src/camera_rig_calibration/input/video.py"),
            "--video",
            str(moving.video.resolve()),
            "--dataset",
            str(dataset_root),
            "--target-fps",
            str(config.sampling.target_hz),
            "--start-s",
            str(config.sampling.start_seconds),
        ]
        if config.sampling.end_seconds is not None:
            argv += ["--end-s", str(config.sampling.end_seconds)]
        if config.sampling.maximum_frames is not None:
            argv += ["--max-frames", str(config.sampling.maximum_frames)]
        argv.append("--overwrite")
        commands.append(
            CommandSpec(
                "prepare_moving_frames",
                "Extract moving-camera video frames",
                tuple(argv),
                repository_root,
                dataset_root / "raw_images" / "moving",
            )
        )

    info_destination = dataset_root / "raw_images" / "camera_info" / f"{moving.id}.json"
    if moving.intrinsics is not None:
        _normalize_intrinsics(moving.intrinsics.resolve(), info_destination, moving.id)
    elif (
        moving.intrinsic_calibration_video is not None
        or moving.intrinsic_calibration_images is not None
    ):
        calibration_source = (
            moving.intrinsic_calibration_video
            or moving.intrinsic_calibration_images
        )
        assert calibration_source is not None
        source_option = (
            "--video"
            if moving.intrinsic_calibration_video is not None
            else "--images"
        )
        commands.append(
            CommandSpec(
                "prepare_moving_intrinsics",
                "Calibrate moving-camera intrinsics",
                (
                    sys.executable,
                    "-m",
                    "camera_rig_calibration.input.intrinsics",
                    "--script",
                    str(
                        repository_root
                        / "src/camera_rig_calibration/input/intrinsics_calibration.py"
                    ),
                    source_option,
                    str(calibration_source.resolve()),
                    "--work-directory",
                    str(dataset_root / "metadata" / "intrinsic_calibration"),
                    "--destination",
                    str(info_destination),
                    "--camera-id",
                    moving.id,
                    "--cols",
                    str(moving.checkerboard_columns),
                    "--rows",
                    str(moving.checkerboard_rows),
                    "--max-views",
                    str(moving.intrinsic_maximum_views),
                    "--minimum-frame-gap",
                    str(moving.intrinsic_minimum_frame_gap),
                    "--minimum-detections",
                    str(moving.intrinsic_minimum_detections),
                    "--scan-mode",
                    moving.intrinsic_scan.mode,
                    "--scan-target-hz",
                    str(moving.intrinsic_scan.target_hz),
                    "--preview-max-dimension",
                    str(moving.intrinsic_scan.preview_max_dimension),
                ),
                repository_root,
                info_destination.parent,
            )
        )

    return PreparationPlan(
        dataset_root,
        commands,
        sources,
        source_hashes,
        prepared_input=False,
    )
