"""Focused wizard responsibilities extracted from the compatibility facade."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_user_config
from ..config.models import (
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    McapSettings,
    MethodSettings,
    MarkerSettings,
    MovingCameraSettings,
    IntrinsicScanSettings,
    InputSourceKind,
    ObservationQualitySettings,
    ProjectSettings,
    RigConfig,
    SamplingSettings,
    SceneType,
    SelectionSettings,
    SimulationSettings,
    StaticCameraSettings,
    effective_observation_quality,
)
from ..dataset.discovery import (
    IMAGE_SUFFIXES,
    discover_image_directories,
    discover_inputs,
    inspect_prepared_dataset,
    media_path_role,
    safe_id,
)
from ..doctor import run_checks
from ..experiments import automatic_method_label
from ..input.topics import McapTopic, list_mcap_topics
from ..input.video_geometry import probe_video_geometry
from ..intrinsics_profiles import (
    IntrinsicProfile,
    discover_intrinsic_profiles,
    intrinsic_dimensions,
)
from ..inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    PreparedDatasetSummary,
    RawInputSummary,
    SimulationExperimentSummary,
    discover_prepared_datasets,
    discover_raw_input_folders,
    discover_simulation_experiments,
    find_matching_simulation,
    format_simulation_parameters,
)
from ..registry import (
    calibration_methods,
    experiment_providers,
    input_adapters,
)
from ..runtime import PipelineOrchestrator
from ..observation_quality import filter_observations
from ..observations import ResolvedSelections, resolve_selections
from ..queueing import SelectionReviewJob, save_batch
# Compatibility hooks wrapped by the product policy stack. The concrete result
# browser lives under ui/, but these names remain stable until the wrappers are
# converted to explicit composition.
from ..publication import reconcile_existing_experiment
from ..visualization import launch_isolated_rviz



from .wizard_input_metadata import (
    _stored_prepared_sampling,
)
from .wizard_models import (
    WizardBack,
)
from .wizard_prompts import (
    _checkerboard_sources,
    _choice,
    _moving_media_dimensions,
    _preferred_path,
    _prompt_intrinsic_scan_settings,
    _select_checkerboard_source,
    _select_detected_path,
    _show_video_geometry,
)

def _moving_source(
    console: Console,
    input_root: Path,
    repository_root: Path | None = None,
) -> tuple[MovingCameraSettings, SamplingSettings, str]:
    discovered = discover_inputs(input_root, recursive=True)
    all_videos = [item.path for item in discovered if item.kind == "video"]
    videos = [
        path
        for path in all_videos
        if media_path_role(path, input_root) not in {"checkerboard", "static"}
    ]
    intrinsics_files = [item.path for item in discovered if item.kind == "intrinsics"]
    image_directories = discover_image_directories(input_root)
    frame_directories = image_directories["moving"]
    checkerboard_sources = _checkerboard_sources(input_root)
    has_local_images = any(item.kind == "image" for item in discovered)
    mode = _choice(
        "Moving-camera input",
        {
            "1": f"video ({len(videos)} detected)",
            "2": f"extracted frames ({len(frame_directories)} folders detected)",
        },
        "1" if videos or not frame_directories else "2",
    )
    if mode == "1":
        preferred = _preferred_path(videos, ("moving", "route", "drive", "record"))
        video = _select_detected_path(
            "Moving-camera video", videos, directory=False, preferred=preferred
        )
        frames = None
        suggested_id = safe_id(video.stem)
        _show_video_geometry(console, "Moving video geometry", video)
    else:
        video = None
        frames = _select_detected_path(
            "Moving-camera frames directory",
            frame_directories,
            directory=True,
        )
        suggested_id = safe_id(frames.name)
    checkerboard_paths = {item[1].resolve() for item in checkerboard_sources}
    generic_checkerboard_videos = [
        path
        for path in all_videos
        if path != video and path.resolve() not in checkerboard_paths
    ]
    if not generic_checkerboard_videos and not checkerboard_sources and video is not None:
        generic_checkerboard_videos = [video]
    checkerboard_sources.extend(
        ("video", path, f"video: {path}")
        for path in generic_checkerboard_videos
    )

    intrinsics = None
    intrinsics_profile = None
    calibration_video = None
    calibration_images = None
    profile_id = None
    profiles = (
        discover_intrinsic_profiles(repository_root)
        if repository_root is not None
        else []
    )
    media_dimensions = _moving_media_dimensions(video, frames)
    if media_dimensions is not None:
        profiles = [
            profile
            for profile in profiles
            if (profile.width, profile.height) == media_dimensions
        ]
    static_intrinsic_paths = {
        camera_info.resolve()
        for _, _, _, camera_info in _detected_static_camera_groups(
            input_root,
            MovingCameraSettings(id="moving_calib_camera", video=video, frames=frames),
        )
    }
    options: list[tuple[str, Path | IntrinsicProfile | None, str]] = []
    ordered_intrinsics = sorted(
        [
            path
            for path in intrinsics_files
            if path.resolve() not in static_intrinsic_paths
        ],
        key=lambda path: (
            0
            if any(
                token in path.stem.lower()
                for token in ("moving", "calib", "intrinsic", "camera_info")
            )
            else 1,
            str(path).lower(),
        ),
    )
    options.extend(
        ("file", path, f"use local intrinsic file: {path}")
        for path in ordered_intrinsics
    )
    if (checkerboard_sources or has_local_images) and not ordered_intrinsics:
        options.append(
            (
                "calibrate",
                None,
                "calculate a reusable profile from checkerboard images/video "
                "(recommended: no bound intrinsics detected)",
            )
        )
    options.extend(
        (
            "profile",
            profile,
            (
                f"use profile {profile.label} [{profile.key}] "
                f"({profile.width}x{profile.height}, "
                f"{profile.distortion_model})"
            ),
        )
        for profile in profiles
    )
    if (checkerboard_sources or has_local_images) and ordered_intrinsics:
        options.append(
            (
                "calibrate",
                None,
                "calculate a new reusable profile from checkerboard images/video",
            )
        )
    if not options:
        raise RuntimeError(
            "No moving-camera intrinsics or compatible managed profile was "
            f"discovered below {input_root}. Add CameraInfo/intrinsics or "
            "checkerboard data below data_local and restart the wizard."
        )
    typer.echo("\nMoving intrinsics:")
    for index, (_, _, description) in enumerate(options, 1):
        typer.echo(f"  {index}. {description}")
    while True:
        selected_intrinsics = typer.prompt(
            "Selection", default=1, type=int
        )
        if 1 <= selected_intrinsics <= len(options):
            break
        typer.echo(f"Choose 1-{len(options)}")
    intrinsic_mode, selected_value, _ = options[selected_intrinsics - 1]
    if intrinsic_mode == "file":
        assert isinstance(selected_value, Path)
        intrinsics = selected_value
    elif intrinsic_mode == "profile":
        assert isinstance(selected_value, IntrinsicProfile)
        intrinsics = selected_value.intrinsics
        intrinsics_profile = selected_value.key
    else:
        calibration_video, calibration_images = _select_checkerboard_source(
            input_root,
            checkerboard_sources,
        )
        calibration_source = calibration_video or calibration_images
        assert calibration_source is not None
        if calibration_video is not None:
            _show_video_geometry(
                console,
                "Intrinsic video geometry",
                calibration_video,
            )
        profile_id = typer.prompt(
            "New intrinsics profile ID",
            default=safe_id(calibration_source.stem),
        ).strip()
    camera_id = (
        _camera_file_key(intrinsics)
        if intrinsics is not None
        else "moving_calib_camera"
    )
    target_hz = (
        typer.prompt("Video sampling rate [Hz]", default=3.0, type=float)
        if video is not None
        else _stored_prepared_sampling(input_root)[0]
    )
    start_seconds = 0.0
    end_seconds = None
    maximum_frames = None
    advanced_preparation = False
    calibration_requested = (
        calibration_video is not None or calibration_images is not None
    )
    intrinsic_scan = (
        _prompt_intrinsic_scan_settings()
        if calibration_requested
        else IntrinsicScanSettings()
    )
    if video is not None and calibration_requested:
        advanced_preparation = typer.confirm(
            "Open advanced video sampling and checkerboard settings?",
            default=False,
        )
    elif video is not None:
        advanced_preparation = typer.confirm(
            "Open advanced video sampling settings?", default=False
        )
    elif calibration_requested:
        advanced_preparation = typer.confirm(
            "Open advanced checkerboard calibration settings?", default=False
        )
    if video is not None and advanced_preparation:
        start_seconds = typer.prompt("Start time [s]", default=0.0, type=float)
        end_text = typer.prompt("End time [s] (blank = end of input)", default="").strip()
        maximum_text = typer.prompt(
            "Maximum frames (blank = unlimited)", default=""
        ).strip()
        end_seconds = float(end_text) if end_text else None
        maximum_frames = int(maximum_text) if maximum_text else None
    checkerboard_columns = 8
    checkerboard_rows = 6
    intrinsic_maximum_views = 80
    intrinsic_minimum_frame_gap = 0 if calibration_images is not None else 5
    intrinsic_minimum_detections = 20
    if calibration_requested and advanced_preparation:
        console.print("Checkerboard settings (press Enter for recommended defaults):")
        checkerboard_columns = typer.prompt("Inner corner columns", default=8, type=int)
        checkerboard_rows = typer.prompt("Inner corner rows", default=6, type=int)
        intrinsic_maximum_views = typer.prompt("Maximum selected views", default=80, type=int)
        intrinsic_minimum_frame_gap = typer.prompt(
            "Minimum frame gap",
            default=intrinsic_minimum_frame_gap,
            type=int,
        )
        intrinsic_minimum_detections = typer.prompt("Minimum detections", default=20, type=int)
        scan_target_hz = 3.0
        preview_max_dimension = 1920
        if intrinsic_scan.mode == "balanced":
            scan_target_hz = typer.prompt(
                "Initial checkerboard scan rate [Hz]",
                default=3.0,
                type=float,
            )
            preview_max_dimension = typer.prompt(
                "Preview maximum dimension [px]",
                default=1920,
                type=int,
            )
        intrinsic_scan = IntrinsicScanSettings(
            mode=intrinsic_scan.mode,
            target_hz=scan_target_hz,
            preview_max_dimension=preview_max_dimension,
        )
    elif calibration_requested:
        console.print(
            "Checkerboard settings: recommended defaults "
            f"(8x6 inner corners, max 80 views, frame gap "
            f"{intrinsic_minimum_frame_gap}, "
            "minimum 20 detections)."
        )
    moving = MovingCameraSettings(
        id=camera_id,
        video=video,
        frames=frames,
        intrinsics=intrinsics,
        intrinsics_profile=intrinsics_profile or profile_id,
        intrinsic_calibration_video=calibration_video,
        intrinsic_calibration_images=calibration_images,
        intrinsic_scan=intrinsic_scan,
        checkerboard_columns=checkerboard_columns,
        checkerboard_rows=checkerboard_rows,
        intrinsic_maximum_views=intrinsic_maximum_views,
        intrinsic_minimum_frame_gap=intrinsic_minimum_frame_gap,
        intrinsic_minimum_detections=intrinsic_minimum_detections,
    )
    return (
        moving,
        SamplingSettings(
            target_hz=target_hz,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            maximum_frames=maximum_frames,
        ),
        suggested_id,
    )


def _camera_file_key(path: Path) -> str:
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            declared = payload.get(
                "camera_name", payload.get("camera_id", payload.get("name"))
            )
            if isinstance(declared, str) and declared.strip():
                return safe_id(declared.lower())
    value = path.stem.lower()
    for suffix in (
        "_intrinsics",
        "-intrinsics",
        "_camera_info",
        "-camera-info",
        "_calibration",
        "-calibration",
    ):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return safe_id(value)


def _detected_static_pairs(
    input_root: Path, moving: MovingCameraSettings
) -> list[tuple[Path, Path]]:
    discovered = discover_inputs(input_root, recursive=True)
    images = [item.path for item in discovered if item.kind == "image"]
    if moving.frames is not None:
        images = [
            path for path in images if not path.is_relative_to(moving.frames.resolve())
        ]
    images = [
        path
        for path in images
        if not any(
            token in path.parent.name.lower()
            for token in ("moving", "frame", "extract", "checker")
        )
    ]
    intrinsics = [item.path for item in discovered if item.kind == "intrinsics"]
    by_key: dict[str, list[Path]] = {}
    for path in intrinsics:
        by_key.setdefault(_camera_file_key(path), []).append(path)
    pairs: list[tuple[Path, Path]] = [
        (image, by_key[_camera_file_key(image)][0])
        for image in images
        if _camera_file_key(image) in by_key
    ]
    if not pairs and len(images) == 1 and len(intrinsics) == 1:
        pairs = [(images[0], intrinsics[0])]
    return pairs


def _static_group_key(image: Path, input_root: Path) -> str:
    relative = image.resolve().relative_to(input_root.resolve())
    directory_parts = list(relative.parts[:-1])
    static_index = next(
        (
            index
            for index, part in enumerate(directory_parts)
            if "static" in safe_id(part).lower()
        ),
        None,
    )
    if static_index is not None and static_index + 1 < len(directory_parts):
        return safe_id(directory_parts[static_index + 1])
    return _camera_file_key(image)


def _detected_static_camera_groups(
    input_root: Path,
    moving: MovingCameraSettings,
) -> list[tuple[str, list[Path], Path | None, Path]]:
    discovered = discover_inputs(input_root, recursive=True)
    role_directories = discover_image_directories(input_root)
    static_directories = role_directories["static"]
    excluded_directories = {
        path.resolve()
        for role in ("moving", "checkerboard")
        for path in role_directories[role]
    }
    images = [
        item.path
        for item in discovered
        if item.kind == "image"
        and item.path.parent.resolve() not in excluded_directories
        and (
            static_directories
            and any(
                item.path.resolve().is_relative_to(directory)
                for directory in static_directories
            )
            or not static_directories
        )
    ]
    if moving.frames is not None:
        images = [
            path
            for path in images
            if not path.is_relative_to(moving.frames.resolve())
        ]
    grouped_images: dict[str, list[Path]] = {}
    for image in images:
        grouped_images.setdefault(
            _static_group_key(image, input_root), []
        ).append(image)
    grouped_videos: dict[str, list[Path]] = {}
    for item in discovered:
        if (
            item.kind == "video"
            and media_path_role(item.path, input_root) == "static"
        ):
            grouped_videos.setdefault(
                _static_group_key(item.path, input_root), []
            ).append(item.path)
    intrinsic_paths = [
        item.path for item in discovered if item.kind == "intrinsics"
    ]
    by_key: dict[str, list[Path]] = {}
    for path in intrinsic_paths:
        by_key.setdefault(_camera_file_key(path), []).append(path)
    result: list[tuple[str, list[Path], Path | None, Path]] = []
    camera_ids = sorted(set(grouped_images).union(grouped_videos))
    for camera_id in camera_ids:
        camera_images = sorted(grouped_images.get(camera_id, []))
        camera_videos = sorted(grouped_videos.get(camera_id, []))
        # Explicit images avoid decoding when both representations were supplied.
        camera_video = None if camera_images else (
            camera_videos[0] if len(camera_videos) == 1 else None
        )
        if not camera_images and camera_video is None:
            continue
        matching = by_key.get(camera_id, [])
        if not matching:
            camera_directories = {
                path.parent.resolve()
                for path in [*camera_images, *camera_videos]
            }
            matching = [
                path
                for path in intrinsic_paths
                if path.parent.resolve() in camera_directories
            ]
        if len(matching) == 1:
            result.append(
                (
                    camera_id,
                    camera_images,
                    camera_video,
                    matching[0],
                )
            )
    return result


def _direct_static_cameras(
    console: Console, input_root: Path, moving: MovingCameraSettings
) -> list[StaticCameraSettings]:
    groups = _detected_static_camera_groups(input_root, moving)
    if groups:
        table = Table(title="Detected direct static camera inputs")
        table.add_column("#", justify="right")
        table.add_column("Camera suggestion")
        table.add_column("Media")
        table.add_column("Intrinsics")
        for index, (camera_id, images, video, camera_info) in enumerate(groups, 1):
            table.add_row(
                str(index),
                camera_id,
                (
                    str(images[0].relative_to(input_root))
                    if len(images) == 1
                    else f"{images[0].parent.relative_to(input_root)} ({len(images)})"
                    if images
                    else f"video: {video.relative_to(input_root)}"
                ),
                str(camera_info.relative_to(input_root)),
            )
        console.print(table)
        keys = [camera_id for camera_id, _, _, _ in groups]
        if len(keys) == len(set(keys)):
            cameras = [
                StaticCameraSettings(
                    id=camera_id,
                    images=images,
                    video=video,
                    intrinsics=camera_info,
                )
                for camera_id, images, video, camera_info in groups
            ]
            console.print(
                "[green]Camera IDs are unambiguous; no confirmation is required.[/green]"
            )
            return cameras
        console.print(
            "Duplicate camera basenames make the binding ambiguous; choose the "
            "static pairs and IDs explicitly."
        )
        selected = typer.prompt(
            "Static camera numbers (comma-separated)",
            default=",".join(str(index) for index in range(1, len(groups) + 1)),
        )
        indices = [
            int(value.strip()) - 1
            for value in selected.split(",")
            if value.strip()
        ]
        if any(index < 0 or index >= len(groups) for index in indices):
            raise typer.BadParameter("Invalid static camera number")
        cameras = []
        for index in indices:
            camera_id, images, video, camera_info = groups[index]
            camera_id = typer.prompt(
                (
                    f"Camera ID for {images[0].parent.name}"
                    if images
                    else f"Camera ID for {video.name}"
                ),
                default=camera_id,
            ).strip()
            cameras.append(
                StaticCameraSettings(
                    id=camera_id,
                    images=images,
                    video=video,
                    intrinsics=camera_info,
                )
            )

        return cameras
    raise RuntimeError(
        "No unambiguous static image/intrinsics pairs were discovered below "
        f"{input_root}. Add matching media and CameraInfo files below data_local "
        "and restart the wizard."
    )


def _relative_display(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path)


def _show_input_inventory(
    repository_root: Path,
    console: Console,
    prepared: list[PreparedDatasetSummary],
    raw_inputs: list[RawInputSummary],
) -> None:
    console.print(
        Panel(
            "Prepared means that frames and intrinsics already exist in the canonical "
            "raw_images/{static,moving,camera_info} layout. Choosing one reuses those "
            "files; no video extraction or Gazebo capture is performed. Camera IDs are "
            "accepted without a prompt when the manifest or exact image/intrinsic "
            "basenames bind them uniquely.",
            title="What is a prepared dataset?",
        )
    )
    real_prepared = [item for item in prepared if item.category == "real_vehicle"]
    prepared_table = Table(title="Existing real-vehicle prepared datasets")
    prepared_table.add_column("#", justify="right")
    prepared_table.add_column("Dataset")
    prepared_table.add_column("Meaning")
    prepared_table.add_column("Cameras")
    prepared_table.add_column("Moving frames", justify="right")
    prepared_table.add_column("Results")
    for index, item in enumerate(real_prepared, 1):
        prepared_table.add_row(
            str(index),
            item.display_name,
            item.description,
            str(len(item.static_camera_ids)),
            str(item.moving_frames),
            "available" if item.has_results else "input only",
        )
    if real_prepared:
        console.print(prepared_table)
    else:
        console.print("No prepared real-vehicle dataset was detected.")

    raw_table = Table(title="Local raw input (data_local)")
    raw_table.add_column("Folder")
    raw_table.add_column("Videos", justify="right")
    raw_table.add_column("Images", justify="right")
    raw_table.add_column("Intrinsics", justify="right")
    raw_table.add_column("MCAP/DB3", justify="right")
    if raw_inputs:
        raw_table.add_row(
            "data_local",
            str(sum(item.videos for item in raw_inputs)),
            str(sum(item.images for item in raw_inputs)),
            str(sum(item.intrinsics for item in raw_inputs)),
            str(sum(item.recordings for item in raw_inputs)),
        )
        console.print(raw_table)
    else:
        console.print(
            Panel(
                f"Put every file for one recording anywhere below:\n"
                f"{repository_root / 'data_local'}\n\n"
                "Subfolders are optional. rigcal scans recursively for moving videos, "
                "frame folders, static images, YAML/JSON intrinsics, checkerboard "
                "videos, and .mcap/.db3 ROS recordings.",
                title="No local real-data input detected",
            )
        )
    console.print(
        Panel(
            "Recommended zero-config layout:\n"
            "data_local/<dataset-id>/moving_frames/frame_*.png\n"
            "data_local/<dataset-id>/static/<camera-id>.png\n"
            "data_local/<dataset-id>/intrinsics/<camera-id>.yaml\n"
            "data_local/<dataset-id>/intrinsics/moving_calib_camera.yaml\n\n"
            "A moving video may replace moving_frames/. Static image and intrinsic "
            "basenames must match. Files are still scanned recursively, so the "
            "subfolders are a clear convention rather than hard-coded camera names.",
            title="Direct videos, frames and PNG/JPG inputs",
        )
    )


def _show_prepared_choices(
    repository_root: Path,
    console: Console,
    prepared: list[PreparedDatasetSummary],
    *,
    title: str,
) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Dataset")
    table.add_column("Meaning")
    table.add_column("Cameras", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Results")
    for index, item in enumerate(prepared, 1):
        table.add_row(
            str(index),
            item.display_name,
            item.description,
            str(len(item.static_camera_ids)),
            str(item.moving_frames),
            "available" if item.has_results else "input only",
        )
    if prepared:
        console.print(table)


def _prepared_input(
    console: Console,
    repository_root: Path,
    prepared: list[PreparedDatasetSummary],
) -> tuple[Path, list[StaticCameraSettings], MovingCameraSettings, str]:
    if not prepared:
        raise RuntimeError(
            "No prepared dataset was discovered. Place canonical inputs below "
            "results/<category>/<rate-or-factor>/<experiment>/ or add raw "
            "input below data_local/, then restart the wizard."
        )
    selected = typer.prompt(
        "Prepared dataset number (0 = back)",
        default=1,
        type=int,
    )
    if selected == 0:
        raise WizardBack()
    if selected < 1 or selected > len(prepared):
        raise typer.BadParameter("Invalid prepared dataset number")
    root = prepared[selected - 1].path
    selected_summary = prepared[selected - 1]
    inspection = inspect_prepared_dataset(root)
    detected_ids = list(inspection["static_camera_ids"])
    intrinsic_ids = sorted(inspection["intrinsic_ids"])
    remaining_intrinsics = sorted(set(intrinsic_ids) - set(detected_ids))
    manifest_static = list(inspection["manifest_static_camera_ids"])
    manifest_moving = inspection["manifest_moving_camera_id"]
    manifest_unambiguous = (
        bool(manifest_static)
        and bool(manifest_moving)
        and set(manifest_static).issubset(set(intrinsic_ids))
        and str(manifest_moving) in intrinsic_ids
        and not set(manifest_static).intersection({str(manifest_moving)})
    )
    basename_unambiguous = (
        bool(detected_ids)
        and set(detected_ids).issubset(set(intrinsic_ids))
        and len(remaining_intrinsics) == 1
    )
    unambiguous = manifest_unambiguous or basename_unambiguous
    if manifest_unambiguous:
        detected_ids = manifest_static
        remaining_intrinsics = [str(manifest_moving)]
    sampling_hz, sampling_source = _stored_prepared_sampling(root)
    console.print(
        Panel(
            f"Dataset: {selected_summary.display_name if selected_summary else root.name}\n"
            f"Full source: {root}\n"
            f"Static cameras: {len(detected_ids)}\n"
            f"  {', '.join(detected_ids) or 'none'}\n"
            f"Moving cameras: {1 if unambiguous else '?'}\n"
            f"  {remaining_intrinsics[0] if unambiguous else 'ambiguous'}\n"
            f"Moving frames: {len(inspection['moving_frames'])}\n"
            f"Sampling: "
            f"{f'{sampling_hz:g} Hz ({sampling_source})' if sampling_hz is not None else 'unknown'}\n\n"
            "IDs are derived from the manifest or exact image/intrinsic basename "
            "matches. No image-content identity is guessed.",
            title="Selected prepared dataset",
        )
    )
    if unambiguous:
        camera_ids = detected_ids
        moving_id = remaining_intrinsics[0]
    else:
        console.print(
            Panel(
                "Automatic camera binding is ambiguous. Every static image must "
                "have one intrinsic with the same basename and exactly one "
                "intrinsic must remain for the moving camera.",
                title="Camera binding needs review",
            )
        )
        value = typer.prompt(
            "Static camera IDs (comma-separated)",
            default=",".join(detected_ids),
        )
        camera_ids = [item.strip() for item in value.split(",") if item.strip()]
        if not camera_ids:
            raise typer.BadParameter("At least one static camera ID is required")
        candidates = sorted(set(intrinsic_ids) - set(camera_ids))
        moving_id = typer.prompt(
            "Moving camera ID",
            default=candidates[0] if len(candidates) == 1 else "moving_calib_camera",
        ).strip()
    cameras = [StaticCameraSettings(id=camera_id) for camera_id in camera_ids]
    suggested_id = selected_summary.id if selected_summary else safe_id(root.name)
    return root, cameras, MovingCameraSettings(id=moving_id), suggested_id


__all__ = [
    '_moving_source',
    '_camera_file_key',
    '_detected_static_pairs',
    '_static_group_key',
    '_detected_static_camera_groups',
    '_direct_static_cameras',
    '_relative_display',
    '_show_input_inventory',
    '_show_prepared_choices',
    '_prepared_input',
]
