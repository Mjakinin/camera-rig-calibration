from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .components import register_builtin_components
from .config import config_fingerprint, load_config, save_config
from .config.models import (
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
    SimulationSettings,
    StaticCameraSettings,
)
from .dataset.discovery import (
    IMAGE_SUFFIXES,
    discover_image_directories,
    discover_inputs,
    inspect_prepared_dataset,
    media_path_role,
    safe_id,
)
from .doctor import run_checks
from .input.topics import McapTopic, list_mcap_topics
from .intrinsics_profiles import (
    IntrinsicProfile,
    delete_profile,
    discover_intrinsic_profiles,
    intrinsic_dimensions,
    intrinsic_profile_references,
    update_profile_alias,
)
from .inventory import (
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
from .registry import (
    calibration_methods,
    experiment_providers,
    input_adapters,
)
from .results import ResultEntry, index_results
from .runtime import PipelineOrchestrator
from .observations import ResolvedSelections
from .simulation_worlds import (
    SimulationWorldManifest,
    discover_world_manifests,
    install_world_manifest,
    load_world_manifest,
)
from .storage import (
    build_cleanup_plan,
    build_data_local_cleanup_plan,
    execute_cleanup,
)


@dataclass(frozen=True)
class WizardOutcome:
    config: RigConfig
    path: Path
    queued_runs: tuple["QueuedRun", ...] = ()

    @property
    def runs(self) -> tuple["QueuedRun", ...]:
        return (QueuedRun(self.config, self.path), *self.queued_runs)


@dataclass(frozen=True)
class QueuedRun:
    config: RigConfig
    path: Path


@dataclass
class MethodQueueJob:
    method_id: str
    label: str
    methods: MethodSettings
    markers: MarkerSettings
    observation_quality: ObservationQualitySettings
    colmap: ColmapSettings
    evaluation: EvaluationSettings


class WizardBack(Exception):
    """Return from a nested selection to the previous wizard menu."""


def _clear_terminal() -> None:
    if sys.stdin.isatty() and sys.stdout.isatty():
        typer.echo("\033[2J\033[H", nl=False)


def _show_input_error(message: str) -> None:
    typer.echo(f"Error: {message}")
    if sys.stdin.isatty():
        typer.prompt("Press Enter to continue", default="", show_default=False)


def _prompt_index(
    label: str,
    *,
    default: int | None = None,
    minimum: int = 1,
    maximum: int | None = None,
) -> int | None:
    """Prompt for an integer while treating 0/b/back as navigation."""
    while True:
        raw = str(
            typer.prompt(
                label,
                default=default,
                show_default=default is not None,
            )
        ).strip()
        if raw.lower() in {"0", "b", "back"}:
            _clear_terminal()
            return None
        try:
            value = int(raw)
        except ValueError:
            _show_input_error("Enter a number, or b to go back.")
            continue
        if value < minimum or (
            maximum is not None and value > maximum
        ):
            _show_input_error(
                f"Choose a number from {minimum}"
                + (f" to {maximum}" if maximum is not None else " upward")
                + "."
            )
            continue
        return value


def _simulation_experiment_id(parameters: dict[str, object]) -> str:
    """Build a stable readable experiment ID from deviations from baseline."""
    deviations: list[str] = []
    route = str(parameters.get("route", "route2"))
    if route != str(BASELINE_SIMULATION_PARAMETERS["route"]):
        deviations.append(
            route if route.startswith("route") else f"route_{route}"
        )
    width = int(parameters.get("moving_width", 1280))
    height = int(parameters.get("moving_height", 720))
    if (
        width != BASELINE_SIMULATION_PARAMETERS["moving_width"]
        or height != BASELINE_SIMULATION_PARAMETERS["moving_height"]
    ):
        deviations.append(f"res_{width}x{height}")
    fov = float(parameters.get("moving_hfov_deg", 69.1))
    if fov != float(
        BASELINE_SIMULATION_PARAMETERS["moving_hfov_deg"]
    ):
        deviations.append(f"fov_{fov:g}deg")
    lighting = str(parameters.get("lighting", "baseline"))
    lighting_scale = float(parameters.get("lighting_scale", 1.0))
    if lighting != "baseline" or lighting_scale != 1.0:
        deviations.append(
            f"light_{lighting}_{lighting_scale:g}x"
        )
    blur = int(parameters.get("motion_blur_kernel", 0))
    blur_angle = float(parameters.get("motion_blur_angle_deg", 0.0))
    if blur or blur_angle:
        deviations.append(f"blur_k{blur}_{blur_angle:g}deg")
    frames = int(parameters.get("target_route_frames", 189))
    if frames != int(
        BASELINE_SIMULATION_PARAMETERS["target_route_frames"]
    ):
        deviations.append(f"frames_{frames}")
    sampling = str(
        parameters.get(
            "route_sampling_strategy", "original_route_poses"
        )
    )
    if sampling != "original_route_poses":
        deviations.append(f"sampling_{sampling}")
    settle = float(parameters.get("settle_seconds", 0.35))
    if settle != 0.35:
        deviations.append(f"settle_{settle:g}s")
    skip = int(parameters.get("post_pose_skip", 5))
    if skip != 5:
        deviations.append(f"skip_{skip}")
    frame_timeout = float(
        parameters.get("frame_timeout_seconds", 3.0)
    )
    if frame_timeout != 3.0:
        deviations.append(f"frame_timeout_{frame_timeout:g}s")
    startup_timeout = float(
        parameters.get("startup_timeout_seconds", 60.0)
    )
    if startup_timeout != 60.0:
        deviations.append(f"startup_timeout_{startup_timeout:g}s")
    if not deviations:
        return "route2"
    readable = "__".join(deviations)
    if len(readable) <= 72:
        return safe_id(readable)
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:8]
    return safe_id(f"{readable[:63]}_{digest}")


def _prompt_path(
    label: str,
    default: Path | None = None,
    *,
    directory: bool | None = None,
    allow_back: bool = True,
) -> Path:
    default_text = str(default) if default is not None else None
    while True:
        suffix = " (0 = back)" if allow_back else ""
        value = typer.prompt(
            label + suffix,
            default=default_text,
            show_default=default_text is not None,
        )
        if allow_back and str(value).strip().lower() in {"0", "b", "back"}:
            raise WizardBack()
        path = Path(value).expanduser().resolve()
        valid = path.is_dir() if directory is True else path.is_file() if directory is False else path.exists()
        if valid:
            return path
        typer.echo(f"Path does not exist or has the wrong type: {path}")


def _choice(label: str, choices: dict[str, str], default: str) -> str:
    typer.echo(f"\n{label}:")
    for key, description in choices.items():
        typer.echo(f"  {key}. {description}")
    while True:
        value = typer.prompt("Selection", default=default).strip()
        if value.lower() in {"b", "back"}:
            _clear_terminal()
            if "0" in choices:
                return "0"
            raise WizardBack()
        if value in choices:
            return value
        _show_input_error("Choose one of: " + ", ".join(choices))


def _select_detected_path(
    label: str,
    paths: list[Path],
    *,
    directory: bool,
    preferred: int = 0,
) -> Path:
    if not paths:
        return _prompt_path(label, directory=directory)
    typer.echo(f"\n{label}:")
    for index, path in enumerate(paths, 1):
        typer.echo(f"  {index}. {path}")
    typer.echo("  0. enter another path")
    selected = _prompt_index(
        "Selection",
        default=preferred + 1,
        maximum=len(paths),
    )
    if selected is None:
        return _prompt_path(label, directory=directory)
    return paths[selected - 1]


def _preferred_path(paths: list[Path], tokens: tuple[str, ...]) -> int:
    for index, path in enumerate(paths):
        if any(token in path.stem.lower() for token in tokens):
            return index
    return 0


def _looks_like_checkerboard_video(
    path: Path, input_root: Path | None = None
) -> bool:
    if (
        input_root is not None
        and media_path_role(path, input_root) == "checkerboard"
    ):
        return True
    searchable = "/".join(part.lower() for part in path.parts[-4:])
    return any(
        token in searchable
        for token in ("checker", "chess", "intrinsic", "calibration")
    )


def _checkerboard_sources(
    input_root: Path,
) -> list[tuple[str, Path, str]]:
    discovered = discover_inputs(input_root, recursive=True)
    videos = [
        item.path
        for item in discovered
        if item.kind == "video"
        and _looks_like_checkerboard_video(item.path, input_root)
    ]
    image_directories = discover_image_directories(input_root)["checkerboard"]
    sources = [
        ("video", path, f"video: {path}") for path in videos
    ]
    for path in image_directories:
        count = sum(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            for child in path.iterdir()
        )
        sources.append(
            ("images", path, f"image folder ({count} images): {path}")
        )
    return sources


def _select_checkerboard_source(
    input_root: Path,
    sources: list[tuple[str, Path, str]] | None = None,
) -> tuple[Path | None, Path | None]:
    sources = sources if sources is not None else _checkerboard_sources(input_root)
    if not sources:
        typer.echo(
            "\nNo named checkerboard source was detected; enter a video or "
            "image-folder path."
        )
        path = _prompt_path("Checkerboard video or image folder")
        return (None, path) if path.is_dir() else (path, None)
    if len(sources) == 1:
        kind, path, description = sources[0]
        typer.echo(f"\nCheckerboard calibration input: {description} (automatic)")
        return (path, None) if kind == "video" else (None, path)
    typer.echo("\nCheckerboard calibration input:")
    for index, (_, _, description) in enumerate(sources, 1):
        typer.echo(f"  {index}. {description}")
    typer.echo("  0. enter another path")
    selected = _prompt_index(
        "Selection",
        default=1,
        maximum=len(sources),
    )
    if selected is None:
        path = _prompt_path("Checkerboard video or image folder")
        return (None, path) if path.is_dir() else (path, None)
    kind, path, _ = sources[selected - 1]
    return (path, None) if kind == "video" else (None, path)


def _moving_media_dimensions(
    video: Path | None, frames: Path | None
) -> tuple[int, int] | None:
    try:
        import cv2
    except ImportError:
        return None
    if video is not None:
        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return (width, height) if width > 0 and height > 0 else None
    if frames is not None:
        candidates = [
            path
            for path in sorted(frames.iterdir())
            if path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        ]
        if candidates:
            image = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                height, width = image.shape[:2]
                return int(width), int(height)
    return None


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
        options.append(("file_manual", None, "enter an intrinsic file path"))
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
    elif intrinsic_mode == "file_manual":
        intrinsics = _prompt_path(
            "Moving-camera intrinsics JSON/YAML", directory=False
        )
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
    intrinsic_scan = IntrinsicScanSettings()
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
        scan_mode = _prompt_enum_choice(
            "Checkerboard scan mode",
            "balanced",
            (
                (
                    "balanced",
                    "3/6/12 Hz preview detection with full-resolution refinement",
                ),
                (
                    "exhaustive_compatibility",
                    "legacy every-frame full-resolution detector",
                ),
            ),
        )
        scan_target_hz = 3.0
        preview_max_dimension = 1920
        if scan_mode == "balanced":
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
            mode=scan_mode,
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

    count = typer.prompt("Number of static cameras", type=int)
    if count < 1:
        raise typer.BadParameter("At least one static camera is required")
    cameras = cameras if groups else []
    for index in range(count):
        console.print(f"\nStatic camera {index + 1}/{count}")
        camera_id = typer.prompt("Camera ID", default=f"camera_{index + 1}").strip()
        image = _prompt_path("Static image", directory=False)
        intrinsics = _prompt_path("Static intrinsics JSON/YAML", directory=False)
        cameras.append(
            StaticCameraSettings(id=camera_id, images=[image], intrinsics=intrinsics)
        )
    return cameras


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

    raw_table = Table(title="Local raw input folders (data_local)")
    raw_table.add_column("#", justify="right")
    raw_table.add_column("Folder")
    raw_table.add_column("Videos", justify="right")
    raw_table.add_column("Images", justify="right")
    raw_table.add_column("Intrinsics", justify="right")
    raw_table.add_column("MCAP/DB3", justify="right")
    for index, item in enumerate(raw_inputs, 1):
        raw_table.add_row(
            str(index),
            _relative_display(item.path, repository_root),
            str(item.videos),
            str(item.images),
            str(item.intrinsics),
            str(item.recordings),
        )
    if raw_inputs:
        console.print(raw_table)
    else:
        console.print(
            Panel(
                f"Put every file for one recording anywhere below:\n"
                f"{repository_root / 'data_local' / '<dataset-id>'}\n\n"
                "Subfolders are optional. rigcal scans recursively for moving videos, "
                "frame folders, static images, YAML/JSON intrinsics, checkerboard "
                "videos, and .mcap/.db3 ROS recordings.",
                title="No local real-data folders detected",
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
    if prepared:
        selected = typer.prompt(
            "Prepared dataset number (0 = back, -1 = another path)",
            default=1,
            type=int,
        )
        if selected == 0:
            raise WizardBack()
        if selected < -1 or selected > len(prepared):
            raise typer.BadParameter("Invalid prepared dataset number")
        root = (
            prepared[selected - 1].path
            if selected > 0
            else _prompt_path("Prepared dataset directory", Path.cwd(), directory=True)
        )
        selected_summary = prepared[selected - 1] if selected > 0 else None
    else:
        root = _prompt_path("Prepared dataset directory", Path.cwd(), directory=True)
        selected_summary = None
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


def _prepared_moving_intrinsics(
    console: Console,
    repository_root: Path,
    prepared_root: Path,
    moving: MovingCameraSettings,
) -> MovingCameraSettings:
    current = (
        prepared_root
        / "raw_images"
        / "camera_info"
        / f"{moving.id}.json"
    )
    current_dimensions = (
        intrinsic_dimensions(current) if current.is_file() else (0, 0)
    )
    profiles = discover_intrinsic_profiles(repository_root)
    compatible = [
        profile
        for profile in profiles
        if current_dimensions == (0, 0)
        or (profile.width, profile.height) == current_dimensions
    ]
    checkerboard_videos = [
        item.path
        for item in discover_inputs(
            repository_root / "data_local", recursive=True
        )
        if item.kind == "video"
    ]
    checkerboard_image_directories = discover_image_directories(
        repository_root / "data_local"
    )["checkerboard"]
    local_images_exist = any(
        item.kind == "image"
        for item in discover_inputs(
            repository_root / "data_local", recursive=True
        )
    )
    options: list[tuple[str, object | None, str]] = []
    if current.is_file():
        options.append(
            (
                "prepared",
                current,
                (
                    "use the intrinsics already stored with this prepared "
                    f"dataset ({current_dimensions[0]}x{current_dimensions[1]})"
                ),
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
        for profile in compatible
        if profile.intrinsics.resolve() != current.resolve()
    )
    if checkerboard_videos or checkerboard_image_directories or local_images_exist:
        options.append(
            (
                "calibrate",
                None,
                "calculate a new profile from data_local checkerboard images/video",
            )
        )
    options.append(("manual", None, "enter another intrinsic file path"))
    typer.echo("\nMoving intrinsics for the selected prepared frames:")
    for index, (_, _, description) in enumerate(options, 1):
        typer.echo(f"  {index}. {description}")
    while True:
        selection = typer.prompt("Selection", default=1, type=int)
        if 1 <= selection <= len(options):
            break
        typer.echo(f"Choose 1-{len(options)}")
    mode, value, _ = options[selection - 1]
    if mode == "prepared":
        return moving
    if mode == "profile":
        assert isinstance(value, IntrinsicProfile)
        return moving.model_copy(
            update={
                "intrinsics": value.intrinsics,
                "intrinsics_profile": value.key,
            },
            deep=True,
        )
    if mode == "manual":
        path = _prompt_path(
            "Moving-camera intrinsics JSON/YAML", directory=False
        )
        return moving.model_copy(update={"intrinsics": path}, deep=True)

    calibration_sources = [
        ("video", path, f"video: {path}") for path in checkerboard_videos
    ]
    calibration_sources.extend(
        (
            "images",
            path,
            f"image folder: {path}",
        )
        for path in checkerboard_image_directories
    )
    calibration_video, calibration_images = _select_checkerboard_source(
        repository_root / "data_local",
        calibration_sources,
    )
    calibration_source = calibration_video or calibration_images
    assert calibration_source is not None
    profile_id = typer.prompt(
        "New intrinsics profile ID",
        default=safe_id(calibration_source.stem),
    ).strip()
    updates: dict[str, object] = {
        "intrinsic_calibration_video": calibration_video,
        "intrinsic_calibration_images": calibration_images,
        "intrinsics_profile": profile_id,
        "intrinsic_minimum_frame_gap": (
            0 if calibration_images is not None else 5
        ),
    }
    if typer.confirm(
        "Open advanced checkerboard calibration settings?", default=False
    ):
        updates.update(
            {
                "checkerboard_columns": typer.prompt(
                    "Inner corner columns", default=8, type=int
                ),
                "checkerboard_rows": typer.prompt(
                    "Inner corner rows", default=6, type=int
                ),
                "intrinsic_maximum_views": typer.prompt(
                    "Maximum selected views", default=80, type=int
                ),
                "intrinsic_minimum_frame_gap": typer.prompt(
                    "Minimum frame gap",
                    default=0 if calibration_images is not None else 5,
                    type=int,
                ),
                "intrinsic_minimum_detections": typer.prompt(
                    "Minimum detections", default=20, type=int
                ),
            }
        )
        scan_mode = _prompt_enum_choice(
            "Checkerboard scan mode",
            "balanced",
            (
                ("balanced", "fast adaptive scan with 4K refinement"),
                (
                    "exhaustive_compatibility",
                    "legacy every-frame full-resolution scan",
                ),
            ),
        )
        target_hz = 3.0
        preview_dimension = 1920
        if scan_mode == "balanced":
            target_hz = typer.prompt(
                "Initial checkerboard scan rate [Hz]",
                default=3.0,
                type=float,
            )
            preview_dimension = typer.prompt(
                "Preview maximum dimension [px]",
                default=1920,
                type=int,
            )
        updates["intrinsic_scan"] = IntrinsicScanSettings(
            mode=scan_mode,
            target_hz=target_hz,
            preview_max_dimension=preview_dimension,
        )
    else:
        console.print(
            "Checkerboard settings: balanced 3/6/12 Hz scan, 1920 px "
            "preview, full-resolution refinement, 8x6 board, max 80 views."
        )
    return moving.model_copy(update=updates, deep=True)


def _stored_prepared_sampling(root: Path) -> tuple[float | None, str]:
    candidates = [
        root / "dataset_manifest.json",
        root / "metadata" / "moving_video_extraction" / "MOVING_VIDEO_EXTRACTION.json",
        root / "raw_images" / "metadata" / "moving_video_extraction" / "MOVING_VIDEO_EXTRACTION.json",
    ]
    keys = (
        "sampling_hz",
        "target_hz",
        "requested_hz",
        "output_hz",
        "moving_sampling_hz",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in keys:
            value = payload.get(key) if isinstance(payload, dict) else None
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number, "stored metadata"
    return None, "unknown"


def _choose_raw_input_root(
    console: Console,
    raw_inputs: list[RawInputSummary],
    repository_root: Path,
) -> Path:
    if not raw_inputs:
        landing = repository_root / "data_local"
        landing.mkdir(parents=True, exist_ok=True)
        return _prompt_path("Folder containing the real-data files", landing, directory=True)
    selected = typer.prompt(
        "Local input folder number (0 = back, -1 = another path)",
        default=1,
        type=int,
    )
    if selected == 0:
        raise WizardBack()
    if selected < -1 or selected > len(raw_inputs):
        raise typer.BadParameter("Invalid local input folder number")
    return (
        raw_inputs[selected - 1].path
        if selected > 0
        else _prompt_path("Input directory", Path.cwd(), directory=True)
    )


def _ros_image_stream_prefix(topic_name: str) -> str:
    normalized = topic_name.rstrip("/")
    for suffix in (
        "/image_rect_raw/compressed",
        "/image_raw/compressed",
        "/image_rect/compressed",
        "/image/compressed",
        "/image_rect_raw",
        "/image_raw",
        "/image_rect",
        "/image",
    ):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized.rsplit("/", 1)[0]


def _related_camera_info_topics(
    image_topic: McapTopic, info_topics: list[McapTopic]
) -> list[str]:
    prefix = _ros_image_stream_prefix(image_topic.name)
    return [
        item.name
        for item in info_topics
        if item.name.rstrip("/") == f"{prefix}/camera_info"
    ]


def _camera_id_from_ros_topic(topic_name: str) -> str:
    parts = [part for part in _ros_image_stream_prefix(topic_name).split("/") if part]
    if "camera" in parts:
        camera_index = parts.index("camera")
        candidate = parts[camera_index - 1] if camera_index > 0 else parts[0]
    else:
        candidate = parts[-1] if parts else "static_camera"
    if re.fullmatch(r"edge[_-]?\d+", candidate, flags=re.IGNORECASE):
        candidate = f"cam_{candidate}"
    return safe_id(candidate)


def _mcap_camera_sources(
    console: Console, mcap: Path, *, moving_from_recording: bool
) -> tuple[list[StaticCameraSettings], MovingCameraSettings | None]:
    topics = list_mcap_topics(mcap)
    image_topics = sorted(
        (topic for topic in topics if topic.is_image),
        key=lambda topic: topic.name,
    )
    info_topics = sorted(
        (topic for topic in topics if topic.is_camera_info),
        key=lambda topic: topic.name,
    )
    table = Table(title="MCAP camera topics")
    table.add_column("#")
    table.add_column("Topic")
    table.add_column("Type")
    table.add_column("Suggested role")
    for index, topic in enumerate(image_topics, 1):
        lower_name = topic.name.lower()
        role = (
            "static color (recommended)"
            if "/color/" in lower_name and "/depth/" not in lower_name
            else "depth / auxiliary"
            if "/depth/" in lower_name
            else "image"
        )
        table.add_row(str(index), topic.name, topic.message_type, role)
    console.print(table)
    if not image_topics:
        raise RuntimeError("The selected MCAP contains no supported image topics")
    moving: MovingCameraSettings | None = None
    moving_index: int | None = None
    if moving_from_recording:
        preferred = next(
            (
                index
                for index, topic in enumerate(image_topics)
                if any(token in topic.name.lower() for token in ("moving", "calib"))
            ),
            0,
        )
        moving_number = typer.prompt(
            "Moving-camera image topic number",
            default=preferred + 1,
            type=int,
        )
        if moving_number < 1 or moving_number > len(image_topics):
            raise typer.BadParameter("Invalid moving-camera topic number")
        moving_index = moving_number - 1
        moving_topic = image_topics[moving_index]
        moving_id = typer.prompt(
            "Moving camera ID", default="moving_calib_camera"
        ).strip()
        related = _related_camera_info_topics(moving_topic, info_topics)
        moving_info = related[0] if len(related) == 1 else typer.prompt(
            "Moving-camera CameraInfo topic", default=""
        ).strip()
        moving_intrinsics = None
        if not moving_info:
            moving_intrinsics = _prompt_path(
                f"External intrinsics for {moving_id}", directory=False
            )
        moving = MovingCameraSettings(
            id=moving_id,
            intrinsics=moving_intrinsics,
            image_topic=moving_topic.name,
            camera_info_topic=moving_info or None,
        )
    static_indices = [
        index for index in range(len(image_topics)) if index != moving_index
    ]
    if not static_indices:
        raise RuntimeError("At least one static image topic must remain in the ROS bag")
    preferred_static_indices = [
        index
        for index in static_indices
        if "/color/" in image_topics[index].name.lower()
        and "/depth/" not in image_topics[index].name.lower()
    ]
    default_static_indices = preferred_static_indices or static_indices
    selected = typer.prompt(
        "Static image topic numbers (comma-separated)",
        default=",".join(str(index + 1) for index in default_static_indices),
    )
    indices = [int(value.strip()) - 1 for value in selected.split(",") if value.strip()]
    invalid_indices = [
        index for index in indices if index < 0 or index >= len(image_topics)
    ]
    if invalid_indices:
        raise typer.BadParameter(
            "Invalid image topic number(s): "
            + ", ".join(str(index + 1) for index in invalid_indices)
        )
    proposed_ids = [_camera_id_from_ros_topic(image_topics[index].name) for index in indices]
    ids_are_unambiguous = len(proposed_ids) == len(set(proposed_ids))
    if ids_are_unambiguous:
        typer.echo(
            "Detected static camera IDs: " + ", ".join(proposed_ids)
        )
    cameras = []
    for camera_number, index in enumerate(indices):
        topic = image_topics[index]
        default_id = proposed_ids[camera_number]
        camera_id = (
            default_id
            if ids_are_unambiguous
            else typer.prompt(
                f"Camera ID for {topic.name}", default=default_id
            ).strip()
        )
        related = _related_camera_info_topics(topic, info_topics)
        info_topic = (
            related[0]
            if len(related) == 1
            else typer.prompt(
                f"CameraInfo topic for {camera_id} (optional)", default=""
            ).strip()
        )
        intrinsics = None
        if not info_topic:
            intrinsics = _prompt_path(
                f"Existing intrinsics for {camera_id}", directory=False
            )
        cameras.append(
            StaticCameraSettings(
                id=camera_id,
                intrinsics=intrinsics,
                image_topic=topic.name,
                camera_info_topic=info_topic or None,
            )
        )
    return cameras, moving


def _real_data_input(
    repository_root: Path,
    console: Console,
    raw_inputs: list[RawInputSummary],
) -> tuple[
    Path,
    list[StaticCameraSettings],
    MovingCameraSettings,
    SamplingSettings,
    McapSettings,
    str,
]:
    input_root = _choose_raw_input_root(console, raw_inputs, repository_root)
    discovered = discover_inputs(input_root, recursive=True)
    recordings = [item.path for item in discovered if item.kind == "mcap"]
    videos = [
        item.path
        for item in discovered
        if item.kind == "video"
        and media_path_role(item.path, input_root)
        not in {"checkerboard", "static"}
    ]
    frame_directories = set(
        discover_image_directories(input_root)["moving"]
    )
    local_moving_available = bool(videos or frame_directories)
    direct_groups = _detected_static_camera_groups(
        input_root, MovingCameraSettings(id="moving_calib_camera")
    )
    if not local_moving_available and not recordings:
        raise RuntimeError(
            "No moving-camera video, moving frame folder, .mcap, or .db3 was "
            f"detected below {input_root}. Add the files and start the wizard again."
        )

    if local_moving_available and direct_groups:
        proposal = "moving video/frames + directly paired static images/intrinsics"
    elif local_moving_available and recordings:
        proposal = "moving video/frames + static camera topics from the ROS recording"
    elif recordings:
        proposal = "moving and static camera topics from the ROS recording"
    else:
        proposal = "moving video/frames + manually confirmed static files"
    console.print(
        Panel(
            f"Folder: {input_root}\n"
            f"Videos: {len(videos)}; moving-frame folders: {len(frame_directories)}; "
            f"direct static cameras: {len(direct_groups)}; "
            f"checkerboard image folders: "
            f"{len(discover_image_directories(input_root)['checkerboard'])}; "
            f"ROS recordings: {len(recordings)}\n\n"
            f"Proposed interpretation: {proposal}\n\n"
            "Files may be mixed or placed in arbitrary subfolders. rigcal proposes "
            "roles from filenames, directory names, intrinsic metadata, and ROS topic "
            "types, but asks you to confirm camera identity before saving.",
            title="Automatic real-data input detection",
        )
    )

    mcap = McapSettings()
    if local_moving_available:
        moving, sampling, suggested_id = _moving_source(
            console, input_root, repository_root
        )
        use_recording_for_static = bool(recordings and not direct_groups)
        if recordings and direct_groups and typer.confirm(
            "Use ROS recording topics instead of the detected direct static pairs?",
            default=False,
        ):
            use_recording_for_static = True
        if use_recording_for_static:
            mcap_path = _select_detected_path(
                "ROS recording", recordings, directory=False
            )
            cameras, _ = _mcap_camera_sources(
                console, mcap_path, moving_from_recording=False
            )
            mcap = McapSettings(path=mcap_path)
        else:
            cameras = _direct_static_cameras(console, input_root, moving)
    else:
        mcap_path = _select_detected_path("ROS recording", recordings, directory=False)
        cameras, moving_from_bag = _mcap_camera_sources(
            console, mcap_path, moving_from_recording=True
        )
        assert moving_from_bag is not None
        moving = moving_from_bag
        # ROS recordings retain their stored timestamps/frames. target_hz is
        # requested only when rigcal extracts a newly selected local video.
        sampling = SamplingSettings(target_hz=None)
        suggested_id = safe_id(input_root.name)
        mcap = McapSettings(path=mcap_path)
    return input_root, cameras, moving, sampling, mcap, suggested_id


def _lighting_profiles(repository_root: Path) -> dict[str, dict[str, object]]:
    path = (
        repository_root
        / "src/calib_lab/bus_real_data/worlds/lighting/LIGHTING_VARIANTS.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    attenuation = payload["attenuation"]
    profiles: dict[str, dict[str, object]] = {
        "baseline": {
            "ambient": [0.7, 0.7, 0.7, 1.0],
            "light_diffuse": [0.8, 0.8, 0.8, 1.0],
            "light_specular": [0.2, 0.2, 0.2, 1.0],
            "panel_emissive": "not defined in the original SDF",
            "attenuation": "not defined for the original directional light",
        }
    }
    for raw_name, values in payload["variants"].items():
        name = raw_name.removeprefix("ceiling_")
        profiles[name] = {**values, "attenuation": attenuation}
    return profiles


def _show_lighting_profiles(
    repository_root: Path, console: Console
) -> None:
    table = Table(title="Lighting profiles from LIGHTING_VARIANTS.json")
    table.add_column("Profile")
    table.add_column("Ambient RGB(A)", overflow="fold")
    table.add_column("Diffuse RGB(A)", overflow="fold")
    table.add_column("Specular RGB(A)", overflow="fold")
    table.add_column("Emissive RGB(A)", overflow="fold")
    table.add_column("Attenuation", overflow="fold")
    for name, values in _lighting_profiles(repository_root).items():
        table.add_row(
            name,
            str(values["ambient"]),
            str(values["light_diffuse"]),
            str(values["light_specular"]),
            str(values["panel_emissive"]),
            str(values["attenuation"]),
        )
    console.print(table)


def _edit_simulation_parameters(
    repository_root: Path,
    console: Console,
    parameters: dict[str, object],
    route: Path,
    *,
    world_name: str = "Bus",
    capabilities: Iterable[str] | None = None,
    available_routes: dict[str, Path] | None = None,
    lighting_profiles: Iterable[str] | None = None,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    capture: dict[str, object] = {
        "settle_seconds": 0.35,
        "post_pose_skip": 5,
        "frame_timeout_seconds": 3.0,
        "startup_timeout_seconds": 60.0,
    }
    defaults = {
        **parameters,
        "route_file": str(route),
        **capture,
    }
    enabled = set(
        capabilities
        or {
            "route",
            "density",
            "resolution",
            "fov",
            "lighting",
            "motion_blur",
            "capture",
        }
    )
    routes = available_routes or {
        "route2": (
            repository_root
            / "src/calib_lab/bus_real_data/config/moving_camera_route2_interpolated_final.json"
        ).resolve(),
        "route1": (
            repository_root
            / "src/calib_lab/bus_real_data/config/moving_camera_route1_interpolated_final.json"
        ).resolve(),
    }
    supported_lighting = set(
        lighting_profiles
        or {"baseline", "dark_extreme", "low", "normal", "bright", "custom"}
    )
    meanings = {
        "route": "Moving-camera route: Route 2 baseline, Route 1, or custom JSON.",
        "route_file": "Exact moving-camera route JSON saved in the resolved configuration.",
        "target_route_frames": "Moving-camera frames; at least 2. Static cameras still use one snapshot.",
        "route_sampling_strategy": "Derived automatically from route and frame count.",
        "moving_width": "Moving-camera width; integer >= 64. Static cameras are unchanged.",
        "moving_height": "Moving-camera height; integer >= 64. Static cameras are unchanged.",
        "moving_hfov_deg": "Moving-camera horizontal FOV; >1 and <179°. Static cameras are unchanged.",
        "lighting": "World illumination: affects rendered pixels, never camera intrinsics.",
        "lighting_scale": "Positive multiplier up to 10; mainly for custom/physical profiles.",
        "motion_blur_kernel": "Moving frames only: 0 disables blur; otherwise an odd integer.",
        "motion_blur_angle_deg": "Moving-frame blur direction from -180 to 180 degrees.",
        "settle_seconds": "Non-negative wait after setting each route pose.",
        "post_pose_skip": "Non-negative number of fresh frames discarded per pose.",
        "frame_timeout_seconds": "Positive timeout for a captured frame.",
        "startup_timeout_seconds": "Positive Gazebo/ROS startup timeout.",
    }

    while True:
        console.print(
            Panel(
                "Route, frame count, resolution, FOV and motion blur apply only "
                "to the moving camera. Every static camera contributes one "
                "snapshot and keeps its SDF or explicitly provided intrinsics. "
                "Lighting is a world setting: it changes rendered appearance, "
                "but never K or D.",
                title="Camera parameter scope",
            )
        )
        grouped_rows = {
            "route": [
                ("route", parameters["route"]),
                ("route_file", str(route)),
            ],
            "density": [
                ("target_route_frames", parameters["target_route_frames"]),
                (
                    "route_sampling_strategy",
                    parameters["route_sampling_strategy"],
                ),
            ],
            "resolution": [
                ("moving_width", parameters["moving_width"]),
                ("moving_height", parameters["moving_height"]),
            ],
            "fov": [
                ("moving_hfov_deg", parameters["moving_hfov_deg"]),
            ],
            "lighting": [
                ("lighting", parameters["lighting"]),
                ("lighting_scale", parameters["lighting_scale"]),
            ],
            "motion_blur": [
                ("motion_blur_kernel", parameters["motion_blur_kernel"]),
                (
                    "motion_blur_angle_deg",
                    parameters["motion_blur_angle_deg"],
                ),
            ],
            "capture": [
                ("settle_seconds", capture["settle_seconds"]),
                ("post_pose_skip", capture["post_pose_skip"]),
                (
                    "frame_timeout_seconds",
                    capture["frame_timeout_seconds"],
                ),
                (
                    "startup_timeout_seconds",
                    capture["startup_timeout_seconds"],
                ),
            ],
        }
        rows = [
            row
            for capability, capability_rows in grouped_rows.items()
            if capability in enabled
            for row in capability_rows
        ]
        table = Table(title=f"{world_name} simulation parameter editor")
        table.add_column("#", justify="right")
        table.add_column("Parameter")
        table.add_column("Current")
        table.add_column("Default")
        table.add_column("Meaning / valid values", overflow="fold")
        for index, (key, current) in enumerate(rows, 1):
            table.add_row(
                str(index),
                key,
                str(current),
                str(defaults.get(key, "derived")),
                meanings[key],
            )
        console.print(table)
        selection = typer.prompt(
            "Parameter rows to change together "
            "(comma-separated; Enter = done; b = back)",
            default="",
            show_default=False,
        ).strip()
        if not selection:
            break
        if selection.lower() in {"0", "b", "back"}:
            _clear_terminal()
            raise WizardBack()
        try:
            selected = list(
                dict.fromkeys(int(value.strip()) for value in selection.split(","))
            )
        except ValueError:
            typer.echo("Use comma-separated row numbers, for example 5,7,10.")
            continue
        if not selected or min(selected) < 1 or max(selected) > len(rows):
            typer.echo(f"Choose rows between 1 and {len(rows)}.")
            continue
        back_to_simulation_table = False
        for index in selected:
            key, current = rows[index - 1]
            if key == "route_sampling_strategy":
                typer.echo("Sampling strategy is derived; edit route or frame count.")
                continue
            while True:
                try:
                    def field_value(label: str, default: object) -> str:
                        raw = str(
                            typer.prompt(label, default=str(default))
                        ).strip()
                        if raw.lower() in {"0", "b", "back"}:
                            raise WizardBack()
                        return raw

                    if key == "route":
                        route_names = ", ".join(routes)
                        value = field_value(
                            f"Route ({route_names}, or path to JSON)",
                            current,
                        )
                        if value in routes:
                            route = routes[value]
                            parameters["route"] = value
                        else:
                            candidate = Path(value).expanduser().resolve()
                            if not candidate.is_file():
                                raise ValueError(f"route file does not exist: {candidate}")
                            route = candidate
                            parameters["route"] = safe_id(candidate.stem)
                    elif key == "route_file":
                        candidate = Path(
                            field_value("Route JSON", route)
                        ).expanduser().resolve()
                        if not candidate.is_file():
                            raise ValueError(f"route file does not exist: {candidate}")
                        route = candidate
                        parameters["route"] = safe_id(candidate.stem)
                    elif key in {"moving_width", "moving_height"}:
                        value = int(field_value(key, current))
                        if value < 64:
                            raise ValueError("value must be at least 64")
                        parameters[key] = value
                    elif key == "target_route_frames":
                        value = int(field_value(key, current))
                        if value < 2:
                            raise ValueError("at least two frames are required")
                        parameters[key] = value
                    elif key == "moving_hfov_deg":
                        value = float(field_value(key, current))
                        if not 1 < value < 179:
                            raise ValueError("FOV must be greater than 1 and less than 179")
                        parameters[key] = value
                    elif key == "lighting":
                        _show_lighting_profiles(repository_root, console)
                        value = (
                            field_value(key, current)
                            .lower()
                            .removeprefix("ceiling_")
                        )
                        if value not in supported_lighting:
                            raise ValueError(
                                "unknown lighting profile; supported: "
                                + ", ".join(sorted(supported_lighting))
                            )
                        parameters[key] = value
                    elif key == "lighting_scale":
                        value = float(field_value(key, current))
                        if not 0 < value <= 10:
                            raise ValueError("scale must be greater than 0 and at most 10")
                        parameters[key] = value
                    elif key == "motion_blur_kernel":
                        value = int(field_value(key, current))
                        if value < 0 or (value != 0 and value % 2 == 0):
                            raise ValueError("kernel must be 0 or a positive odd integer")
                        parameters[key] = value
                    elif key == "motion_blur_angle_deg":
                        value = float(field_value(key, current))
                        if not -180 <= value <= 180:
                            raise ValueError("angle must be between -180 and 180")
                        parameters[key] = value
                    elif key == "post_pose_skip":
                        value = int(field_value(key, current))
                        if value < 0:
                            raise ValueError("value must be non-negative")
                        capture[key] = value
                    else:
                        value = float(field_value(key, current))
                        if (
                            key == "settle_seconds" and value < 0
                            or key != "settle_seconds" and value <= 0
                        ):
                            raise ValueError("timeout must be positive; settle may be zero")
                        capture[key] = value
                    break
                except WizardBack:
                    _clear_terminal()
                    back_to_simulation_table = True
                    break
                except (ValueError, typer.BadParameter) as exc:
                    _show_input_error(
                        f"Invalid value: {exc}. Try this parameter again."
                    )
            if back_to_simulation_table:
                break

        route_payload = json.loads(route.read_text(encoding="utf-8"))
        source_count = len(route_payload.get("frames", []))
        parameters["route_sampling_strategy"] = (
            "original_route_poses"
            if int(parameters["target_route_frames"]) == source_count
            else "resampled_route_poses"
        )
    resolved = Table(title="Complete resolved simulation parameter vector")
    resolved.add_column("Parameter")
    resolved.add_column("Value")
    for key, value in [*parameters.items(), *capture.items()]:
        resolved.add_row(key, str(value))
    resolved.add_row("route_file", str(route))
    console.print(resolved)
    return parameters, route, capture


def _simulation_input(
    repository_root: Path, console: Console
) -> tuple[
    list[StaticCameraSettings],
    MovingCameraSettings,
    SimulationSettings,
    str,
    Path | None,
]:
    registered_worlds = discover_world_manifests(repository_root)
    bus_world = next(
        (world for world in registered_worlds if world.id == "bus"),
        None,
    )
    if bus_world is None:
        raise RuntimeError(
            "The built-in bus world manifest is missing: "
            "config/simulation_worlds/bus.yaml"
        )
    experiments = discover_simulation_experiments(repository_root)
    console.print(
        Panel(
            format_simulation_parameters(BASELINE_SIMULATION_PARAMETERS),
            title="Default simulation baseline (used unless explicitly changed)",
        )
    )
    if experiments:
        table = Table(title="Existing simulation experiments")
        table.add_column("#", justify="right", overflow="fold")
        table.add_column("Variant", overflow="fold")
        table.add_column("Changed factor", overflow="fold")
        table.add_column("Value", overflow="fold")
        table.add_column("Frames", justify="right")
        table.add_column("Results")
        table.add_column("Complete parameter vector", overflow="fold")
        for index, experiment in enumerate(experiments, 1):
            table.add_row(
                str(index),
                experiment.variant,
                experiment.factor,
                experiment.value,
                str(experiment.moving_frames or "?"),
                "available" if experiment.has_results else "input only",
                format_simulation_parameters(experiment.parameters),
            )
        console.print(table)

    preset = _choice(
        "Gazebo simulation setup",
        {
            "1": "use the Route-2 baseline (recommended; reuse existing capture)",
            "2": "reuse one existing simulation/ablation by its table number",
            "3": "create a new bus-simulation parameter combination from baseline",
            "4": "import a new Gazebo world/rig (advanced: SDF, route and ROS topics)",
            "0": "back to input type",
        },
        "1",
    )
    if preset == "0":
        raise WizardBack()
    reused_dataset: Path | None = None
    capture_id: str | None = None
    if preset == "2":
        selected = _prompt_index(
            "Existing simulation number (0/b = back)",
            default=1,
            maximum=len(experiments),
        )
        if selected is None:
            raise WizardBack()
        experiment = experiments[selected - 1]
        if experiment.dataset_root is None:
            raise RuntimeError(
                f"{experiment.variant} has no reusable canonical captured dataset"
            )
        inspection = inspect_prepared_dataset(experiment.dataset_root)
        cameras = [
            StaticCameraSettings(id=camera_id)
            for camera_id in inspection["static_camera_ids"]
        ]
        intrinsic_ids = sorted(
            inspection["intrinsic_ids"] - set(inspection["static_camera_ids"])
        )
        moving_id = (
            intrinsic_ids[0] if len(intrinsic_ids) == 1 else "moving_calib_camera"
        )
        moving = MovingCameraSettings(id=moving_id)
        parameters = dict(experiment.parameters)
        world = (
            repository_root
            / "src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"
        ).resolve()
        route_name = str(parameters.get("route", "route2"))
        route_file = (
            "moving_camera_route1_interpolated_final.json"
            if route_name == "route1"
            else "moving_camera_route2_interpolated_final.json"
        )
        route = (
            repository_root / "src/calib_lab/bus_real_data/config" / route_file
        ).resolve()
        simulation = SimulationSettings(
            enabled=True,
            preset=f"existing_{safe_id(experiment.variant)}",
            world=world,
            route=route,
            moving_model_name="moving_calib_camera",
            route_name=route_name,
            moving_width=int(parameters["moving_width"]),
            moving_height=int(parameters["moving_height"]),
            moving_hfov_deg=float(parameters["moving_hfov_deg"]),
            lighting=str(parameters["lighting"]),
            lighting_scale=float(parameters["lighting_scale"]),
            motion_blur_kernel=int(parameters["motion_blur_kernel"]),
            motion_blur_angle_deg=float(parameters["motion_blur_angle_deg"]),
            target_route_frames=int(parameters["target_route_frames"]),
            route_sampling_strategy=str(parameters["route_sampling_strategy"]),
            settle_seconds=float(
                parameters.get("settle_seconds", 0.35)
            ),
            post_pose_skip=int(parameters.get("post_pose_skip", 5)),
            frame_timeout_seconds=float(
                parameters.get("frame_timeout_seconds", 3.0)
            ),
            startup_timeout_seconds=float(
                parameters.get("startup_timeout_seconds", 60.0)
            ),
        )
        console.print(
            Panel(
                f"Variant: {experiment.variant}\n"
                f"Parameters: {format_simulation_parameters(parameters)}\n"
                f"Results: {'available' if experiment.has_results else 'input only'}\n"
                f"Captured input: {experiment.dataset_root}",
                title="Selected existing simulation",
            )
        )
        return (
            cameras,
            moving,
            simulation,
            safe_id(experiment.variant),
            experiment.dataset_root,
        )
    if preset in {"1", "3"}:
        world = bus_world.sdf
        route = bus_world.baseline_route.path
        preset_id = "bus_baseline"
        cameras = [
            StaticCameraSettings(
                id=camera.id,
                image_topic=camera.image_topic,
                camera_info_topic=camera.camera_info_topic,
            )
            for camera in bus_world.static_cameras
        ]
        moving = MovingCameraSettings(
            id=bus_world.moving_camera.id,
            image_topic=bus_world.moving_camera.image_topic,
            camera_info_topic=bus_world.moving_camera.camera_info_topic,
        )
        model_name = bus_world.moving_camera.model_name
        moving_sensor_name = bus_world.moving_camera.sensor_name
        resource_paths = bus_world.resource_paths
        world_id = bus_world.id
        world_baseline = bus_world.baseline
        parameters = {
            key: value
            for key, value in BASELINE_SIMULATION_PARAMETERS.items()
            if key
            not in {
                "settle_seconds",
                "post_pose_skip",
                "frame_timeout_seconds",
                "startup_timeout_seconds",
            }
        }
        if preset == "3":
            parameters, route, capture_overrides = _edit_simulation_parameters(
                repository_root,
                console,
                parameters,
                route,
                world_name=bus_world.display_name,
                capabilities=bus_world.capabilities,
                available_routes={
                    item.id: item.path for item in bus_world.routes
                },
                lighting_profiles=bus_world.lighting_profiles,
            )
            lighting = str(parameters["lighting"])
            profile_world = bus_world.lighting_profiles.get(lighting)
            if profile_world is not None:
                world = profile_world
            preset_id = "bus_composed"
        else:
            capture_overrides = {
                "settle_seconds": 0.35,
                "post_pose_skip": 5,
                "frame_timeout_seconds": 3.0,
                "startup_timeout_seconds": 60.0,
            }
        parameters.update(capture_overrides)
        match = find_matching_simulation(experiments, parameters)
        if match is not None:
            console.print(
                Panel(
                    f"Variant: {match.variant}\n"
                    f"Changed factor: {match.factor}\n"
                    f"Value: {match.value}\n"
                    f"Results: {'available' if match.has_results else 'not available'}",
                    title="Matching existing experiment found",
                )
            )
            if preset == "3":
                console.print(
                    "New combination was selected: rigcal will record a new input "
                    "capture even though the same parameter vector already exists."
                )
            elif match.dataset_root is not None and typer.confirm(
                "Reuse its captured dataset instead of recording it again?", default=True
            ):
                reused_dataset = match.dataset_root
    else:
        custom_worlds = [
            item for item in registered_worlds if item.id != "bus"
        ]
        table = Table(title="Registered Gazebo worlds")
        table.add_column("#", justify="right")
        table.add_column("World")
        table.add_column("Manifest", overflow="fold")
        for index, item in enumerate(custom_worlds, 1):
            table.add_row(
                str(index),
                f"{item.display_name} ({item.id})",
                str(item.manifest_path),
            )
        register_number = len(custom_worlds) + 1
        table.add_row(
            str(register_number),
            "register/import another world",
            "manifest YAML or guided SDF/route setup",
        )
        console.print(table)
        selected = _prompt_index(
            "World number (0/b = back)",
            default=1 if custom_worlds else register_number,
            maximum=register_number,
        )
        if selected is None:
            raise WizardBack()
        if selected == register_number:
            manifest_text = typer.prompt(
                "Existing world-manifest YAML (Enter = guided SDF setup)",
                default="",
                show_default=False,
            ).strip()
            if manifest_text:
                manifest = install_world_manifest(
                    repository_root, Path(manifest_text).expanduser()
                )
            else:
                console.print(
                    Panel(
                        "This writes a reusable manifest below "
                        "config/simulation_worlds/. It is discovered on every "
                        "future rigcal start; no Python changes are required.",
                        title="Register a Gazebo world",
                    )
                )
                world_path = _prompt_path("Gazebo SDF world", directory=False)
                route_path = _prompt_path(
                    "Moving-camera baseline route JSON", directory=False
                )
                world_id_input = safe_id(
                    typer.prompt("World ID", default=world_path.stem).strip()
                )
                display_name = typer.prompt(
                    "Display name", default=world_id_input
                ).strip()
                count = typer.prompt(
                    "Number of static simulation cameras", type=int
                )
                if count < 1:
                    raise typer.BadParameter(
                        "At least one static camera is required"
                    )
                static_payload = []
                for index in range(count):
                    camera_id = typer.prompt(
                        f"Static camera {index + 1} ID",
                        default=f"camera_{index + 1}",
                    ).strip()
                    static_payload.append(
                        {
                            "id": camera_id,
                            "model_name": typer.prompt(
                                f"Gazebo model for {camera_id}",
                                default=camera_id,
                            ).strip(),
                            "sensor_name": typer.prompt(
                                f"Gazebo sensor for {camera_id} "
                                "(blank = auto)",
                                default="",
                                show_default=False,
                            ).strip()
                            or None,
                            "image_topic": typer.prompt(
                                f"Image topic for {camera_id}",
                                default=f"/{camera_id}/image",
                            ).strip(),
                            "camera_info_topic": typer.prompt(
                                f"CameraInfo topic for {camera_id}",
                                default=f"/{camera_id}/camera_info",
                            ).strip(),
                        }
                    )
                moving_id = typer.prompt(
                    "Moving camera ID", default="calibration_camera"
                ).strip()
                manifest_path = (
                    repository_root
                    / "config"
                    / "simulation_worlds"
                    / f"{world_id_input}.yaml"
                )
                payload = {
                    "schema_version": 1,
                    "id": world_id_input,
                    "display_name": display_name,
                    "sdf": str(world_path),
                    "resource_paths": [str(world_path.parent)],
                    "static_cameras": static_payload,
                    "moving_camera": {
                        "id": moving_id,
                        "model_name": typer.prompt(
                            "Gazebo moving-camera model name",
                            default=moving_id,
                        ).strip(),
                        "sensor_name": typer.prompt(
                            "Gazebo moving-camera sensor name "
                            "(blank = auto)",
                            default="",
                            show_default=False,
                        ).strip()
                        or None,
                        "image_topic": typer.prompt(
                            "Moving-camera image topic",
                            default=f"/{moving_id}/image",
                        ).strip(),
                        "camera_info_topic": typer.prompt(
                            "Moving-camera CameraInfo topic",
                            default=f"/{moving_id}/camera_info",
                        ).strip(),
                    },
                    "routes": [
                        {
                            "id": safe_id(route_path.stem),
                            "path": str(route_path),
                            "baseline": True,
                        }
                    ],
                }
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = manifest_path.with_suffix(".yaml.tmp")
                temporary.write_text(
                    yaml.safe_dump(
                        payload, sort_keys=False, allow_unicode=True
                    ),
                    encoding="utf-8",
                )
                temporary.replace(manifest_path)
                manifest = load_world_manifest(manifest_path)
        else:
            manifest = custom_worlds[selected - 1]
        world = manifest.sdf
        route = manifest.baseline_route.path
        cameras = [
            StaticCameraSettings(
                id=camera.id,
                image_topic=camera.image_topic,
                camera_info_topic=camera.camera_info_topic,
            )
            for camera in manifest.static_cameras
        ]
        moving = MovingCameraSettings(
            id=manifest.moving_camera.id,
            image_topic=manifest.moving_camera.image_topic,
            camera_info_topic=manifest.moving_camera.camera_info_topic,
        )
        model_name = manifest.moving_camera.model_name
        moving_sensor_name = manifest.moving_camera.sensor_name
        resource_paths = manifest.resource_paths
        world_id = manifest.id
        world_baseline = manifest.baseline
        preset_id = f"world_{manifest.id}"
        parameters = {
            "route": manifest.baseline["route_name"],
            **{
                key: value
                for key, value in manifest.baseline.items()
                if key != "route_name"
            },
        }
        parameters, route, capture_overrides = _edit_simulation_parameters(
            repository_root,
            console,
            parameters,
            route,
            world_name=manifest.display_name,
            capabilities=manifest.capabilities,
            available_routes={
                item.id: item.path for item in manifest.routes
            },
            lighting_profiles=manifest.lighting_profiles,
        )
        selected_lighting_world = manifest.lighting_profiles.get(
            str(parameters["lighting"])
        )
        if selected_lighting_world is not None:
            world = selected_lighting_world

    capture_overrides = locals().get(
        "capture_overrides",
        {
            "settle_seconds": 0.35,
            "post_pose_skip": 5,
            "frame_timeout_seconds": 3.0,
            "startup_timeout_seconds": 60.0,
        },
    )
    settle = float(capture_overrides["settle_seconds"])
    post_pose_skip = int(capture_overrides["post_pose_skip"])
    frame_timeout = float(capture_overrides["frame_timeout_seconds"])
    startup_timeout = float(capture_overrides["startup_timeout_seconds"])
    if reused_dataset is None:
        capture_id = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    if preset != "3" and typer.confirm(
        "Open advanced simulation capture settings?", default=False
    ):
        settle = typer.prompt("Settle time per route pose [s]", default=settle, type=float)
        post_pose_skip = typer.prompt(
            "Fresh frames skipped after each pose", default=post_pose_skip, type=int
        )
        frame_timeout = typer.prompt(
            "Frame timeout per route pose [s]", default=frame_timeout, type=float
        )
        startup_timeout = typer.prompt(
            "Gazebo/bridge startup timeout [s]", default=startup_timeout, type=float
        )
    simulation = SimulationSettings(
        enabled=True,
        preset=preset_id,
        capture_id=capture_id,
        world_id=locals().get("world_id", "bus"),
        world_baseline=locals().get(
            "world_baseline", BASELINE_SIMULATION_PARAMETERS
        ),
        world=world,
        route=route,
        resource_paths=locals().get("resource_paths", []),
        moving_model_name=model_name,
        moving_sensor_name=locals().get("moving_sensor_name"),
        settle_seconds=settle,
        post_pose_skip=post_pose_skip,
        frame_timeout_seconds=frame_timeout,
        startup_timeout_seconds=startup_timeout,
        route_name=str(parameters["route"]),
        moving_width=int(parameters["moving_width"]),
        moving_height=int(parameters["moving_height"]),
        moving_hfov_deg=float(parameters["moving_hfov_deg"]),
        lighting=str(parameters["lighting"]),
        lighting_scale=float(parameters["lighting_scale"]),
        motion_blur_kernel=int(parameters["motion_blur_kernel"]),
        motion_blur_angle_deg=float(parameters["motion_blur_angle_deg"]),
        target_route_frames=int(parameters["target_route_frames"]),
        route_sampling_strategy=str(parameters["route_sampling_strategy"]),
    )
    parameters.update(
        {
            "settle_seconds": settle,
            "post_pose_skip": post_pose_skip,
            "frame_timeout_seconds": frame_timeout,
            "startup_timeout_seconds": startup_timeout,
        }
    )
    summary = Table(title="Resolved simulation input parameters")
    summary.add_column("Parameter")
    summary.add_column("Value")
    for key, value in parameters.items():
        summary.add_row(key, str(value))
    summary.add_row("capture", "reuse existing dataset" if reused_dataset else "new Gazebo capture")
    summary.add_row("Gazebo mode", "headless server; closes after capture")
    summary.add_row(
        "Camera scope",
        "route/resolution/FOV/blur = moving only; static = one unchanged snapshot",
    )
    summary.add_row(
        "Intrinsics",
        "SDF/CameraInfo unchanged unless an explicit static or moving file is loaded",
    )
    if capture_id is not None:
        summary.add_row("capture_id", capture_id)
    console.print(summary)
    suggested_id = (
        safe_id(match.variant)
        if reused_dataset is not None and match is not None
        else _simulation_experiment_id(parameters)
    )
    return cameras, moving, simulation, suggested_id, reused_dataset


def _new_method_job(
    method_id: str,
    *,
    prompt_for_single_marker: bool,
) -> MethodQueueJob:
    method = calibration_methods.get(method_id)
    methods = MethodSettings(enabled=[method_id])
    extensions: dict[str, dict] = {}
    if method_id not in {"ap01", "ap02", "ap03"}:
        try:
            extensions[method_id] = method.config_model().model_dump(mode="python")
        except ValidationError:
            extensions[method_id] = _prompt_component_options(
                method.display_name, method.config_model
            )
        methods = methods.model_copy(update={"extensions": extensions}, deep=True)
    return MethodQueueJob(
        method_id=method_id,
        label=safe_id(f"{method_id}_baseline"),
        methods=methods,
        markers=MarkerSettings(),
        observation_quality=ObservationQualitySettings(),
        colmap=ColmapSettings(),
        evaluation=EvaluationSettings(),
    )


def _method_job_summary(job: MethodQueueJob) -> str:
    if job.method_id == "ap01":
        return (
            f"matcher={job.colmap.matcher}, GPU={job.colmap.gpu_mode}, "
            f"root={job.methods.ap01.root_camera}, observations=all passing quality"
        )
    if job.method_id == "ap02":
        value = job.methods.ap02
        return (
            f"nfev={value.max_nfev_static}/{value.max_nfev_moving}, "
            f"loss={value.ba_robust_loss}@{value.ba_robust_loss_scale_px:g}px, "
            f"ref={value.reference_marker_id}, observations=all passing quality"
        )
    if job.method_id == "ap03":
        single = job.methods.ap03.single
        multi = job.methods.ap03.multi
        ids = multi.marker_ids
        ids_text = ids if ids == "auto" else ",".join(map(str, ids))
        return (
            f"single={single.scale_marker_id}, multi={ids_text}, "
            f"matcher={job.colmap.matcher}; one COLMAP, multi primary"
        )
    return "registered extension defaults"


def _show_method_queue(console: Console, jobs: list[MethodQueueJob]) -> None:
    table = Table(title="Calibration queue — one reproducible result run per row")
    table.add_column("#", justify="right")
    table.add_column("Run label")
    table.add_column("Method")
    table.add_column("Resolved baseline/config summary", overflow="fold")
    for index, job in enumerate(jobs, 1):
        table.add_row(
            str(index),
            job.label,
            calibration_methods.get(job.method_id).display_name,
            _method_job_summary(job),
        )
    console.print(table)
    console.print(
        "[dim]Each row is independent: duplicate a row to compare another marker, "
        "matcher or parameter set without overwriting the baseline.[/dim]"
    )


def _ids_text(value: str | list[int]) -> str:
    return value if value == "auto" else ",".join(map(str, value))


def _parse_ids(value: str) -> str | list[int]:
    value = value.strip()
    if value.lower() == "auto":
        return "auto"
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("enter 'auto' or at least one marker ID")
    return result


METHOD_JOB_GROUPS = frozenset(
    {
        "OBSERVATION QUALITY",
        "METHOD-SPECIFIC SETTINGS",
        "COLMAP SETTINGS",
    }
)


def _setting_rows(
    job: MethodQueueJob,
    groups: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, str, str, object, object, str]]:
    defaults = _new_method_job(job.method_id, prompt_for_single_marker=False)
    accepted = (
        "all detected IDs"
        if job.markers.accepted_ids == "all_detected"
        else ",".join(map(str, job.markers.accepted_ids))
    )
    rows: list[tuple[str, str, str, object, object, str]] = [
        ("label", "RUN IDENTITY", "Run label", job.label, defaults.label, "Names this independent result snapshot."),
        ("evaluation_anchor", "COMMON EVALUATION", "Evaluation anchor", job.evaluation.anchor_marker_id, defaults.evaluation.anchor_marker_id, "auto_common selects one marker shared by every compared method after estimation."),
        ("evaluation_reprojection", "COMMON EVALUATION", "Evaluation reprojection threshold [px]", job.evaluation.reprojection_threshold_px, defaults.evaluation.reprojection_threshold_px, "Smaller requires tighter common post-hoc triangulation support."),
        ("evaluation_inliers", "COMMON EVALUATION", "Evaluation minimum inliers", job.evaluation.minimum_inliers, defaults.evaluation.minimum_inliers, "Higher requires more common-support observations."),
        ("evaluation_ransac", "COMMON EVALUATION", "Evaluation RANSAC iterations", job.evaluation.ransac_iterations, defaults.evaluation.ransac_iterations, "Higher tests more hypotheses and increases evaluation runtime."),
        ("evaluation_angle", "COMMON EVALUATION", "Minimum triangulation angle [deg]", job.evaluation.minimum_triangulation_angle_deg, defaults.evaluation.minimum_triangulation_angle_deg, "Larger rejects weak-baseline triangulation geometry."),
        ("evaluation_max_observations", "COMMON EVALUATION", "Maximum moving observations per marker", job.evaluation.maximum_moving_observations_per_marker, defaults.evaluation.maximum_moving_observations_per_marker, "Caps deterministic evaluation work per marker."),
        ("accepted_ids", "ARUCO INPUT", "Accepted marker IDs", accepted, "all detected IDs", "Use all detections or a comma-separated ID list."),
        ("dictionary", "ARUCO INPUT", "ArUco dictionary", job.markers.dictionary, defaults.markers.dictionary, "Must match the printed marker family."),
        ("marker_length", "ARUCO INPUT", "Marker edge length [m]", job.markers.length_m, defaults.markers.length_m, "Positive physical size used for metric scale."),
        ("quality_reprojection", "OBSERVATION QUALITY", "Maximum PnP reprojection RMSE [px]", job.observation_quality.maximum_pnp_reprojection_error_px, 25.0, "Smaller rejects imprecise PnP observations; 25 px is the compatibility default."),
        ("quality_area", "OBSERVATION QUALITY", "Minimum marker area [px²]", job.observation_quality.minimum_marker_area_px2, 0.0, "Larger rejects small/distant detections; 0 preserves compatibility."),
        ("quality_distance", "OBSERVATION QUALITY", "Maximum marker distance [m]", job.observation_quality.maximum_marker_distance_m, "disabled", "Smaller keeps near PnP observations; useful tests depend on rig size."),
    ]
    if job.method_id == "ap01":
        value, base = job.methods.ap01, defaults.methods.ap01
        rows.extend([
            ("root_camera", "METHOD-SPECIFIC SETTINGS", "Root camera", value.root_camera, base.root_camera, "Coordinate origin; auto is resolved from filtered graph coverage."),
        ])
    elif job.method_id == "ap02":
        value, base = job.methods.ap02, defaults.methods.ap02
        rows.extend([
            ("ap02_reference", "METHOD-SPECIFIC SETTINGS", "Reference marker", value.reference_marker_id, base.reference_marker_id, "Pose-graph anchor selected after filtered observations."),
            ("max_nfev_static", "METHOD-SPECIFIC SETTINGS", "Static-only BA function evaluation limit", value.max_nfev_static, base.max_nfev_static, "Bundle-adjustment optimizer budget for the diagnostic static-only result."),
            ("max_nfev_moving", "METHOD-SPECIFIC SETTINGS", "Combined static + moving BA function evaluation limit", value.max_nfev_moving, base.max_nfev_moving, "Bundle-adjustment optimizer budget for the primary combined result."),
            ("ba_loss", "METHOD-SPECIFIC SETTINGS", "BA robust loss", value.ba_robust_loss, base.ba_robust_loss, "soft_l1 baseline; huber is piecewise robust; linear disables robustness."),
            ("ba_loss_scale", "METHOD-SPECIFIC SETTINGS", "BA robust loss scale [px]", value.ba_robust_loss_scale_px, base.ba_robust_loss_scale_px, "Smaller downweights residuals earlier; typical tests: 1, 3, 5 px."),
        ])
    elif job.method_id == "ap03":
        single, multi, scale = (
            job.methods.ap03.single,
            job.methods.ap03.multi,
            job.methods.ap03.scale,
        )
        base_single, base_multi, base_scale = (
            defaults.methods.ap03.single,
            defaults.methods.ap03.multi,
            defaults.methods.ap03.scale,
        )
        rows.extend([
            ("single_marker", "METHOD-SPECIFIC SETTINGS", "Single diagnostic scale marker", single.scale_marker_id, base_single.scale_marker_id, "Diagnostic marker chosen after filtered observations."),
            ("multi_markers", "METHOD-SPECIFIC SETTINGS", "Multi primary marker set", _ids_text(multi.marker_ids), "auto", "Auto uses every compatible filtered marker."),
            ("scale_reprojection", "METHOD-SPECIFIC SETTINGS", "Scale RANSAC threshold [px]", scale.reprojection_threshold_px, base_scale.reprojection_threshold_px, "Shared by Single and Multi; smaller requires tighter triangulation support."),
            ("scale_ransac", "METHOD-SPECIFIC SETTINGS", "Scale RANSAC iterations", scale.ransac_iterations, base_scale.ransac_iterations, "Shared by Single and Multi; higher explores more hypotheses but increases runtime."),
            ("scale_inliers", "METHOD-SPECIFIC SETTINGS", "Scale minimum inliers", scale.minimum_inliers, base_scale.minimum_inliers, "Shared by Single and Multi; higher requires more supporting views per marker corner."),
        ])
    if job.method_id in {"ap01", "ap03"}:
        rows.extend([
            ("matcher", "COLMAP SETTINGS", "Matcher", job.colmap.matcher, "exhaustive", "Exhaustive compares all image pairs; sequential limits temporal pairs."),
            ("gpu_mode", "COLMAP SETTINGS", "GPU mode", job.colmap.gpu_mode, "auto", "auto probes capability; true fails preflight without a compatible GPU."),
            ("mapper_matches", "COLMAP SETTINGS", "Mapper minimum matches", job.colmap.mapper_minimum_matches, 8, "Higher requires stronger image pairs; lower may add weak registrations."),
            ("maximum_image_size", "COLMAP SETTINGS", "Maximum feature image size", job.colmap.maximum_image_size, 2400, "Larger preserves detail but increases memory/runtime."),
            ("maximum_features", "COLMAP SETTINGS", "Maximum features per image", job.colmap.maximum_features, 8192, "Larger improves difficult matching but increases runtime."),
        ])
        if job.colmap.matcher == "sequential":
            rows.extend([
                ("sequential_overlap", "COLMAP SETTINGS", "Sequential overlap", job.colmap.sequential_overlap, 20, "Number of temporal neighbors; higher widens matching."),
                ("loop_detection", "COLMAP SETTINGS", "Sequential loop detection", job.colmap.loop_detection, True, "Adds non-local loop candidates for sequential matching."),
            ])
    return rows if groups is None else [row for row in rows if row[1] in groups]


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes", "true", "1", "on"}:
        return True
    if normalized in {"n", "no", "false", "0", "off"}:
        return False
    raise ValueError("enter yes or no")


def _prompt_enum_choice(
    label: str,
    current: str,
    choices: tuple[tuple[str, str], ...],
) -> str:
    """Show finite choices before prompting and keep invalid input local."""

    typer.echo(f"\n{label}:")
    values = [value for value, _ in choices]
    for index, (value, meaning) in enumerate(choices, 1):
        suffix = " (current)" if value == current else ""
        typer.echo(f"  {index}. {value}{suffix} — {meaning}")
    default = values.index(current) + 1 if current in values else 1
    while True:
        raw = str(typer.prompt("Selection", default=default)).strip().lower()
        if raw in {"0", "b", "back"}:
            raise WizardBack()
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(values):
                return values[selected - 1]
        if raw in values:
            return raw
        matches = [value for value in values if value.startswith(raw)]
        if raw and len(matches) == 1:
            return matches[0]
        _show_input_error(
            f"Choose 1-{len(values)} or enter one of: {', '.join(values)}."
        )


def _edit_method_job(
    console: Console,
    job: MethodQueueJob,
    *,
    groups: set[str] | frozenset[str] = METHOD_JOB_GROUPS,
    title: str | None = None,
) -> MethodQueueJob:
    while True:
        rows = _setting_rows(job, groups)
        table = Table(title=title or f"Method settings for {job.label}")
        table.add_column("#", justify="right")
        table.add_column("Group")
        table.add_column("Parameter")
        table.add_column("Current")
        table.add_column("Default")
        table.add_column("Meaning", overflow="fold")
        for index, (_, group, label, current, default, meaning) in enumerate(rows, 1):
            table.add_row(
                str(index), group, label, str(current), str(default), meaning
            )
        console.print(table)
        selection = typer.prompt(
            "Setting numbers to change together "
            "(comma-separated; Enter = keep all; b = back)",
            default="",
            show_default=False,
        ).strip()
        if not selection:
            return job
        if selection.lower() in {"0", "b", "back"}:
            _clear_terminal()
            return job
        try:
            indices = [int(value.strip()) for value in selection.split(",")]
        except ValueError:
            _show_input_error(
                "Use comma-separated setting numbers, for example 2,5,9."
            )
            continue
        if not indices or min(indices) < 1 or max(indices) > len(rows):
            _show_input_error(
                f"Choose setting numbers between 1 and {len(rows)}."
            )
            continue
        back_to_table = False
        for index in dict.fromkeys(indices):
            key, _, label, current, _, _ = rows[index - 1]
            if key == "colmap_na":
                typer.echo("COLMAP is not applicable to AP02.")
                continue
            try:
                if key == "matcher":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            (
                                "exhaustive",
                                "compare every image pair; baseline and best for unordered captures",
                            ),
                            (
                                "sequential",
                                "compare temporal neighbors; faster for ordered video frames",
                            ),
                        ),
                    )
                elif key == "gpu_mode":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("auto", "use GPU only when the capability probe succeeds"),
                            ("true", "require a compatible GPU or fail preflight"),
                            ("false", "always run COLMAP on CPU"),
                        ),
                    )
                elif key == "ba_loss":
                    value = _prompt_enum_choice(
                        label,
                        str(current),
                        (
                            ("soft_l1", "smooth robust baseline loss"),
                            ("huber", "piecewise robust loss"),
                            ("linear", "plain least squares without robust downweighting"),
                        ),
                    )
                else:
                    value = typer.prompt(
                        f"{label} (b = back)",
                        default=str(current),
                    ).strip()
                    if value.lower() in {"b", "back"}:
                        raise WizardBack()
            except WizardBack:
                _clear_terminal()
                back_to_table = True
                break
            if key == "label":
                job.label = safe_id(value)
            elif key == "evaluation_anchor":
                anchor: str | int = (
                    "auto_common"
                    if value.lower() in {"auto", "auto_common"}
                    else int(value)
                )
                job.evaluation = job.evaluation.model_copy(
                    update={"anchor_marker_id": anchor}
                )
            elif key in {
                "evaluation_reprojection",
                "evaluation_inliers",
                "evaluation_ransac",
                "evaluation_angle",
                "evaluation_max_observations",
            }:
                field = {
                    "evaluation_reprojection": "reprojection_threshold_px",
                    "evaluation_inliers": "minimum_inliers",
                    "evaluation_ransac": "ransac_iterations",
                    "evaluation_angle": "minimum_triangulation_angle_deg",
                    "evaluation_max_observations": (
                        "maximum_moving_observations_per_marker"
                    ),
                }[key]
                typed_value: float | int = (
                    float(value)
                    if key in {"evaluation_reprojection", "evaluation_angle"}
                    else int(value)
                )
                job.evaluation = job.evaluation.model_copy(
                    update={field: typed_value}
                )
            elif key == "accepted_ids":
                parsed = _parse_ids(
                    "auto"
                    if value.lower() in {"all detected ids", "all_detected"}
                    else value
                )
                job.markers = job.markers.model_copy(
                    update={
                        "accepted_ids": (
                            "all_detected" if parsed == "auto" else parsed
                        )
                    }
                )
            elif key == "dictionary":
                job.markers = job.markers.model_copy(update={"dictionary": value})
            elif key == "marker_length":
                job.markers = job.markers.model_copy(update={"length_m": float(value)})
            elif key in {
                "quality_reprojection",
                "quality_area",
                "quality_distance",
            }:
                field = {
                    "quality_reprojection": "maximum_pnp_reprojection_error_px",
                    "quality_area": "minimum_marker_area_px2",
                    "quality_distance": "maximum_marker_distance_m",
                }[key]
                typed: str | float = (
                    "disabled"
                    if value.lower() == "disabled"
                    else float(value)
                )
                job.observation_quality = job.observation_quality.model_copy(
                    update={field: typed}
                )
            elif key == "root_camera":
                job.methods = job.methods.model_copy(
                    update={
                        "ap01": job.methods.ap01.model_copy(
                            update={"root_camera": value}
                        )
                    },
                    deep=True,
                )
            elif key == "ap02_reference":
                job.methods = job.methods.model_copy(
                    update={
                        "ap02": job.methods.ap02.model_copy(
                            update={
                                "reference_marker_id": _optional_marker(value)
                            }
                        )
                    },
                    deep=True,
                )
            elif key in {"max_nfev_static", "max_nfev_moving"}:
                field = {
                    "max_nfev_static": "static_only_ba_max_function_evaluations",
                    "max_nfev_moving": "combined_ba_max_function_evaluations",
                }[key]
                job.methods = job.methods.model_copy(update={"ap02": job.methods.ap02.model_copy(update={field: int(value)})}, deep=True)
            elif key in {"ba_loss", "ba_loss_scale"}:
                field = (
                    "ba_robust_loss"
                    if key == "ba_loss"
                    else "ba_robust_loss_scale_px"
                )
                typed = value if key == "ba_loss" else float(value)
                job.methods = job.methods.model_copy(
                    update={
                        "ap02": job.methods.ap02.model_copy(
                            update={field: typed}
                        )
                    },
                    deep=True,
                )
            elif key == "single_marker":
                ap03 = job.methods.ap03.model_copy(
                    update={
                        "single": job.methods.ap03.single.model_copy(
                            update={"scale_marker_id": _optional_marker(value)}
                        )
                    },
                    deep=True,
                )
                job.methods = job.methods.model_copy(update={"ap03": ap03}, deep=True)
            elif key == "multi_markers":
                ap03 = job.methods.ap03.model_copy(
                    update={
                        "multi": job.methods.ap03.multi.model_copy(
                            update={"marker_ids": _parse_ids(value)}
                        )
                    },
                    deep=True,
                )
                job.methods = job.methods.model_copy(update={"ap03": ap03}, deep=True)
            elif key in {
                "scale_reprojection",
                "scale_ransac",
                "scale_inliers",
            }:
                if key.endswith("reprojection"):
                    field = "reprojection_threshold_px"
                elif key.endswith("ransac"):
                    field = "ransac_iterations"
                else:
                    field = "minimum_inliers"
                typed = (
                    float(value)
                    if field == "reprojection_threshold_px"
                    else int(value)
                )
                ap03 = job.methods.ap03.model_copy(
                    update={
                        "scale": job.methods.ap03.scale.model_copy(
                            update={field: typed}
                        )
                    },
                    deep=True,
                )
                job.methods = job.methods.model_copy(update={"ap03": ap03}, deep=True)
            elif key in {"colmap_executable", "matcher"}:
                field = "executable" if key == "colmap_executable" else key
                job.colmap = job.colmap.model_copy(update={field: value})
            elif key == "gpu_mode":
                job.colmap = job.colmap.model_copy(update={"gpu_mode": value.lower()})
            elif key == "loop_detection":
                job.colmap = job.colmap.model_copy(update={key: _bool_value(value)})
            elif key in {"mapper_matches", "maximum_image_size", "maximum_features", "sequential_overlap"}:
                field = {
                    "mapper_matches": "mapper_minimum_matches",
                    "maximum_image_size": "maximum_image_size",
                    "maximum_features": "maximum_features",
                    "sequential_overlap": "sequential_overlap",
                }[key]
                job.colmap = job.colmap.model_copy(update={field: int(value)})
            elif key in {"ap03_image_size", "ap03_features"}:
                field = "ap03_maximum_image_size" if key == "ap03_image_size" else "ap03_maximum_features"
                job.colmap = job.colmap.model_copy(update={field: None if value.lower() in {"", "none", "runner default"} else int(value)})
            elif key == "extension":
                payload = yaml.safe_load(value) or {}
                method = calibration_methods.get(job.method_id)
                validated = method.config_model.model_validate(payload).model_dump(mode="python")
                extensions = dict(job.methods.extensions)
                extensions[job.method_id] = validated
                job.methods = job.methods.model_copy(update={"extensions": extensions}, deep=True)
        if back_to_table:
            continue
        # Validate all copied models and then show the resolved values again. This also
        # reveals sequential-only settings immediately after changing the matcher.
        job.methods = MethodSettings.model_validate(job.methods.model_dump(mode="python"))
        job.markers = MarkerSettings.model_validate(job.markers.model_dump(mode="python"))
        job.colmap = ColmapSettings.model_validate(job.colmap.model_dump(mode="python"))
        job.observation_quality = ObservationQualitySettings.model_validate(
            job.observation_quality.model_dump(mode="python")
        )
        if not typer.confirm("Change more values in this menu?", default=False):
            return job


def _clone_method_job(job: MethodQueueJob, label: str) -> MethodQueueJob:
    return MethodQueueJob(
        method_id=job.method_id,
        label=safe_id(label),
        methods=job.methods.model_copy(deep=True),
        markers=job.markers.model_copy(deep=True),
        observation_quality=job.observation_quality.model_copy(deep=True),
        colmap=job.colmap.model_copy(deep=True),
        evaluation=job.evaluation.model_copy(deep=True),
    )


def _method_queue(console: Console) -> list[MethodQueueJob]:
    methods = [
        calibration_methods.get(method_id)
        for method_id in ("ap01", "ap02", "ap03")
    ]
    recommended_ids = ["ap01", "ap02", "ap03"]
    explanations = {
        "ap01": "experimental baseline; marker-direct and moving-COLMAP relay",
        "ap02": "primary candidate; static-only diagnostic and combined bundle adjustment",
        "ap03": "primary candidate; one COLMAP reconstruction, single and multi scale",
    }

    def show_choices() -> None:
        for index, method in enumerate(methods, 1):
            typer.echo(
                f"  {index}. {method.id.upper()} — {explanations[method.id]}"
            )

    while True:
        console.print(
            Panel(
                "Each selected method becomes one queue row and one separate run. "
                "Defaults are the current baseline. Root camera and method markers "
                "are reviewed together after static and moving observations exist. "
                "Duplicate any row to test another configuration without overwriting it.",
                title="Choose calibration jobs",
            )
        )
        show_choices()
        default_numbers = ",".join(
            str(index)
            for index, method in enumerate(methods, 1)
            if method.id in recommended_ids
        )
        raw = typer.prompt(
            "Method numbers for the baseline queue "
            "(comma-separated; 0/b = back)",
            default=default_numbers,
        ).strip()
        if raw.lower() in {"0", "b", "back"}:
            _clear_terminal()
            raise WizardBack()
        try:
            numbers = list(dict.fromkeys(int(value.strip()) for value in raw.split(",")))
        except ValueError:
            _show_input_error(
                "Use comma-separated method numbers, for example 1,2,3."
            )
            continue
        if not numbers or min(numbers) < 1 or max(numbers) > len(methods):
            typer.echo(f"Choose method numbers between 1 and {len(methods)}.")
            continue
        jobs = [
            _new_method_job(
                methods[number - 1].id,
                prompt_for_single_marker=True,
            )
            for number in numbers
        ]
        while True:
            _show_method_queue(console, jobs)
            action = _choice(
                "Queue action",
                {
                    "1": "accept this queue and continue",
                    "2": "add another method job",
                    "3": "duplicate a job (best for ablations/parameter comparisons)",
                    "4": "edit one method job (quality, method and COLMAP settings)",
                    "5": "rename one method job",
                    "6": "edit queue-wide ArUco input",
                    "7": "edit queue-wide common evaluation",
                    "8": "remove jobs (comma-separated or all)",
                    "0": "back to method selection",
                },
                "1",
            )
            if action == "1":
                labels = [job.label for job in jobs]
                if len(labels) != len(set(labels)):
                    typer.echo("Run labels must be unique; edit the duplicated labels.")
                    continue
                if any(
                    job.markers != jobs[0].markers
                    or job.evaluation != jobs[0].evaluation
                    for job in jobs[1:]
                ):
                    typer.echo(
                        "ArUco input and common evaluation belong to the queue. "
                        "Use actions 6 and 7 to edit the queue-wide values."
                    )
                    continue
                return jobs
            if action == "0":
                break
            if action == "2":
                show_choices()
                number = _prompt_index(
                    "Method number (0/b = back)",
                    maximum=len(methods),
                )
                if number is None:
                    continue
                new_job = _new_method_job(
                    methods[number - 1].id,
                    prompt_for_single_marker=True,
                )
                new_job.markers = jobs[0].markers.model_copy(deep=True)
                new_job.evaluation = jobs[0].evaluation.model_copy(deep=True)
                new_job.label = safe_id(
                    typer.prompt("New run label", default=new_job.label)
                )
                jobs.append(new_job)
            elif action == "3":
                number = _prompt_index(
                    "Queue row to duplicate (0/b = back)",
                    maximum=len(jobs),
                )
                if number is None:
                    continue
                source = jobs[number - 1]
                label = typer.prompt(
                    "Label for the copied variant",
                    default=f"{source.label}_variant2",
                )
                jobs.append(_clone_method_job(source, label))
            elif action == "4":
                number = _prompt_index(
                    "Queue row to edit (0/b = back)",
                    maximum=len(jobs),
                )
                if number is None:
                    continue
                current = jobs[number - 1]
                candidate = _clone_method_job(current, current.label)
                try:
                    jobs[number - 1] = _edit_method_job(
                        console, candidate
                    )
                except (ValidationError, ValueError, TypeError, yaml.YAMLError) as exc:
                    _show_input_error(
                        f"Invalid method setting: {exc}. "
                        "The previous row was kept unchanged."
                    )
            elif action == "5":
                number = _prompt_index(
                    "Queue row to rename (0/b = back)",
                    maximum=len(jobs),
                )
                if number is None:
                    continue
                jobs[number - 1].label = safe_id(
                    typer.prompt(
                        "New run label",
                        default=jobs[number - 1].label,
                    )
                )
            elif action == "6":
                candidate = _clone_method_job(jobs[0], jobs[0].label)
                try:
                    source = _edit_method_job(
                        console,
                        candidate,
                        groups={"ARUCO INPUT"},
                        title="Queue-wide ArUco input",
                    )
                except (ValidationError, ValueError, TypeError, yaml.YAMLError) as exc:
                    _show_input_error(
                        f"Invalid ArUco setting: {exc}. "
                        "Queue values were kept unchanged."
                    )
                    continue
                for target in jobs:
                    target.markers = source.markers.model_copy(deep=True)
                typer.echo(
                    "Applied the ArUco input to every queue job."
                )
            elif action == "7":
                candidate = _clone_method_job(jobs[0], jobs[0].label)
                try:
                    source = _edit_method_job(
                        console,
                        candidate,
                        groups={"COMMON EVALUATION"},
                        title="Queue-wide common evaluation",
                    )
                except (ValidationError, ValueError, TypeError, yaml.YAMLError) as exc:
                    _show_input_error(
                        f"Invalid evaluation setting: {exc}. "
                        "Queue values were kept unchanged."
                    )
                    continue
                for target in jobs:
                    target.evaluation = source.evaluation.model_copy(deep=True)
                typer.echo(
                    "Applied the common evaluation settings to every queue job."
                )
            elif action == "8":
                value = typer.prompt(
                    "Queue rows to remove "
                    "(comma-separated, all, or 0/b = back)"
                ).strip().lower()
                if value in {"0", "b", "back"}:
                    _clear_terminal()
                    continue
                if value == "all":
                    if typer.confirm(
                        "Remove every queue job and return to method selection?",
                        default=False,
                    ):
                        jobs.clear()
                        break
                    continue
                try:
                    numbers = sorted(
                        set(
                            int(item.strip())
                            for item in value.split(",")
                            if item.strip()
                        ),
                        reverse=True,
                    )
                except ValueError:
                    _show_input_error(
                        "Use comma-separated row numbers or 'all'."
                    )
                    continue
                if (
                    not numbers
                    or min(numbers) < 1
                    or max(numbers) > len(jobs)
                ):
                    _show_input_error("Invalid queue row selection.")
                    continue
                for number in numbers:
                    jobs.pop(number - 1)
                if not jobs:
                    typer.echo("The queue needs at least one job; returning to method selection.")
                    break


def _prompt_component_options(display_name: str, model_class: type) -> dict:
    try:
        defaults = model_class().model_dump(mode="python")
    except ValidationError:
        defaults = {}
    default_text = yaml.safe_dump(defaults, default_flow_style=True).strip()
    while True:
        value = typer.prompt(
            f"{display_name} options as a YAML mapping",
            default=default_text,
        )
        try:
            payload = yaml.safe_load(value)
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                raise ValueError("options must be a mapping")
            return model_class.model_validate(payload).model_dump(mode="python")
        except (ValidationError, ValueError, yaml.YAMLError) as exc:
            typer.echo(f"Invalid options: {exc}")


def _optional_marker(value: str) -> str | int:
    value = value.strip().lower()
    return value if value == "auto" else int(value)


def _base_project(
    repository_root: Path,
    run_label: str = "baseline",
    execution_mode: str = "complete",
) -> ProjectSettings:
    return ProjectSettings(
        workspace_root=repository_root / "workspace",
        dataset_cache_root=repository_root / "datasets",
        output_root=repository_root / "results",
        run_label=run_label,
        execution_mode=execution_mode,
    )


def _create_intrinsic_profile_only(
    repository_root: Path, console: Console
) -> None:
    input_root = repository_root / "data_local"
    sources = _checkerboard_sources(input_root)
    local_images_exist = any(
        item.kind == "image"
        for item in discover_inputs(input_root, recursive=True)
    )
    if not sources and not local_images_exist:
        console.print(
            Panel(
                "Place a checkerboard video or image folder below "
                "data_local/<id>/ and start this action again. Recommended "
                "folder names: intrinsics_images/ or checkerboard/.",
                title="No checkerboard input detected",
            )
        )
        return
    video, images = _select_checkerboard_source(
        input_root,
        sources,
    )
    source = video or images
    assert source is not None
    profile_id = typer.prompt(
        "New intrinsics profile ID", default=safe_id(source.stem)
    ).strip()
    columns = 8
    rows = 6
    maximum_views = 80
    minimum_gap = 0 if images is not None else 5
    minimum_detections = 20
    scan = IntrinsicScanSettings()
    if typer.confirm(
        "Open advanced checkerboard calibration settings?", default=False
    ):
        columns = typer.prompt("Inner corner columns", default=8, type=int)
        rows = typer.prompt("Inner corner rows", default=6, type=int)
        maximum_views = typer.prompt(
            "Maximum selected views", default=80, type=int
        )
        minimum_gap = typer.prompt("Minimum frame gap", default=5, type=int)
        minimum_detections = typer.prompt(
            "Minimum detections", default=20, type=int
        )
        mode = _prompt_enum_choice(
            "Checkerboard scan mode",
            "balanced",
            (
                ("balanced", "fast adaptive scan with 4K refinement"),
                (
                    "exhaustive_compatibility",
                    "legacy every-frame full-resolution scan",
                ),
            ),
        )
        target_hz = 3.0
        preview = 1920
        if mode == "balanced":
            target_hz = typer.prompt(
                "Initial checkerboard scan rate [Hz]",
                default=3.0,
                type=float,
            )
            preview = typer.prompt(
                "Preview maximum dimension [px]",
                default=1920,
                type=int,
            )
        scan = IntrinsicScanSettings(
            mode=mode,
            target_hz=target_hz,
            preview_max_dimension=preview,
        )
    destination = (
        repository_root
        / "workspace"
        / "intrinsics_profile_exports"
        / f"{safe_id(profile_id)}.json"
    )
    command = [
        sys.executable,
        "-m",
        "camera_rig_calibration.input.intrinsics",
        "--script",
        str(
            repository_root
            / "run/real_vehicle_data/02_calibrate_intrinsics_from_video.py"
        ),
        "--video" if video is not None else "--images",
        str(source),
        "--work-directory",
        str(destination.parent / f".{safe_id(profile_id)}_work"),
        "--destination",
        str(destination),
        "--camera-id",
        "moving_calib_camera",
        "--repository",
        str(repository_root),
        "--profile-id",
        profile_id,
        "--cols",
        str(columns),
        "--rows",
        str(rows),
        "--max-views",
        str(maximum_views),
        "--minimum-frame-gap",
        str(minimum_gap),
        "--minimum-detections",
        str(minimum_detections),
        "--scan-mode",
        scan.mode,
        "--scan-target-hz",
        str(scan.target_hz),
        "--preview-max-dimension",
        str(scan.preview_max_dimension),
    ]
    source_text = (
        f"Video: {video}"
        if video is not None
        else f"Checkerboard images: {images}"
    )
    console.print(
        Panel(
            f"{source_text}\n"
            f"Profile: {profile_id}\n"
            f"Scan: {scan.mode}, initial {scan.target_hz:g} Hz, "
            f"preview max {scan.preview_max_dimension}px",
            title="Intrinsic profile summary",
        )
    )
    if not typer.confirm("Create this reusable intrinsics profile?", default=True):
        return
    try:
        subprocess.run(command, cwd=repository_root, check=True)
    except subprocess.CalledProcessError as exc:
        console.print(
            f"[red]Intrinsic profile generation failed with exit code "
            f"{exc.returncode}. No profile was published.[/red]"
        )
        return
    profiles = [
        profile
        for profile in discover_intrinsic_profiles(repository_root)
        if profile.profile_id == safe_id(profile_id)
    ]
    if profiles:
        console.print(
            f"[green]Profile ready: {profiles[-1].key}\n"
            f"{profiles[-1].root}[/green]"
        )


def _new_dataset_id(repository_root: Path, suggested: str) -> str:
    exists = any(
        (repository_root / directory / suggested).exists()
        for directory in ("workspace", "datasets", "results")
    )
    if not exists:
        return suggested
    return f"{suggested}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def new_calibration_wizard(
    repository_root: Path, console: Console
) -> WizardOutcome | None:
    register_builtin_components()
    console.print(Panel("One guided setup produces one reproducible configuration.", title="New calibration"))
    prepared_inventory = discover_prepared_datasets(repository_root)
    raw_inventory = discover_raw_input_folders(repository_root)
    _show_input_inventory(
        repository_root, console, prepared_inventory, raw_inventory
    )
    real_prepared = [
        item for item in prepared_inventory if item.category == "real_vehicle"
    ]
    other_prepared = [
        item for item in prepared_inventory if item.category == "other"
    ]
    while True:
        mode = _choice(
            "Input type",
            {
                "1": (
                    "real data from data_local or prepared recordings "
                    f"({len(raw_inventory)} local, {len(real_prepared)} prepared; "
                    "recommended)"
                ),
                "2": "Gazebo simulation — baseline, existing ablation, or new combination",
                "3": "other prepared canonical dataset / manual path",
                "0": "back to main menu",
            },
            "1",
        )
        if mode == "0":
            return None
        prepared_root = None
        mcap = McapSettings()
        simulation = SimulationSettings()
        scene_type = SceneType.OTHER
        category = DatasetCategory.REAL_VEHICLE
        source_kind = InputSourceKind.PREPARED
        reused_existing_simulation = False
        try:
            if mode == "1":
                real_action = _choice(
                    "Real data",
                    {
                        "1": "start calibration from an acquisition (recommended)",
                        "2": "create or recalculate moving-camera intrinsics only",
                        "0": "back to input type",
                    },
                    "1",
                )
                if real_action == "0":
                    continue
                if real_action == "2":
                    _create_intrinsic_profile_only(repository_root, console)
                    continue
                acquisition_source = _choice(
                    "Acquisition source",
                    {
                        "1": (
                            "data_local recording, videos or frames "
                            f"({len(raw_inventory)} detected; recommended)"
                        ),
                        "2": (
                            "prepared real data "
                            f"({len(real_prepared)} detected; reuse frames)"
                        ),
                        "0": "back to real data",
                    },
                    "1",
                )
                if acquisition_source == "0":
                    continue
                if acquisition_source == "1":
                    (
                        input_root,
                        cameras,
                        moving,
                        sampling,
                        mcap,
                        suggested_id,
                    ) = _real_data_input(
                        repository_root, console, raw_inventory
                    )
                    source_kind = (
                        InputSourceKind.VIDEO
                        if moving.video is not None
                        else InputSourceKind.FRAMES
                        if moving.frames is not None
                        else InputSourceKind.ROSBAG
                    )
                else:
                    _show_prepared_choices(
                        repository_root,
                        console,
                        real_prepared,
                        title="Reusable real-vehicle datasets",
                    )
                    (
                        prepared_root,
                        cameras,
                        moving,
                        suggested_id,
                    ) = _prepared_input(
                        console, repository_root, real_prepared
                    )
                    moving = _prepared_moving_intrinsics(
                        console,
                        repository_root,
                        prepared_root,
                        moving,
                    )
                    stored_hz, _ = _stored_prepared_sampling(prepared_root)
                    sampling = SamplingSettings(target_hz=stored_hz)
                    input_root = prepared_root
                    source_kind = InputSourceKind.PREPARED
            elif mode == "2":
                cameras, moving, simulation, suggested_id, reused_root = _simulation_input(
                    repository_root, console
                )
                sampling = SamplingSettings(target_hz=None)
                if reused_root is not None:
                    prepared_root = reused_root
                    reused_existing_simulation = True
                    simulation = simulation.model_copy(update={"enabled": False})
                    input_root = reused_root
                else:
                    input_root = (
                        simulation.world.parent
                        if simulation.world is not None
                        else None
                    )
                scene_type = SceneType.SIMULATION
                category = DatasetCategory.SIMULATION
                source_kind = InputSourceKind.PREPARED
            else:
                _show_prepared_choices(
                    repository_root,
                    console,
                    other_prepared,
                    title="Other prepared datasets",
                )
                prepared_root, cameras, moving, suggested_id = _prepared_input(
                    console, repository_root, other_prepared
                )
                stored_hz, _ = _stored_prepared_sampling(prepared_root)
                sampling = SamplingSettings(target_hz=stored_hz)
                input_root = prepared_root
            break
        except WizardBack:
            continue

    default_dataset_id = (
        suggested_id
        if scene_type is SceneType.SIMULATION
        else _new_dataset_id(repository_root, suggested_id)
    )
    dataset_id = typer.prompt(
        "Dataset / experiment ID", default=default_dataset_id
    ).strip()
    execution_mode = "complete"
    if simulation.enabled:
        action = _choice(
            "After simulation capture",
            {
                "1": "run the complete calibration pipeline",
                "2": "prepare and validate inputs only (no AP methods)",
            },
            "1",
        )
        execution_mode = "prepare_only" if action == "2" else "complete"
    elif reused_existing_simulation:
        action = _choice(
            "Existing simulation dataset selected",
            {
                "1": "run a new calibration configuration on the reused frames",
                "2": "validate/review only; do not execute AP methods",
            },
            "2",
        )
        execution_mode = "prepare_only" if action == "2" else "complete"
    if execution_mode == "prepare_only":
        console.print(
            "Prepare-only mode: AP methods are not scheduled; baseline method "
            "defaults remain stored for a later complete experiment."
        )
        jobs: list[MethodQueueJob] = []
    else:
        try:
            jobs = _method_queue(console)
        except WizardBack:
            return new_calibration_wizard(repository_root, console)
    common = {
        "dataset": DatasetSettings(
            id=dataset_id,
            category=category,
            source_kind=source_kind,
            scene_type=scene_type,
            prepared_root=prepared_root,
            input_root=input_root,
        ),
        "static_cameras": cameras,
        "moving_camera": moving,
        "mcap": mcap,
        "simulation": simulation,
        "sampling": sampling,
    }
    if execution_mode == "prepare_only":
        config = RigConfig(
            project=_base_project(repository_root, execution_mode=execution_mode),
            methods=MethodSettings(),
            **common,
        )
        path = config.project.workspace_root / dataset_id / "rigcal.yaml"
        save_config(config, path)
        return WizardOutcome(config, path)

    queued: list[QueuedRun] = []
    queue_root = repository_root / "workspace" / dataset_id / "queue"
    for index, job in enumerate(jobs, 1):
        config = RigConfig(
            project=_base_project(repository_root, run_label=job.label),
            markers=job.markers,
            observation_quality=job.observation_quality,
            colmap=job.colmap,
            methods=job.methods,
            evaluation=job.evaluation,
            **common,
        )
        path = queue_root / f"{index:02d}_{job.label}.yaml"
        save_config(config, path)
        queued.append(QueuedRun(config, path))
    manifest = {
        "kind": "rigcal_queue",
        "schema_version": 5,
        "id": f"{dataset_id}_queue",
        "continue_independent": True,
        "common": {
            "dataset": queued[0].config.dataset.model_dump(
                mode="json", exclude_none=True
            ),
            "aruco": queued[0].config.markers.model_dump(
                mode="json", exclude_none=True
            ),
            "evaluation": queued[0].config.evaluation.model_dump(
                mode="json", exclude_none=True
            ),
        },
        "entries": [
            {
                "id": queued_run.config.project.run_label,
                "config": queued_run.path.name,
                "depends_on": [],
            }
            for queued_run in queued
        ],
    }
    queue_root.mkdir(parents=True, exist_ok=True)
    serialized_queue = yaml.safe_dump(
        manifest, sort_keys=False, allow_unicode=True
    )
    for name in ("queue.yaml", "queue_manifest.yaml"):
        (queue_root / name).write_text(
            serialized_queue, encoding="utf-8"
        )
    first, *rest = queued
    return WizardOutcome(first.config, first.path, tuple(rest))


def _config_candidates(repository_root: Path) -> list[Path]:
    register_builtin_components()
    candidates = list((repository_root / "workspace").glob("**/*.yaml"))
    candidates += [
        path
        for path in (repository_root / "results").rglob(
            "resolved_config.yaml"
        )
        if "run_history" not in path.parts
    ]
    unique: dict[str, Path] = {}
    for path in sorted(set(path.resolve() for path in candidates)):
        try:
            config = load_config(path)
            matching = [
                adapter
                for adapter in input_adapters
                if adapter.matches(config)
            ]
            if (
                not matching
                or not matching[0].requirements(config).compatible
            ):
                continue
            fingerprint = config_fingerprint(config)
        except Exception:
            # Queue manifests and unrelated YAML metadata are not runnable
            # RigConfig files and must never appear as broken saved setups.
            continue
        current = unique.get(fingerprint)
        if current is None or (
            "workspace" in path.parts,
            str(path),
        ) > ("workspace" in current.parts, str(current)):
            unique[fingerprint] = path
    return sorted(unique.values(), reverse=True)


def saved_setup_count(repository_root: Path) -> int:
    return len(_config_candidates(repository_root))


def _save_wizard_queue(
    directory: Path,
    queue_id: str,
    queued: list[QueuedRun],
) -> Path:
    if not queued:
        raise ValueError("A queue must contain at least one method job")
    dataset_ids = {run.config.dataset.id for run in queued}
    if len(dataset_ids) != 1:
        raise ValueError(
            "A schema-v5 queue contains exactly one dataset"
        )
    common = queued[0].config
    manifest = {
        "kind": "rigcal_queue",
        "schema_version": 5,
        "id": queue_id,
        "continue_independent": True,
        "common": {
            "dataset": common.dataset.model_dump(
                mode="json", exclude_none=True
            ),
            "aruco": common.markers.model_dump(
                mode="json", exclude_none=True
            ),
            "evaluation": common.evaluation.model_dump(
                mode="json", exclude_none=True
            ),
        },
        "entries": [
            {
                "id": (
                    f"{run.config.dataset.id}__"
                    f"{run.config.project.run_label}__{index:02d}"
                ),
                "config": run.path.name,
                "depends_on": [],
            }
            for index, run in enumerate(queued, 1)
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(
        manifest, sort_keys=False, allow_unicode=True
    )
    destination = directory / "queue.yaml"
    destination.write_text(serialized, encoding="utf-8")
    (directory / "queue_manifest.yaml").write_text(
        serialized, encoding="utf-8"
    )
    return destination


def choose_config(repository_root: Path, console: Console) -> tuple[RigConfig, Path]:
    paths = _config_candidates(repository_root)
    if not paths:
        raise RuntimeError("No saved setup exists yet. Start a new calibration first.")
    table = Table(title="Saved calibration setups")
    table.add_column("#")
    table.add_column("Configuration")
    for index, path in enumerate(paths, 1):
        table.add_row(str(index), str(path.relative_to(repository_root)))
    console.print(table)
    index = typer.prompt("Setup number", default=1, type=int)
    if index < 1 or index > len(paths):
        raise typer.BadParameter("Invalid setup number")
    return load_config(paths[index - 1]), paths[index - 1]


def repeat_setup_wizard(repository_root: Path, console: Console) -> WizardOutcome:
    config, source = choose_config(repository_root, console)
    queue_manifest = source.parent / "queue.yaml"
    if not queue_manifest.is_file():
        queue_manifest = source.parent / "queue_manifest.yaml"
    if queue_manifest.is_file() and typer.confirm(
        "This setup belongs to a saved queue. Repeat the complete queue?",
        default=True,
    ):
        payload = yaml.safe_load(queue_manifest.read_text(encoding="utf-8")) or {}
        definitions = payload.get("entries", payload.get("runs", []))
        loaded: list[RigConfig] = []
        for definition in definitions:
            candidate = Path(str(definition.get("config", "")))
            if not candidate.is_absolute():
                candidate = (queue_manifest.parent / candidate).resolve()
            loaded.append(load_config(candidate))
        if not loaded:
            raise RuntimeError(f"Saved queue has no runnable configs: {queue_manifest}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repeat_root = (
            config.project.workspace_root
            / config.dataset.id
            / "repeats"
            / f"queue_{stamp}"
        )
        queued: list[QueuedRun] = []
        for index, queued_config in enumerate(loaded, 1):
            destination = repeat_root / (
                f"{index:02d}_{queued_config.project.run_label}.yaml"
            )
            save_config(queued_config, destination)
            queued.append(QueuedRun(queued_config, destination))
        _save_wizard_queue(
            repeat_root,
            f"{config.dataset.id}_repeat_{stamp}",
            queued,
        )
        console.print(f"Template queue: {queue_manifest}")
        first, *rest = queued
        return WizardOutcome(first.config, first.path, tuple(rest))
    label = typer.prompt("New run label", default=config.project.run_label).strip()
    config = config.model_copy(
        update={"project": config.project.model_copy(update={"run_label": label})},
        deep=True,
    )
    destination = (
        config.project.workspace_root
        / config.dataset.id
        / "repeats"
        / f"{label}.yaml"
    )
    save_config(config, destination)
    console.print(f"Template: {source}")
    return WizardOutcome(config, destination)


def advanced_wizard(repository_root: Path, console: Console) -> WizardOutcome | None:
    register_builtin_components()
    action = _choice(
        "Advanced experiments",
        {
            "1": "clone and edit one method variant",
            "2": "list extension components",
            "3": "queue exhaustive and sequential COLMAP variants",
            "4": "build an overnight queue from saved setups/datasets",
            "0": "back to main menu",
        },
        "1",
    )
    if action == "0":
        return None
    if action == "2":
        table = Table(title="Registered extension components")
        table.add_column("Type")
        table.add_column("ID")
        table.add_column("Name")
        for method in calibration_methods:
            table.add_row("CalibrationMethod", method.id, method.display_name)
        for provider in experiment_providers:
            table.add_row("ExperimentProvider", provider.id, provider.display_name)
        console.print(table)
        return None
    if action == "4":
        candidates = _config_candidates(repository_root)
        if not candidates:
            raise RuntimeError(
                "No saved setups exist. Create a calibration setup first."
            )
        table = Table(title="Saved setups for an overnight queue")
        table.add_column("#", justify="right")
        table.add_column("Dataset")
        table.add_column("Method(s)")
        table.add_column("Run label")
        table.add_column("Configuration", overflow="fold")
        loaded = [load_config(path) for path in candidates]
        for index, (config, path) in enumerate(
            zip(loaded, candidates, strict=True), 1
        ):
            table.add_row(
                str(index),
                config.dataset.id,
                ", ".join(config.methods.enabled),
                config.project.run_label,
                str(path.relative_to(repository_root)),
            )
        console.print(table)
        raw = typer.prompt(
            "Setup numbers in queue order (comma-separated; 0 = back)"
        ).strip()
        if raw in {"0", "back", "b"}:
            return None
        numbers = [
            int(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
        if not numbers or any(
            number < 1 or number > len(candidates)
            for number in numbers
        ):
            raise typer.BadParameter("Invalid saved setup number list")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            repository_root
            / "workspace"
            / "queues"
            / f"overnight_{stamp}"
        )
        queued: list[QueuedRun] = []
        for number in numbers:
            source_config = loaded[number - 1]
            for method_id in source_config.methods.enabled:
                config = source_config.model_copy(
                    update={
                        "methods": source_config.methods.model_copy(
                            update={"enabled": [method_id]}, deep=True
                        )
                    },
                    deep=True,
                )
                label = safe_id(
                    f"{config.dataset.id}_{config.project.run_label}_{method_id}"
                )
                config = config.model_copy(
                    update={
                        "project": config.project.model_copy(
                            update={"run_label": label}
                        )
                    },
                    deep=True,
                )
                destination = (
                    directory / f"{len(queued) + 1:02d}_{label}.yaml"
                )
                save_config(config, destination)
                queued.append(QueuedRun(config, destination))
        _save_wizard_queue(
            directory, f"overnight_{stamp}", queued
        )
        console.print(
            f"Queued {len(queued)} independent method variants from "
            f"{len(numbers)} saved setup selections."
        )
        first, *rest = queued
        return WizardOutcome(first.config, first.path, tuple(rest))

    config, source = choose_config(repository_root, console)
    if action == "3":
        variants = experiment_providers.get("colmap_matcher").variants(config)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            config.project.workspace_root
            / config.dataset.id
            / "experiments"
            / f"colmap_matchers_{stamp}"
        )
        queued = []
        for label, variant in variants:
            destination = directory / f"{label}.yaml"
            save_config(variant, destination)
            queued.append(QueuedRun(variant, destination))
        _save_wizard_queue(
            directory,
            f"{config.dataset.id}_colmap_matchers_{stamp}",
            queued,
        )
        console.print(
            f"Created a two-row matcher queue in {directory}"
        )
        first, *rest = queued
        return WizardOutcome(first.config, first.path, tuple(rest))

    enabled = list(config.methods.enabled)
    if len(enabled) > 1:
        console.print(
            "This older setup contains several methods. A clean experiment row "
            "contains one method so variants cannot overwrite or hide each other."
        )
        for index, method_id in enumerate(enabled, 1):
            typer.echo(
                f"  {index}. {calibration_methods.get(method_id).display_name}"
            )
        selected = typer.prompt("Method to clone (0 = back)", default=1, type=int)
        if selected == 0:
            return None
        if selected < 1 or selected > len(enabled):
            raise typer.BadParameter("Invalid method number")
        method_id = enabled[selected - 1]
    else:
        method_id = enabled[0]
    methods = config.methods.model_copy(update={"enabled": [method_id]}, deep=True)
    default_label = f"{method_id}_experiment_01"
    label = typer.prompt("Experiment run label", default=default_label).strip()
    execution_choice = _choice(
        "Execution mode",
        {"1": "complete pipeline", "2": "prepare/validate inputs only"},
        "2" if config.project.execution_mode == "prepare_only" else "1",
    )
    execution_mode = "prepare_only" if execution_choice == "2" else "complete"
    job = MethodQueueJob(
        method_id=method_id,
        label=safe_id(label),
        methods=methods,
        markers=config.markers.model_copy(deep=True),
        observation_quality=config.observation_quality.model_copy(deep=True),
        colmap=config.colmap.model_copy(deep=True),
        evaluation=config.evaluation.model_copy(deep=True),
    )
    console.print(
        Panel(
            "All applicable values are shown together. Enter only the setting "
            "numbers you want to change; Enter keeps the cloned values.",
            title="Edit experiment row",
        )
    )
    job = _edit_method_job(console, job)
    config = config.model_copy(
        update={
            "project": config.project.model_copy(
                update={"run_label": job.label, "execution_mode": execution_mode}
            ),
            "methods": job.methods,
            "markers": job.markers,
            "observation_quality": job.observation_quality,
            "colmap": job.colmap,
            "evaluation": job.evaluation,
        },
        deep=True,
    )
    config = RigConfig.model_validate(config.model_dump(mode="python"))
    destination = (
        config.project.workspace_root
        / config.dataset.id
        / "experiments"
        / f"{job.label}.yaml"
    )
    save_config(config, destination)
    console.print(f"Template: {source}")
    return WizardOutcome(config, destination)


def show_summary(config: RigConfig, config_path: Path, console: Console) -> None:
    table = Table(title="Final calibration summary")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Configuration", str(config_path))
    table.add_row("Dataset", config.dataset.id)
    table.add_row("Static cameras", ", ".join(camera.id for camera in config.static_cameras))
    table.add_row("Moving camera", config.moving_camera.id)
    calibration_source = (
        config.moving_camera.intrinsic_calibration_video
        or config.moving_camera.intrinsic_calibration_images
    )
    table.add_row(
        "Moving intrinsics",
        (
            f"profile {config.moving_camera.intrinsics_profile}"
            if config.moving_camera.intrinsics_profile
            and calibration_source is None
            else (
                "calculate profile "
                f"{config.moving_camera.intrinsics_profile or 'auto'} from "
                f"{calibration_source} "
                f"({config.moving_camera.intrinsic_scan.mode})"
                if calibration_source is not None
                else str(config.moving_camera.intrinsics or "prepared CameraInfo")
            )
        ),
    )
    table.add_row(
        "Sampling",
        (
            "one frame per simulation route pose"
            if config.simulation.enabled
            else (
                f"{config.sampling.target_hz:g} Hz"
                if config.sampling.target_hz is not None
                else "unknown / prepared input"
            )
        ),
    )
    table.add_row(
        (
            "Stored baseline methods (not scheduled)"
            if config.project.execution_mode == "prepare_only"
            else "Methods"
        ),
        ", ".join(config.methods.enabled),
    )
    table.add_row("Execution", config.project.execution_mode)
    if config.dataset.scene_type is SceneType.SIMULATION:
        table.add_row(
            "Simulation parameters",
            format_simulation_parameters(
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
            ),
        )
        table.add_row(
            "Simulation camera scope",
            "route/resolution/FOV/blur affect the moving camera only; "
            "each static camera contributes one snapshot",
        )
        table.add_row(
            "Simulation intrinsics",
            "static and moving K/D come from SDF CameraInfo unless an explicit "
            "intrinsics file/profile was selected; explicit values are preserved",
        )
        table.add_row(
            "Simulation lighting",
            "world appearance only; may change pixels but never K/D",
        )
    table.add_row("Observations", "all passing quality checks")
    table.add_row(
        "Scientific selections",
        (
            f"AP01 root={config.methods.ap01.root_camera}; "
            f"AP02 ref={config.methods.ap02.reference_marker_id}; "
            f"AP03 single={config.methods.ap03.single.scale_marker_id}; "
            f"AP03 multi={config.methods.ap03.multi.marker_ids}; "
            f"evaluation={config.evaluation.anchor_marker_id}"
        ),
    )
    console.print(table)


def review_selection_candidates(
    config: RigConfig,
    resolved: ResolvedSelections,
    run_directory: Path,
    console: Console,
) -> dict[str, object]:
    """One attended checkpoint after all static and moving observations exist."""
    payload = resolved.payload
    console.print(
        Panel(
            "Intrinsics validate the camera models but do not select a coordinate "
            "origin. The recommendations below use the actual static and moving "
            "ArUco observation graph. These values are frozen before any AP method "
            "starts; the evaluation anchor is selected separately after methods.",
            title="One-time scientific selection review",
        )
    )
    labels = {
        camera.id: camera.label or ""
        for camera in config.static_cameras
    }
    selected_methods = set(config.methods.enabled)
    roots = payload["ap01_root_camera"]["candidates"]
    root_table = Table(title="AP01 root-camera candidates")
    root_table.add_column("#", justify="right")
    root_table.add_column("Camera ID")
    root_table.add_column("Label")
    root_table.add_column("Reachable")
    root_table.add_column("Direct")
    root_table.add_column("Moving bridges")
    root_table.add_column("Markers")
    root_table.add_column("Accepted")
    root_table.add_column("Median PnP RMSE")
    root_table.add_column("Median area")
    root_table.add_column("Status")
    for index, item in enumerate(roots, 1):
        status = (
            "recommended"
            if item.get("recommended")
            else "compatible"
            if item.get("compatible")
            else "not compatible"
        )
        rmse = item.get("median_pnp_reprojection_rmse_px")
        area = item.get("median_marker_area_px2")
        root_table.add_row(
            str(index),
            str(item["id"]),
            labels.get(str(item["id"]), ""),
            f"{len(item['reachable_cameras'])}/{len(config.static_cameras)}",
            str(len(item["direct_connections"])),
            str(len(item["moving_bridges"])),
            str(item["distinct_markers"]),
            str(item["observations"]),
            f"{float(rmse):.2f} px" if rmse is not None else "unknown",
            f"{float(area):.0f} px²" if area is not None else "unknown",
            status,
        )
    if "ap01" in selected_methods:
        console.print(root_table)

    marker_rows = payload["ap03_single_scale_marker"]["candidates"]
    marker_table = Table(title="Method-marker candidates")
    marker_table.add_column("Marker")
    marker_table.add_column("Static-only reached")
    marker_table.add_column("Combined reached")
    marker_table.add_column("Moving frames")
    marker_table.add_column("Static direct")
    marker_table.add_column("Accepted")
    marker_table.add_column("Median PnP RMSE")
    marker_table.add_column("Median area")
    if "ap02" in selected_methods:
        marker_table.add_column("AP02 reference")
    if "ap03" in selected_methods:
        marker_table.add_column("AP03 single scale")
        marker_table.add_column("AP03 multi scale")
    ap02_recommended = {
        int(item["id"])
        for item in payload["ap02_reference_marker"]["candidates"]
        if item.get("recommended")
    }
    for item in marker_rows:
        marker_id = int(item["id"])
        cells = [
            str(marker_id),
            f"{item['static_graph_reachable_count']}/{len(config.static_cameras)}",
            (
                f"{item['combined_graph_reachable_static_count']}/"
                f"{len(config.static_cameras)}"
            ),
            str(item["moving_frames"]),
            str(item["static_camera_count"]),
            str(item["accepted_observations"]),
            (
                f"{float(item['median_pnp_reprojection_rmse_px']):.2f} px"
                if item.get("median_pnp_reprojection_rmse_px") is not None
                else "unknown"
            ),
            (
                f"{float(item['median_marker_area_px2']):.0f} px²"
                if item.get("median_marker_area_px2") is not None
                else "unknown"
            ),
        ]
        if "ap02" in selected_methods:
            ap02_details = next(
                (
                    candidate
                    for candidate in payload[
                        "ap02_reference_marker"
                    ]["candidates"]
                    if int(candidate["id"]) == marker_id
                ),
                item,
            )
            cells.append(
                (
                    "recommended (diagnostic partial)"
                    if ap02_details.get("diagnostic_partial")
                    else "recommended"
                )
                if marker_id in ap02_recommended
                else (
                    "diagnostic partial"
                    if ap02_details.get("diagnostic_partial")
                    else "compatible"
                )
                if ap02_details.get("compatible")
                else "not compatible"
            )
        if "ap03" in selected_methods:
            cells.append(
                "recommended"
                if item.get("recommended")
                else "compatible"
                if item["ap03_compatible"]
                else "not compatible"
            )
            cells.append(
                "included by auto"
                if marker_id in resolved.ap03_multi_marker_ids
                else "not compatible"
            )
        marker_table.add_row(*cells)
    if selected_methods & {"ap02", "ap03"}:
        console.print(marker_table)

    root_default = next(
        index
        for index, item in enumerate(roots, 1)
        if str(item["id"]) == resolved.root_camera
    )
    root = resolved.root_camera
    ap02 = resolved.ap02_reference_marker_id
    single = resolved.ap03_single_scale_marker_id
    multi: str | list[int] = list(
        resolved.ap03_multi_marker_ids
    )
    automatic = Table(title="Automatic selections")
    automatic.add_column("Role")
    automatic.add_column("Resolution")
    if "ap01" in selected_methods:
        automatic.add_row(
            "AP01 root camera",
            f"{config.methods.ap01.root_camera} → {root} (recommended)",
        )
    if "ap02" in selected_methods:
        automatic.add_row(
            "AP02 reference marker",
            (
                f"{config.methods.ap02.reference_marker_id} → marker {ap02} "
                "(recommended)"
            ),
        )
    if "ap03" in selected_methods:
        automatic.add_row(
            "AP03 single scale marker",
            (
                f"{config.methods.ap03.single.scale_marker_id} → marker "
                f"{single} (recommended)"
            ),
        )
        automatic.add_row(
            "AP03 multi marker set",
            (
                f"{config.methods.ap03.multi.marker_ids} → "
                f"{len(resolved.ap03_multi_marker_ids)} compatible markers"
            ),
        )
    console.print(automatic)
    typer.echo("\nEnter  Accept automatic selections")
    typer.echo("O      Override")
    typer.echo("B      Back")
    while True:
        action = typer.prompt(
            "Selection review action",
            default="",
            show_default=False,
        ).strip().lower()
        if action in {"", "accept", "a"}:
            break
        if action in {"b", "back"}:
            _clear_terminal()
            raise WizardBack(
                "Selection review paused; prepared observations remain "
                f"resumable at {run_directory}"
            )
        if action not in {"o", "override"}:
            typer.echo("Press Enter to accept, O to override, or B to go back.")
            continue
        if "ap01" in selected_methods:
            root_raw = typer.prompt(
                "AP01 root camera (table number or exact camera ID; b = back)",
                default=str(root_default),
            ).strip()
            if root_raw.lower() in {"b", "back"}:
                continue
            if root_raw.isdigit() and 1 <= int(root_raw) <= len(roots):
                root = str(roots[int(root_raw) - 1]["id"])
            else:
                root = root_raw
        if "ap02" in selected_methods:
            raw = typer.prompt(
                "AP02 reference marker ID (b = back)",
                default=str(ap02),
            ).strip()
            if raw.lower() in {"b", "back"}:
                continue
            ap02 = int(raw)
        if "ap03" in selected_methods:
            raw = typer.prompt(
                "AP03 Single scale marker ID (b = back)",
                default=str(single),
            ).strip()
            if raw.lower() in {"b", "back"}:
                continue
            single = int(raw)
            multi_default = ",".join(
                str(value) for value in resolved.ap03_multi_marker_ids
            )
            raw = typer.prompt(
                "AP03 Multi marker IDs, comma-separated (b = back)",
                default=multi_default,
            ).strip()
            if raw.lower() in {"b", "back"}:
                continue
            multi = _parse_ids(raw)
        break
    if multi == "auto":
        multi = list(resolved.ap03_multi_marker_ids)
    choices: dict[str, object] = {
        "root_camera": root,
        "ap02_reference_marker_id": ap02,
        "ap03_single_scale_marker_id": single,
        "ap03_multi_marker_ids": multi,
    }
    summary = Table(title="Selections to freeze before the method queue")
    summary.add_column("Role")
    summary.add_column("Selected")
    if "ap01" in selected_methods:
        summary.add_row("AP01 root camera", root)
    if "ap02" in selected_methods:
        summary.add_row("AP02 reference marker", str(ap02))
    if "ap03" in selected_methods:
        summary.add_row("AP03 Single scale marker", str(single))
    if "ap03" in selected_methods:
        summary.add_row(
            "AP03 Multi marker set",
            ",".join(str(value) for value in multi),
        )
    summary.add_row(
        "Evaluation anchor",
        "auto_common after method outputs; evaluation-only rerun is possible",
    )
    console.print(summary)
    return choices


def show_queue_summary(outcome: WizardOutcome, console: Console) -> None:
    if len(outcome.runs) == 1:
        show_summary(outcome.config, outcome.path, console)
        return
    table = Table(title="Final calibration queue")
    table.add_column("#", justify="right")
    table.add_column("Run label")
    table.add_column("Method")
    table.add_column("Key configuration", overflow="fold")
    table.add_column("Saved config", overflow="fold")
    for index, queued in enumerate(outcome.runs, 1):
        method_id = queued.config.methods.enabled[0]
        job = MethodQueueJob(
            method_id=method_id,
            label=queued.config.project.run_label,
            methods=queued.config.methods,
            markers=queued.config.markers,
            observation_quality=queued.config.observation_quality,
            colmap=queued.config.colmap,
            evaluation=queued.config.evaluation,
        )
        table.add_row(
            str(index),
            queued.config.project.run_label,
            calibration_methods.get(method_id).display_name,
            _method_job_summary(job),
            str(queued.path),
        )
    console.print(table)
    console.print(
        f"Dataset: {outcome.config.dataset.id} | {len(outcome.runs)} independent runs | "
        "shared immutable input under "
        "datasets/<category>/<source-or-factor>/<experiment>/inputs/<input-id>"
    )


def show_results(repository_root: Path, console: Console) -> None:
    entries = index_results(repository_root / "results")
    if not entries:
        console.print("No new or historical result runs were found.")
        return
    table = Table(title="Calibration results", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Category", width=10)
    table.add_column("Experiment / dataset", ratio=2, overflow="fold")
    table.add_column("Status", ratio=1, overflow="fold")
    table.add_column("Method result / variant", ratio=2, overflow="fold")
    for index, entry in enumerate(entries, 1):
        status = (
            "input unavailable\nnot rerunnable"
            if entry.status == "input unavailable / not rerunnable"
            else entry.status
        )
        result_label = (
            ", ".join(entry.methods)
            + (f"\n{entry.variant}" if entry.variant else "")
        ).strip()
        if not result_label:
            result_label = (
                "historical result"
                if entry.legacy
                else entry.run_id
            )
        table.add_row(
            str(index),
            {
                "simulation": "SIMULATION",
                "real_vehicle": "REAL DATA",
            }.get(
                entry.category,
                "LEGACY" if entry.legacy else "",
            ),
            entry.experiment_id or entry.dataset_id,
            status,
            result_label,
        )
    console.print(table)
    selected = typer.prompt("Result number to inspect (0 = back)", default=0, type=int)
    if selected == 0:
        return
    if selected < 1 or selected > len(entries):
        raise typer.BadParameter("Invalid result number")
    entry = entries[selected - 1]
    console.print(Panel(str(entry.path), title=f"{entry.dataset_id} / {entry.run_id}"))
    removed_path = entry.path / "INPUT_REMOVED.json"
    if removed_path.is_file():
        removed = json.loads(removed_path.read_text(encoding="utf-8"))
        console.print(
            Panel(
                "Results available; input cleaned; not rerunnable.\n"
                f"Removed files: {removed.get('file_count', 'unknown')}\n"
                f"Recorded reclaimable size: "
                f"{_human_size(int(removed.get('reclaimable_bytes_estimate', 0)))}",
                title="Storage state",
            )
        )
    gallery_candidates = [
        *entry.path.glob(
            "datasets/*/observations/*/connectivity_report.json"
        ),
        *entry.path.glob(
            "observations/*/*/connectivity_report.json"
        ),
    ]
    if gallery_candidates:
        latest_gallery = sorted(gallery_candidates)[-1]
        try:
            gallery = json.loads(
                latest_gallery.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            gallery = {}
        if gallery:
            diagnostics = Table(title="Moving-frame ArUco diagnostics")
            diagnostics.add_column("Metric")
            diagnostics.add_column("Value")
            diagnostics.add_row(
                "Moving frames",
                str(gallery.get("total_moving_frames", "unknown")),
            )
            diagnostics.add_row(
                "Frames with detections",
                str(gallery.get("frames_with_detections", "unknown")),
            )
            diagnostics.add_row(
                "Frames without markers",
                str(gallery.get("frames_without_markers", "unknown")),
            )
            diagnostics.add_row(
                "Frames with multiple markers",
                str(
                    gallery.get(
                        "frames_with_multiple_markers", "unknown"
                    )
                ),
            )
            diagnostics.add_row(
                "AP02 bridge frames",
                str(gallery.get("ap02_bridge_frames", "unknown")),
            )
            diagnostics.add_row(
                "Gallery",
                (
                    str(gallery.get("gallery_path"))
                    if gallery.get("gallery_path")
                    and Path(str(gallery["gallery_path"])).is_dir()
                    else "cleaned"
                ),
            )
            console.print(diagnostics)
    manifest_path = entry.path / "run_manifest.json"
    dataset_manifest_path = entry.path / "00_INPUT" / "dataset_manifest.json"
    comparison_path = entry.path / "07_COMPARISON" / "method_status.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_manifest = (
            json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if dataset_manifest_path.is_file()
            else {}
        )
        detail = Table(title="Run details")
        detail.add_column("Field")
        detail.add_column("Value")
        detail.add_row("Status", str(manifest.get("status", "unknown")))
        detail.add_row(
            "Execution", str(manifest.get("execution_mode", "complete"))
        )
        simulation_parameters = manifest.get("simulation_parameters", {})
        if simulation_parameters:
            detail.add_row(
                "Simulation parameters",
                format_simulation_parameters(simulation_parameters),
            )
        method_label = (
            "Methods configured (not executed)"
            if manifest.get("execution_mode") == "prepare_only"
            else "Methods"
        )
        detail.add_row(
            method_label, ", ".join(manifest.get("enabled_methods", []))
        )
        detail.add_row(
            "Static cameras",
            str(len(dataset_manifest.get("static_cameras", []))),
        )
        detail.add_row(
            "Moving frames",
            str(dataset_manifest.get("moving_camera", {}).get("image_count", "unknown")),
        )
        runtime = sum(
            float(stage.get("runtime_seconds", 0.0) or 0.0)
            for stage in manifest.get("stages", [])
        )
        detail.add_row("Recorded runtime", f"{runtime:.1f} s")
        detail.add_row("Output", str(entry.path))
        console.print(detail)
    elif entry.category in {"simulation", "real_vehicle"}:
        detail = Table(title="Experiment method executions")
        detail.add_column("Method")
        detail.add_column("Status")
        detail.add_column("Variant", overflow="fold")
        detail.add_column("Input")
        detail.add_column("Output", overflow="fold")
        manifests = [
            path
            for path in entry.path.rglob("run_manifest.json")
            if "run_history" not in path.parts and "_views" not in path.parts
        ]
        for child in sorted(manifests):
            try:
                payload = json.loads(child.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            detail.add_row(
                ", ".join(payload.get("enabled_methods", [])) or "-",
                str(payload.get("status", "unknown")),
                str(payload.get("variant", "-")),
                str(payload.get("input_id", "-")),
                str(child.parent),
            )
        legacy_path = entry.path / "legacy_manifest.json"
        if legacy_path.is_file():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            detail.add_row(
                ", ".join(entry.methods) or "legacy",
                str(legacy.get("status", "legacy")),
                "migrated historical executions",
                str(legacy.get("input_id", "-")),
                str(entry.path / "legacy_results"),
            )
        console.print(detail)
    elif (entry.path / "legacy_manifest.json").is_file():
        legacy = json.loads(
            (entry.path / "legacy_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        detail = Table(title="Migrated historical result details")
        detail.add_column("Field")
        detail.add_column("Value", overflow="fold")
        detail.add_row(
            "Experiment",
            str(legacy.get("experiment_id", entry.dataset_id)),
        )
        detail.add_row("Category", str(legacy.get("category", "")))
        detail.add_row("Status", str(legacy.get("status", "")))
        detail.add_row("Input ID", str(legacy.get("input_id", "")))
        detail.add_row(
            "Rerunnable", str(bool(legacy.get("rerunnable", False)))
        )
        verification = legacy.get("input_verification", {})
        detail.add_row(
            "Input verification",
            (
                f"verified={verification.get('verified')}, "
                f"files={verification.get('file_count')}, "
                f"sha256={verification.get('sha256')}"
            ),
        )
        parameters = legacy.get("parameters", {})
        if parameters:
            detail.add_row(
                "Simulation parameters",
                format_simulation_parameters(parameters),
            )
        detail.add_row("Current v2 result", str(entry.path))
        detail.add_row(
            "Historical artifacts",
            str(entry.path / "legacy_results"),
        )
        console.print(detail)
    if comparison_path.is_file():
        rows = json.loads(comparison_path.read_text(encoding="utf-8"))
        methods = Table(title="Method results")
        methods.add_column("Method")
        methods.add_column("Status")
        methods.add_column("Cameras")
        methods.add_column("Moving")
        methods.add_column("Cross RMSE [px]")
        methods.add_column("Warning")
        for row in rows:
            methods.add_row(
                str(row.get("method", "")),
                str(row.get("status", "")),
                str(row.get("static_camera_count", "")),
                str(row.get("registered_moving_frames", "")),
                str(row.get("cross_camera_reprojection_rmse_px", "")),
                str(row.get("warning", "")),
            )
        console.print(methods)
    summary = entry.path / "99_FINAL_RESULTS" / "SUMMARY.txt"
    if summary.is_file():
        console.print(summary.read_text(encoding="utf-8"), markup=False)


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def cleanup_storage_wizard(
    repository_root: Path, console: Console
) -> None:
    console.print(
        Panel(
            "Cleanup removes:\n"
            "- generated dataset caches and inactive staging data\n"
            "- raw static and moving frames stored with experiments\n"
            "- ArUco debug images and galleries\n"
            "- temporary/COLMAP working-image copies\n"
            "- large intrinsic-calibration selected/debug images\n\n"
            "Cleanup keeps:\n"
            "- all method results and final result figures\n"
            "- reports, configs, manifests, logs and timings\n"
            "- observation CSV files and connectivity reports\n"
            "- numeric COLMAP reconstructions\n"
            "- reusable intrinsic parameters and profile metadata",
            title="Cleanup storage",
        )
    )
    plan = build_cleanup_plan(repository_root)
    table = Table(title="Generated input and working data")
    table.add_column("Kind")
    table.add_column("Targets", justify="right")
    grouped: dict[str, int] = {}
    for target in plan.targets:
        grouped[target.kind] = grouped.get(target.kind, 0) + 1
    for kind, count in sorted(grouped.items()):
        table.add_row(kind, str(count))
    table.add_row("Files", str(plan.file_count))
    table.add_row("Logical size", _human_size(plan.logical_bytes))
    table.add_row(
        "Actually reclaimable (hardlink-aware)",
        _human_size(plan.reclaimable_bytes),
    )
    console.print(table)
    if plan.protected_paths:
        console.print(
            f"[yellow]{len(plan.protected_paths)} active/resumable path(s) "
            "are protected and excluded.[/yellow]"
        )
    if plan.targets and typer.confirm(
        "Delete generated input and working data?", default=False
    ):
        result = execute_cleanup(plan)
        console.print(
            "[green]Generated data removed. Scientific results remain. "
            f"Estimated reclaimed space: "
            f"{_human_size(int(result['reclaimable_bytes_estimate']))}.[/green]"
        )
    elif not plan.targets:
        console.print("No generated input or working data is eligible for cleanup.")

    if not typer.confirm(
        "Also delete everything inside data_local/?", default=False
    ):
        return
    local_plan = build_data_local_cleanup_plan(repository_root)
    if not local_plan.targets:
        console.print("data_local is already empty.")
        return
    local_table = Table(title="data_local contents selected for deletion")
    local_table.add_column("Path", overflow="fold")
    local_table.add_column("Type")
    for target in local_plan.targets:
        local_table.add_row(
            str(target.path.relative_to(repository_root)),
            "directory" if target.path.is_dir() else "file",
        )
    local_table.add_row(
        f"{local_plan.file_count} files",
        _human_size(local_plan.reclaimable_bytes),
    )
    console.print(local_table)
    if not typer.confirm(
        "Permanently delete the listed data_local contents?", default=False
    ):
        console.print("data_local was kept.")
        return
    result = execute_cleanup(local_plan)
    console.print(
        "[green]data_local contents removed. Results and intrinsics profiles "
        f"remain. Estimated reclaimed space: "
        f"{_human_size(int(result['reclaimable_bytes_estimate']))}.[/green]"
    )


def manage_intrinsics_profiles(
    repository_root: Path, console: Console
) -> None:
    while True:
        profiles = discover_intrinsic_profiles(repository_root)
        table = Table(title="Moving-camera intrinsics profiles")
        table.add_column("#", justify="right")
        table.add_column("Name / stable key", overflow="fold")
        table.add_column("Resolution")
        table.add_column("Model")
        table.add_column("Size")
        table.add_column("References")
        references: dict[str, tuple[Path, ...]] = {}
        for index, profile in enumerate(profiles, 1):
            refs = intrinsic_profile_references(repository_root, profile)
            references[profile.key] = refs
            table.add_row(
                str(index),
                f"{profile.label}\n{profile.key}",
                f"{profile.width}x{profile.height}",
                profile.distortion_model,
                _human_size(profile.size_bytes),
                str(len(refs)),
            )
        if profiles:
            console.print(table)
        else:
            console.print("No managed intrinsics profiles were found.")
        action = _choice(
            "Intrinsics profiles",
            {
                "1": "create or recalculate a new immutable profile version",
                "2": "rename a profile display alias",
                "3": "delete a profile",
                "0": "back to main menu",
            },
            "1",
        )
        if action == "0":
            return
        if action == "1":
            _create_intrinsic_profile_only(repository_root, console)
            continue
        if not profiles:
            continue
        number = typer.prompt("Profile number", type=int)
        if number < 1 or number > len(profiles):
            _show_input_error("Invalid profile number.")
            continue
        profile = profiles[number - 1]
        if action == "2":
            alias = typer.prompt(
                "New display name", default=profile.label
            ).strip()
            updated = update_profile_alias(profile, alias)
            console.print(
                f"[green]Display name updated to '{updated.label}'. Stable "
                f"key remains {updated.key}.[/green]"
            )
            continue
        refs = references[profile.key]
        if refs:
            console.print(
                f"Profile is referenced by {len(refs)} saved configuration(s). "
                "Completed scientific results will remain, but a configuration "
                "without a published intrinsics snapshot may no longer be rerunnable."
            )
            for reference in refs[:10]:
                console.print(f"  - {reference}")
            if len(refs) > 10:
                console.print(f"  ... and {len(refs) - 10} more")
        if typer.confirm(
            f"Permanently delete profile {profile.key}?",
            default=False,
        ):
            try:
                delete_profile(repository_root, profile)
            except RuntimeError as exc:
                _show_input_error(str(exc))
                continue
            console.print(
                "[green]Profile deleted. Completed method results were kept.[/green]"
            )


def show_doctor(repository_root: Path, console: Console) -> None:
    checks = run_checks(repository_root, needs_ros=True)
    table = Table(title="Installation check")
    table.add_column("Status")
    table.add_column("Component")
    table.add_column("Detail")
    for check in checks:
        table.add_row("OK" if check.ok else "MISSING", check.name, check.detail)
    console.print(table)
    required_failures = [check for check in checks if check.required and not check.ok]
    if required_failures:
        console.print("Required components are missing. See the installation guide in README.md.")
    else:
        console.print("Installation is ready for the configured baseline methods.")


def _manifest_process_is_active(manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        pid = int(payload.get("runner_pid") or 0)
    except Exception:
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, ValueError):
        return False
    return b"rigcal" in command or b"camera_rig_calibration" in command


def _migrate_legacy_staging(repository_root: Path) -> None:
    """Move inactive pre-v5 staging runs out of published result trees."""
    root = repository_root.resolve()
    results_root = root / "results"
    temporary_root = root / "workspace" / "temporary_runs"
    if not results_root.is_dir():
        return
    for staging in sorted(results_root.rglob(".staging")):
        if not staging.is_dir():
            continue
        try:
            relative = staging.parent.relative_to(results_root)
        except ValueError:
            continue
        for child in sorted(staging.iterdir()):
            if not child.is_dir():
                continue
            manifests = list(child.rglob("run_manifest.json"))
            if any(_manifest_process_is_active(path) for path in manifests):
                continue
            queue_id = safe_id(
                "legacy_" + "_".join((*relative.parts, child.name))
            )
            destination = temporary_root / queue_id
            suffix = 2
            while destination.exists():
                destination = temporary_root / f"{queue_id}_{suffix}"
                suffix += 1
            file_inventory = []
            for path in sorted(item for item in child.rglob("*") if item.is_file()):
                file_hash = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(
                        lambda: handle.read(8 * 1024 * 1024), b""
                    ):
                        file_hash.update(chunk)
                digest = file_hash.hexdigest()
                file_inventory.append(
                    {
                        "path": str(path.relative_to(child)),
                        "size_bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )
            destination.mkdir(parents=True, exist_ok=False)
            migrated_run = destination / "jobs" / "legacy"
            migrated_run.parent.mkdir(parents=True, exist_ok=True)
            child.rename(migrated_run)
            transaction = {
                "schema_version": 5,
                "queue_id": destination.name,
                "status": "incomplete",
                "legacy_single_run": True,
                "migrated_from": str(child),
                "migrated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "file_count": len(file_inventory),
                "size_bytes": sum(
                    int(row["size_bytes"]) for row in file_inventory
                ),
                "inventory": file_inventory,
            }
            temporary = destination / "queue_transaction.json.tmp"
            temporary.write_text(
                json.dumps(transaction, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination / "queue_transaction.json")
        if not any(staging.iterdir()):
            staging.rmdir()


def _transaction_payload(path: Path, name: str) -> dict:
    candidate = path / name
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def incomplete_runs(repository_root: Path) -> list[ResultEntry]:
    _migrate_legacy_staging(repository_root)
    entries: list[ResultEntry] = []
    temporary_root = (
        repository_root.resolve() / "workspace" / "temporary_runs"
    )
    if not temporary_root.is_dir():
        return entries
    for transaction in sorted(
        path for path in temporary_root.iterdir() if path.is_dir()
    ):
        state = _transaction_payload(transaction, "queue_state.json")
        journal = _transaction_payload(
            transaction, "queue_transaction.json"
        )
        manifests: list[dict] = []
        for manifest_path in transaction.rglob("run_manifest.json"):
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                manifests.append(payload)
        if not state and not journal and not manifests:
            continue
        statuses = [
            str(row.get("status", "pending"))
            for row in state.get("entries", {}).values()
            if isinstance(row, dict)
        ]
        statuses.extend(
            str(payload.get("status", "unknown")) for payload in manifests
        )
        journal_status = str(journal.get("status", ""))
        if "running" in statuses:
            status = "running"
        elif journal_status == "publication_failed":
            status = "publication_failed"
        elif "failed_preflight" in statuses:
            status = "failed_preflight"
        elif "failed" in statuses:
            status = "failed"
        elif "interrupted" in statuses:
            status = "interrupted"
        elif "waiting_for_selection" in statuses:
            status = "waiting_for_selection"
        else:
            status = journal_status or "incomplete"
        first = manifests[0] if manifests else {}
        methods = tuple(
            dict.fromkeys(
                method
                for payload in manifests
                for method in payload.get("enabled_methods", [])
            )
        )
        entries.append(
            ResultEntry(
                dataset_id=str(
                    first.get("dataset_id")
                    or state.get("dataset_id")
                    or transaction.name
                ),
                run_id=str(
                    journal.get("queue_id")
                    or state.get("queue_id")
                    or transaction.name
                ),
                status=status,
                path=transaction,
                methods=methods,
                category=str(first.get("result_category", "")),
                experiment_id=str(first.get("experiment_id", "")),
            )
        )
    return entries


def _active_run_stage(run: Path) -> str:
    for manifest_path in sorted(run.rglob("run_manifest.json")):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        for stage in manifest.get("stages", []):
            if stage.get("status") in {
                "running",
                "failed",
                "interrupted",
            }:
                return str(stage.get("id", "unknown"))
        for stage in manifest.get("stages", []):
            if stage.get("status") == "pending":
                return str(stage.get("id", "unknown"))
    return "unknown"


def _run_process_is_active(run: Path) -> bool:
    return any(
        _manifest_process_is_active(path)
        for path in run.rglob("run_manifest.json")
    )


def _validated_incomplete_run(repository_root: Path, entry: ResultEntry) -> Path:
    temporary_root = (
        repository_root / "workspace" / "temporary_runs"
    ).resolve()
    run = entry.path.resolve()
    try:
        relative = run.relative_to(temporary_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to modify a run outside workspace/temporary_runs/: {run}"
        ) from exc
    if len(relative.parts) != 1:
        raise RuntimeError(f"Refusing unexpected run path: {run}")
    if not (
        (run / "queue_transaction.json").is_file()
        or (run / "queue_state.json").is_file()
        or any(run.rglob("run_manifest.json"))
    ):
        raise RuntimeError(f"Temporary queue metadata is missing: {run}")
    if _run_process_is_active(run):
        raise RuntimeError(
            "This run still has an active rigcal process. Press Ctrl+C in its original "
            "terminal first, then open Manage incomplete runs again."
        )
    return run


def _interrupt_incomplete_run(run: Path) -> None:
    manifest_paths = list(run.rglob("run_manifest.json"))
    active_pids: set[int] = set()
    for manifest_path in manifest_paths:
        if not _manifest_process_is_active(manifest_path):
            continue
        try:
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            active_pids.add(int(payload.get("runner_pid") or 0))
        except Exception:
            continue
    for pid in sorted(value for value in active_pids if value > 0):
        os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _run_process_is_active(run):
        time.sleep(0.1)
    if _run_process_is_active(run):
        for manifest_path in manifest_paths:
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                pid = int(payload.get("runner_pid") or 0)
            except Exception:
                continue
            if _manifest_process_is_active(manifest_path):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _run_process_is_active(run):
            time.sleep(0.1)
    if _run_process_is_active(run):
        raise RuntimeError(
            "Could not stop every active rigcal process; temporary files "
            "were kept unchanged"
        )
    interrupted_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") not in {"completed", "duplicate_skipped"}:
            manifest["status"] = "interrupted"
        manifest["runner_pid"] = None
        manifest["interrupted_at"] = interrupted_at
        for stage in manifest.get("stages", []):
            if stage.get("status") == "running":
                stage["status"] = "interrupted"
                stage["error"] = "Interrupted from Manage incomplete runs"
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(manifest_path)
    state_path = run / "queue_state.json"
    if state_path.is_file():
        state = _transaction_payload(run, "queue_state.json")
        for row in state.get("entries", {}).values():
            if (
                isinstance(row, dict)
                and row.get("status")
                not in {"completed", "duplicate_skipped"}
            ):
                row["status"] = "interrupted"
        state["updated_at"] = interrupted_at
        temporary = state_path.with_name(state_path.name + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(state_path)


def _delete_incomplete_run(
    repository_root: Path,
    entry: ResultEntry,
    *,
    delete_private_inputs: bool,
) -> list[Path]:
    run = _validated_incomplete_run(repository_root, entry)
    if _run_process_is_active(run):
        raise RuntimeError(
            "This temporary queue is active. Stop it before deletion."
        )
    shutil.rmtree(run)
    return [run]


def incomplete_resume_source(transaction: Path) -> tuple[str, Path]:
    """Return ``("queue"|"run", path)`` for an incomplete transaction."""
    queue = transaction / "requested_queue.yaml"
    if queue.is_file():
        return "queue", queue
    resolved_queue = transaction / "resolved" / "queue.yaml"
    if resolved_queue.is_file():
        return "queue", resolved_queue
    manifests = sorted(transaction.rglob("run_manifest.json"))
    if not manifests:
        raise RuntimeError(
            f"Temporary transaction has no resumable queue or run: {transaction}"
        )
    incomplete = []
    for path in manifests:
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get(
                "status"
            )
        except Exception:
            status = None
        if status not in {"completed", "duplicate_skipped"}:
            incomplete.append(path.parent)
    return "run", (incomplete or [manifests[-1].parent])[0]


def find_incomplete_transaction(
    repository_root: Path, run_id_or_path: str
) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        temporary = (
            repository_root.resolve() / "workspace" / "temporary_runs"
        )
        if not resolved.is_relative_to(temporary):
            raise RuntimeError(
                "--resume accepts only workspace/temporary_runs transactions"
            )
        return resolved
    matches = [
        entry.path
        for entry in incomplete_runs(repository_root)
        if entry.run_id == run_id_or_path
        or entry.path.name == run_id_or_path
    ]
    if not matches:
        raise FileNotFoundError(
            f"Incomplete queue not found: {run_id_or_path}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Incomplete queue ID is ambiguous: {run_id_or_path}"
        )
    return matches[0]


def _remove_failed_queue_jobs(transaction: Path, console: Console) -> None:
    queue_path = transaction / "requested_queue.yaml"
    state_path = transaction / "queue_state.json"
    if not queue_path.is_file() or not state_path.is_file():
        raise RuntimeError(
            "This migrated legacy run has no editable queue manifest"
        )
    queue_payload = yaml.safe_load(
        queue_path.read_text(encoding="utf-8")
    ) or {}
    state = _transaction_payload(transaction, "queue_state.json")
    entry_state = state.get("entries", {})
    removable_statuses = {
        "failed",
        "failed_preflight",
        "interrupted",
        "waiting_for_selection",
        "skipped_dependency",
    }
    candidates = [
        item
        for item in queue_payload.get("entries", [])
        if str(entry_state.get(item.get("id"), {}).get("status"))
        in removable_statuses
    ]
    if not candidates:
        console.print("No failed or blocked queue jobs can be removed.")
        return
    table = Table(title="Failed/blocked jobs")
    table.add_column("#", justify="right")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Config", overflow="fold")
    for index, item in enumerate(candidates, 1):
        row = entry_state.get(item["id"], {})
        table.add_row(
            str(index),
            str(item["id"]),
            str(row.get("status", "unknown")),
            str(item.get("config", "")),
        )
    console.print(table)
    raw = typer.prompt(
        "Job number(s), comma-separated, all, or 0 = back",
        default="0",
    ).strip().lower()
    if raw == "0":
        return
    if raw == "all":
        selected = list(range(1, len(candidates) + 1))
    else:
        try:
            selected = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in raw.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(
                "Use comma-separated job numbers or 'all'"
            ) from exc
    if (
        not selected
        or min(selected) < 1
        or max(selected) > len(candidates)
    ):
        raise typer.BadParameter("Invalid failed job number")
    removed_ids = {candidates[index - 1]["id"] for index in selected}
    remaining = [
        item
        for item in queue_payload.get("entries", [])
        if item.get("id") not in removed_ids
    ]
    if not remaining:
        raise RuntimeError(
            "Removing every job would leave an empty queue; delete the "
            "temporary queue instead"
        )
    blocked = [
        str(item.get("id"))
        for item in remaining
        if removed_ids.intersection(item.get("depends_on", []))
    ]
    if blocked:
        raise RuntimeError(
            "Also remove dependent jobs before their prerequisites: "
            + ", ".join(blocked)
        )
    if not typer.confirm(
        "Remove these jobs from the queue? Their temporary job outputs are "
        "deleted; shared capture and observations remain.",
        default=False,
    ):
        return
    queue_payload["entries"] = remaining
    temporary = queue_path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(
            queue_payload, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    temporary.replace(queue_path)
    for job_id in removed_ids:
        entry_state.pop(job_id, None)
        state.get("source_fingerprints", {}).pop(job_id, None)
        state.get("resolved_configs", {}).pop(job_id, None)
        shutil.rmtree(transaction / "jobs" / job_id, ignore_errors=True)
        for path in (transaction / "resolved").glob(f"*_{job_id}_resolved.yaml"):
            path.unlink(missing_ok=True)
    state["updated_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    temporary_state = state_path.with_name(state_path.name + ".tmp")
    temporary_state.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    temporary_state.replace(state_path)
    console.print(
        "Removed queue job(s): " + ", ".join(sorted(removed_ids))
    )


def manage_incomplete_runs(
    repository_root: Path, console: Console
) -> Path | None:
    entries = incomplete_runs(repository_root)
    if not entries:
        console.print("No incomplete runs exist.")
        return None
    table = Table(title="Incomplete runs")
    table.add_column("#", justify="right")
    table.add_column("Dataset")
    table.add_column("Run")
    table.add_column("Status")
    table.add_column("Current/next stage")
    for index, entry in enumerate(entries, 1):
        table.add_row(
            str(index),
            entry.dataset_id,
            entry.run_id,
            entry.status,
            _active_run_stage(entry.path),
        )
    console.print(table)
    raw_selection = typer.prompt(
        "Run number(s), comma-separated, all, or 0 = back",
        default="1",
    ).strip().lower()
    if raw_selection == "0":
        return None
    if raw_selection == "all":
        selected_numbers = list(range(1, len(entries) + 1))
    else:
        try:
            selected_numbers = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in raw_selection.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(
                "Use one number, comma-separated numbers, or 'all'"
            ) from exc
    if (
        not selected_numbers
        or min(selected_numbers) < 1
        or max(selected_numbers) > len(entries)
    ):
        raise typer.BadParameter("Invalid incomplete run number")
    selected_entries = [entries[number - 1] for number in selected_numbers]
    if len(selected_entries) > 1:
        targets = "\n".join(f"- {entry.path}" for entry in selected_entries)
        console.print(
            Panel(
                f"{targets}\n\nOnly incomplete run folders will be deleted. "
                "Shared/content-addressed inputs are protected.",
                title="Bulk-delete incomplete runs",
            )
        )
        if not typer.confirm(
            f"Delete these {len(selected_entries)} incomplete runs?",
            default=False,
        ):
            console.print("Deletion cancelled.")
            return None
        removed: list[Path] = []
        for selected_entry in selected_entries:
            if _run_process_is_active(selected_entry.path):
                _interrupt_incomplete_run(selected_entry.path)
            removed.extend(
                _delete_incomplete_run(
                    repository_root,
                    selected_entry,
                    delete_private_inputs=False,
                )
            )
        console.print(
            "Deleted incomplete run folders:\n"
            + "\n".join(f"- {path}" for path in removed)
        )
        return None
    entry = selected_entries[0]
    action = _choice(
        "Incomplete run action",
        {
            "1": "resume; completed stages are skipped",
            "2": "stop/abort the active run but keep all files for a later resume",
            "3": "remove failed/blocked jobs from this queue",
            "4": "delete this complete temporary queue including capture and work data",
            "0": "back to main menu",
        },
        "1",
    )
    if action == "0":
        return None
    if action == "1":
        if _run_process_is_active(entry.path):
            raise RuntimeError(
                "This run is already active in another terminal. Do not start it twice."
            )
        return entry.path
    if action == "2":
        _interrupt_incomplete_run(entry.path)
        console.print(
            "Run stopped and marked interrupted. Its files remain resumable."
        )
        return None
    if action == "3":
        _remove_failed_queue_jobs(entry.path, console)
        return None
    console.print(
        Panel(
            f"Temporary queue: {entry.path}\n"
            "This removes its capture, prepared frames, observations, job "
            "outputs and logs. Published results and data_local are untouched.\n"
            "This deletion cannot be undone.",
            title="Confirm deletion",
        )
    )
    if not typer.confirm("Delete the selected incomplete run?", default=False):
        console.print("Deletion cancelled.")
        return None
    if _run_process_is_active(entry.path):
        _interrupt_incomplete_run(entry.path)
    removed = _delete_incomplete_run(
        repository_root, entry, delete_private_inputs=True
    )
    console.print("Deleted:\n" + "\n".join(f"- {path}" for path in removed))
    return None
