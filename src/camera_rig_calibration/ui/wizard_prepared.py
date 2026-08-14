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



from .wizard_prompts import (
    _prompt_intrinsic_scan_settings,
    _select_checkerboard_source,
    _show_video_geometry,
)

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
    if not options:
        raise RuntimeError(
            "No compatible moving-camera intrinsics were discovered for the "
            "prepared frames. Add a managed profile or checkerboard input and "
            "restart the wizard."
        )
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
    updates: dict[str, object] = {
        "intrinsic_calibration_video": calibration_video,
        "intrinsic_calibration_images": calibration_images,
        "intrinsics_profile": profile_id,
        "intrinsic_minimum_frame_gap": (
            0 if calibration_images is not None else 5
        ),
    }
    updates["intrinsic_scan"] = _prompt_intrinsic_scan_settings()
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
        scan = updates["intrinsic_scan"]
        assert isinstance(scan, IntrinsicScanSettings)
        target_hz = 3.0
        preview_dimension = 1920
        if scan.mode == "balanced":
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
            mode=scan.mode,
            target_hz=target_hz,
            preview_max_dimension=preview_dimension,
        )
    else:
        console.print(
            "Checkerboard settings: balanced 3/6/12 Hz scan, 1920 px "
            "preview, full-resolution refinement, 8x6 board, max 80 views."
        )
    return moving.model_copy(update=updates, deep=True)


__all__ = [
    '_prepared_moving_intrinsics',
]
