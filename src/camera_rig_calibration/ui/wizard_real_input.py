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



from .wizard_media import (
    _detected_static_camera_groups,
    _direct_static_cameras,
    _moving_source,
)
from .wizard_bindings import current_wizard_bindings
from .wizard_prompts import (
    _select_detected_path,
)

def _data_local_input_root(repository_root: Path) -> Path:
    landing = (repository_root / "data_local").resolve()
    if not landing.is_dir():
        raise RuntimeError(
            f"No local acquisition was discovered below {landing}. Add the "
            "recording, videos or frame folders there and restart the wizard."
        )
    return landing


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
    topics = current_wizard_bindings().list_mcap_topics(mcap)
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
        if not moving_info:
            raise RuntimeError(
                f"No unambiguous CameraInfo topic was discovered for {moving_id}. "
                "Add CameraInfo to the recording or prepare a canonical dataset "
                "before starting the wizard."
            )
        moving = MovingCameraSettings(
            id=moving_id,
            image_topic=moving_topic.name,
            camera_info_topic=moving_info,
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
        if not info_topic:
            raise RuntimeError(
                f"No unambiguous CameraInfo topic was discovered for {camera_id}. "
                "Add CameraInfo to the recording or prepare a canonical dataset "
                "before starting the wizard."
            )
        cameras.append(
            StaticCameraSettings(
                id=camera_id,
                image_topic=topic.name,
                camera_info_topic=info_topic,
            )
        )
    return cameras, moving


def _real_data_input(
    repository_root: Path,
    console: Console,
) -> tuple[
    Path,
    list[StaticCameraSettings],
    MovingCameraSettings,
    SamplingSettings,
    McapSettings,
    str,
]:
    input_root = _data_local_input_root(repository_root)
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
        proposal = "moving video/frames; no static camera pairs detected"
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


__all__ = [
    '_data_local_input_root',
    '_ros_image_stream_prefix',
    '_related_camera_info_topics',
    '_camera_id_from_ros_topic',
    '_mcap_camera_sources',
    '_real_data_input',
]
