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



from .wizard_models import (
    WizardBack,
)
from .wizard_bindings import current_wizard_bindings

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
        raise RuntimeError(
            f"No {label.lower()} was discovered. Place it below data_local "
            "or in a canonical dataset and restart the wizard."
        )
    typer.echo(f"\n{label}:")
    for index, path in enumerate(paths, 1):
        typer.echo(f"  {index}. {path}")
    selected = _prompt_index(
        "Selection (0/b = back)",
        default=preferred + 1,
        maximum=len(paths),
    )
    if selected is None:
        raise WizardBack()
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
        raise RuntimeError(
            "No checkerboard video or image folder was discovered below "
            f"{input_root}. Add it below data_local and restart the wizard."
        )
    if len(sources) == 1:
        kind, path, description = sources[0]
        typer.echo(f"\nCheckerboard calibration input: {description} (automatic)")
        return (path, None) if kind == "video" else (None, path)
    typer.echo("\nCheckerboard calibration input:")
    for index, (_, _, description) in enumerate(sources, 1):
        typer.echo(f"  {index}. {description}")
    selected = _prompt_index(
        "Selection (0/b = back)",
        default=1,
        maximum=len(sources),
    )
    if selected is None:
        raise WizardBack()
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
        try:
            geometry = probe_video_geometry(video)
        except RuntimeError:
            return None
        return geometry.output_width, geometry.output_height
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


def _show_video_geometry(
    console: Console,
    label: str,
    video: Path,
) -> None:
    geometry = current_wizard_bindings().probe_video_geometry(video)
    console.print(
        f"{label}: encoded "
        f"{geometry.encoded_width}x{geometry.encoded_height}, "
        f"rotation {geometry.display_rotation_degrees:+d} deg, "
        f"normalized {geometry.output_width}x{geometry.output_height} "
        f"({geometry.orientation_policy})."
    )


def _prompt_intrinsic_scan_settings() -> IntrinsicScanSettings:
    mode = _prompt_enum_choice(
        "Intrinsic analysis mode",
        "balanced",
        (
            (
                "balanced",
                "recommended; adaptive 3/6/12 Hz search and full-resolution "
                "corner refinement",
            ),
            (
                "full_frame",
                "every original frame at full resolution; much slower with "
                "maximum search coverage",
            ),
        ),
    )
    return IntrinsicScanSettings(mode=mode)

def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes", "true", "1", "on"}:
        return True
    if normalized in {"n", "no", "false", "0", "off"}:
        return False
    raise ValueError("enter yes or no")


def _format_setting_value(value: object) -> str:
    """Format wizard values without scientific notation or Python booleans."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    return str(value)


_PUBLIC_POLICY_NAMES = {
    "legacy_main_v1": "baseline Direct/Relay",
    "wizard_robustness_v1": "robust consensus",
    "legacy_smart_v1": "smart frame budgets",
    "wizard_graph_preserving_v1": "graph-preserving preselection",
    "legacy_maximum_bottleneck_v1": "maximum-bottleneck tree",
    "wizard_maximum_bottleneck_v2": "path-aware maximum-bottleneck tree",
    "legacy_observation_quality_v1": "geometric observation quality",
    "wizard_selection_score_v2": "shared selection score",
    "legacy_pinhole_v1": "pinhole",
    "distortion_aware_v1": "distortion-aware",
    "legacy_colmap_defaults_v1": "COLMAP defaults",
    "wizard_explicit_limits_v1": "explicit feature limits",
    "legacy_registered_image_redetection_v1": "registered-image detection",
    "wizard_filtered_observations_v1": "filtered registered-image detection",
}


def _public_policy_name(value: object) -> object:
    """Return a stable product label without exposing compatibility names."""

    return _PUBLIC_POLICY_NAMES.get(str(value), value)


def _optional_positive_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "unlimited", "disabled"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("enter a positive integer or null")
    return parsed


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
        typer.echo(
            f"  {index}. {_public_policy_name(value)}{suffix} — {meaning}"
        )
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
            f"Choose 1-{len(values)} or enter a displayed option name."
        )


__all__ = [
    '_clear_terminal',
    '_show_input_error',
    '_prompt_index',
    '_simulation_experiment_id',
    '_choice',
    '_select_detected_path',
    '_preferred_path',
    '_looks_like_checkerboard_video',
    '_checkerboard_sources',
    '_select_checkerboard_source',
    '_moving_media_dimensions',
    '_show_video_geometry',
    '_prompt_intrinsic_scan_settings',
    '_bool_value',
    '_format_setting_value',
    '_PUBLIC_POLICY_NAMES',
    '_public_policy_name',
    '_optional_positive_int',
    '_prompt_enum_choice',
]
