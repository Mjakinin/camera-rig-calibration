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
from .wizard_prompts import (
    _clear_terminal,
    _show_input_error,
)

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
        "route": "Moving-camera route discovered from built-ins or data_local/simulation_routes.",
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
            if key in {"route_file", "route_sampling_strategy"}:
                typer.echo(
                    "This value is derived; edit the route name or frame count."
                )
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
                            f"Route ({route_names})",
                            current,
                        )
                        if value not in routes:
                            raise ValueError(
                                "unknown route; put custom JSON below "
                                "data_local/simulation_routes and restart the wizard"
                            )
                        route = routes[value]
                        parameters["route"] = value
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


__all__ = [
    '_lighting_profiles',
    '_show_lighting_profiles',
    '_edit_simulation_parameters',
]
