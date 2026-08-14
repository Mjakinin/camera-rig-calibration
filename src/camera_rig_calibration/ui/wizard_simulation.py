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
    SimulationQueueJob,
    WizardBack,
    _bus_definition,
)
from .wizard_prompts import (
    _choice,
    _prompt_index,
    _show_input_error,
    _simulation_experiment_id,
)
from .wizard_simulation_parameters import (
    _edit_simulation_parameters,
)
from .wizard_bindings import current_wizard_bindings

def _simulation_input(
    repository_root: Path, console: Console
) -> tuple[
    list[StaticCameraSettings],
    MovingCameraSettings,
    SimulationSettings,
    str,
    Path | None,
]:
    bus_world = _bus_definition(repository_root)
    experiments = current_wizard_bindings().discover_simulation_experiments(
        repository_root
    )
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


def _simulation_job_from_parameters(
    repository_root: Path,
    parameters: dict[str, object],
    *,
    experiment_id: str,
    prepared_root: Path | None,
    source: str,
) -> SimulationQueueJob:
    bus = _bus_definition(repository_root)
    normalized: dict[str, object] = {
        **BASELINE_SIMULATION_PARAMETERS,
        **parameters,
    }
    route_name = str(normalized["route"])
    route = next(
        (candidate.path for candidate in bus.routes if candidate.id == route_name),
        None,
    )
    if route is None:
        raise ValueError(
            f"Unsupported bus route '{route_name}'; choose route1 or route2"
        )
    world = bus.lighting_profiles.get(str(normalized["lighting"]))
    if world is None:
        world = bus.sdf
    cameras = tuple(
        StaticCameraSettings(
            id=camera.id,
            image_topic=camera.image_topic,
            camera_info_topic=camera.camera_info_topic,
        )
        for camera in bus.static_cameras
    )
    moving = MovingCameraSettings(
        id=bus.moving_camera.id,
        image_topic=bus.moving_camera.image_topic,
        camera_info_topic=bus.moving_camera.camera_info_topic,
    )
    capture_id = (
        None
        if prepared_root is not None
        else datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    )
    simulation = SimulationSettings(
        enabled=prepared_root is None,
        preset=(
            f"existing_{safe_id(experiment_id)}"
            if prepared_root is not None
            else "bus_batch_capture"
        ),
        world_id="bus",
        world_baseline=dict(BASELINE_SIMULATION_PARAMETERS),
        capture_id=capture_id,
        world=world,
        route=route,
        resource_paths=list(bus.resource_paths),
        moving_model_name=bus.moving_camera.model_name,
        moving_sensor_name=bus.moving_camera.sensor_name,
        settle_seconds=float(normalized["settle_seconds"]),
        post_pose_skip=int(normalized["post_pose_skip"]),
        frame_timeout_seconds=float(normalized["frame_timeout_seconds"]),
        startup_timeout_seconds=float(
            normalized["startup_timeout_seconds"]
        ),
        route_name=route_name,
        moving_width=int(normalized["moving_width"]),
        moving_height=int(normalized["moving_height"]),
        moving_hfov_deg=float(normalized["moving_hfov_deg"]),
        lighting=str(normalized["lighting"]),
        lighting_scale=float(normalized["lighting_scale"]),
        motion_blur_kernel=int(normalized["motion_blur_kernel"]),
        motion_blur_angle_deg=float(
            normalized["motion_blur_angle_deg"]
        ),
        target_route_frames=int(normalized["target_route_frames"]),
        route_sampling_strategy=str(
            normalized["route_sampling_strategy"]
        ),
    )
    return SimulationQueueJob(
        experiment_id=safe_id(experiment_id),
        parameters=normalized,
        cameras=cameras,
        moving_camera=moving,
        simulation=simulation,
        prepared_root=(
            prepared_root.resolve() if prepared_root is not None else None
        ),
        source=source,
    )


def _simulation_signature(parameters: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _show_simulation_input_queue(
    console: Console, jobs: list[SimulationQueueJob]
) -> None:
    if not jobs:
        console.print("[dim]Simulation experiment queue is empty.[/dim]")
        return
    table = Table(title="Simulation experiment queue")
    table.add_column("#", justify="right")
    table.add_column("Experiment")
    table.add_column("Input")
    table.add_column("Source")
    table.add_column("Complete parameters", overflow="fold")
    for index, job in enumerate(jobs, 1):
        table.add_row(
            str(index),
            job.experiment_id,
            job.input_mode,
            job.source,
            format_simulation_parameters(job.parameters),
        )
    console.print(table)


def _parse_experiment_numbers(
    raw: str, maximum: int
) -> list[int]:
    normalized = raw.strip().lower()
    if normalized == "all":
        return list(range(1, maximum + 1))
    numbers = list(
        dict.fromkeys(
            int(value.strip())
            for value in normalized.split(",")
            if value.strip()
        )
    )
    if not numbers or min(numbers) < 1 or max(numbers) > maximum:
        raise ValueError(f"choose numbers between 1 and {maximum}, or all")
    return numbers


def _simulation_input_queue(
    repository_root: Path, console: Console
) -> list[SimulationQueueJob]:
    experiments = current_wizard_bindings().discover_simulation_experiments(
        repository_root
    )
    planned: list[SimulationQueueJob] = []
    if experiments:
        table = Table(title="Existing bus simulation experiments")
        table.add_column("#", justify="right")
        table.add_column("Experiment")
        table.add_column("Input")
        table.add_column("Result")
        table.add_column("Complete parameters", overflow="fold")
        for index, experiment in enumerate(experiments, 1):
            table.add_row(
                str(index),
                experiment.variant,
                (
                    "reuse local"
                    if experiment.dataset_root is not None
                    else "new capture required"
                ),
                "available" if experiment.has_results else "input only",
                format_simulation_parameters(experiment.parameters),
            )
        console.print(table)

    def append(job: SimulationQueueJob) -> None:
        signature = _simulation_signature(job.parameters)
        if any(
            _simulation_signature(candidate.parameters) == signature
            for candidate in planned
        ):
            console.print(
                f"[yellow]{job.experiment_id} has the same complete parameter "
                "vector as an experiment already in the queue; it was not "
                "added twice.[/yellow]"
            )
            return
        used_ids = {candidate.experiment_id for candidate in planned}
        candidate_id = job.experiment_id
        suffix = 2
        while candidate_id in used_ids:
            candidate_id = safe_id(f"{job.experiment_id}_{suffix}")
            suffix += 1
        if candidate_id != job.experiment_id:
            job = SimulationQueueJob(
                experiment_id=candidate_id,
                parameters=job.parameters,
                cameras=job.cameras,
                moving_camera=job.moving_camera,
                simulation=job.simulation,
                prepared_root=job.prepared_root,
                source=job.source,
            )
        planned.append(job)

    while True:
        _show_simulation_input_queue(console, planned)
        action = _choice(
            "Bus simulation experiment queue",
            {
                "1": "add the Route-2 baseline",
                "2": "add existing experiments (comma-separated or all)",
                "3": "add a new mixed parameter combination",
                "4": "remove queued experiments",
                "5": "accept this experiment queue",
                "0": "back to input type",
            },
            "5" if planned else "1",
        )
        if action == "0":
            raise WizardBack()
        if action == "1":
            parameters = dict(BASELINE_SIMULATION_PARAMETERS)
            match = find_matching_simulation(experiments, parameters)
            append(
                _simulation_job_from_parameters(
                    repository_root,
                    parameters,
                    experiment_id="route2",
                    prepared_root=(
                        match.dataset_root if match is not None else None
                    ),
                    source=(
                        f"existing {match.variant}"
                        if match is not None
                        else "Route-2 baseline"
                    ),
                )
            )
        elif action == "2":
            if not experiments:
                _show_input_error(
                    "No historical bus simulation experiment is discoverable."
                )
                continue
            raw = typer.prompt(
                "Existing experiment number(s), comma-separated, all, or 0/b = back"
            ).strip()
            if raw.lower() in {"0", "b", "back"}:
                continue
            try:
                numbers = _parse_experiment_numbers(raw, len(experiments))
            except (TypeError, ValueError) as exc:
                _show_input_error(str(exc))
                continue
            for number in numbers:
                experiment = experiments[number - 1]
                append(
                    _simulation_job_from_parameters(
                        repository_root,
                        experiment.parameters,
                        experiment_id=experiment.variant,
                        prepared_root=experiment.dataset_root,
                        source=f"existing {experiment.variant}",
                    )
                )
        elif action == "3":
            bases: list[
                tuple[str, dict[str, object], Path | None]
            ] = [
                (
                    "Route-2 baseline",
                    dict(BASELINE_SIMULATION_PARAMETERS),
                    None,
                ),
                *[
                    (
                        f"existing {experiment.variant}",
                        dict(experiment.parameters),
                        experiment.dataset_root,
                    )
                    for experiment in experiments
                ],
                *[
                    (
                        f"queued {job.experiment_id}",
                        dict(job.parameters),
                        job.prepared_root,
                    )
                    for job in planned
                ],
            ]
            base_table = Table(title="Base for the new combination")
            base_table.add_column("#", justify="right")
            base_table.add_column("Base")
            base_table.add_column("Parameters", overflow="fold")
            for index, (label, parameters, _) in enumerate(bases, 1):
                base_table.add_row(
                    str(index),
                    label,
                    format_simulation_parameters(parameters),
                )
            console.print(base_table)
            selected = _prompt_index(
                "Base number (0/b = back)",
                default=1,
                maximum=len(bases),
            )
            if selected is None:
                continue
            label, parameters, _ = bases[selected - 1]
            route_name = str(parameters.get("route", "route2"))
            route = next(
                item.path
                for item in _bus_definition(repository_root).routes
                if item.id == route_name
            )
            edited, _, capture = _edit_simulation_parameters(
                repository_root,
                console,
                dict(parameters),
                route,
            )
            edited.update(capture)
            experiment_id = _simulation_experiment_id(edited)
            append(
                _simulation_job_from_parameters(
                    repository_root,
                    edited,
                    experiment_id=experiment_id,
                    prepared_root=None,
                    source=f"new combination from {label}",
                )
            )
        elif action == "4":
            if not planned:
                _show_input_error("The experiment queue is already empty.")
                continue
            raw = typer.prompt(
                "Queue row number(s), comma-separated, all, or 0/b = back"
            ).strip()
            if raw.lower() in {"0", "b", "back"}:
                continue
            try:
                numbers = _parse_experiment_numbers(raw, len(planned))
            except (TypeError, ValueError) as exc:
                _show_input_error(str(exc))
                continue
            removed = set(numbers)
            planned = [
                job
                for index, job in enumerate(planned, 1)
                if index not in removed
            ]
        elif planned:
            return planned
        else:
            _show_input_error("Add at least one simulation experiment.")


__all__ = [
    '_simulation_input',
    '_simulation_job_from_parameters',
    '_simulation_signature',
    '_show_simulation_input_queue',
    '_parse_experiment_numbers',
    '_simulation_input_queue',
]
