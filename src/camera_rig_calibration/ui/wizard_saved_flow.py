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



from .wizard_method_queue import (
    _edit_method_job,
)
from .wizard_models import (
    MethodQueueJob,
    QueuedRun,
    WizardOutcome,
)
from .wizard_new_flow import (
    _save_wizard_queue,
)
from .wizard_prompts import (
    _choice,
)
from .wizard_bindings import current_wizard_bindings

def _load_saved_setup_config(path: Path) -> RigConfig:
    """Load a setup and rebind published provenance to its experiment input."""

    config = load_config(path)
    is_published_provenance = (
        path.name == "resolved_config.yaml"
        and len(path.parents) >= 5
        and path.parents[3].name == "methods"
        and (path.parents[4] / "dataset.json").is_file()
    )
    if not is_published_provenance:
        return config
    experiment = path.parents[4].resolve()
    moving_intrinsics = (
        experiment
        / "raw_images"
        / "camera_info"
        / f"{config.moving_camera.id}.json"
    )
    return config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "source_kind": InputSourceKind.PREPARED,
                    "prepared_root": experiment,
                    "input_root": experiment,
                },
                deep=True,
            ),
            "moving_camera": config.moving_camera.model_copy(
                update={
                    "intrinsics": (
                        moving_intrinsics
                        if moving_intrinsics.is_file()
                        else config.moving_camera.intrinsics
                    ),
                    "video": None,
                    "frames": None,
                },
                deep=True,
            ),
        },
        deep=True,
    )


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
            config = _load_saved_setup_config(path)
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
    return _load_saved_setup_config(paths[index - 1]), paths[index - 1]


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
            loaded.append(_load_saved_setup_config(candidate))
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
            save_user_config(queued_config, destination)
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
    save_user_config(config, destination)
    console.print(f"Template: {source}")
    return WizardOutcome(config, destination)


def advanced_wizard(repository_root: Path, console: Console) -> WizardOutcome | None:
    hooks = current_wizard_bindings()
    _choice = hooks.choice
    _edit_method_job = hooks.edit_method_job
    _save_wizard_queue = hooks.save_wizard_queue
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
        loaded = [_load_saved_setup_config(path) for path in candidates]
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
                save_user_config(config, destination)
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
            save_user_config(variant, destination)
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
    save_user_config(config, destination)
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
        ", ".join(
            (
                f"{method_id.upper()} "
                f"{getattr(config.methods, method_id).method_contract}"
                if method_id in {"ap01", "ap02", "ap03"}
                else method_id
            )
            for method_id in config.methods.enabled
        ),
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
    selections: list[str] = []
    if "ap01" in config.methods.enabled:
        selections.append(f"AP01 root={config.methods.ap01.root_camera}")
    if "ap02" in config.methods.enabled:
        selections.append(
            f"AP02 ref={config.methods.ap02.reference_marker_id}"
        )
    if "ap03" in config.methods.enabled:
        selections.extend(
            [
                f"AP03 single={config.methods.ap03.single.scale_marker_id}",
                f"AP03 multi={config.methods.ap03.multi.marker_ids}",
            ]
        )
    selections.append(f"evaluation={config.evaluation.anchor_marker_id}")
    table.add_row("Scientific selections", "; ".join(selections))
    console.print(table)


__all__ = [
    '_load_saved_setup_config',
    '_config_candidates',
    'saved_setup_count',
    'choose_config',
    'repeat_setup_wizard',
    'advanced_wizard',
    'show_summary',
]
