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
    _stored_prepared_marker_settings,
    _stored_prepared_sampling,
)
from .wizard_media import (
    _prepared_input,
    _show_input_inventory,
    _show_prepared_choices,
)
from .wizard_method_jobs import (
    _job_methods,
    _job_selection,
)
from .wizard_method_queue import (
    _method_queue,
)
from .wizard_models import (
    MethodQueueJob,
    QueuedRun,
    SelectionDatasetContext,
    SimulationQueueJob,
    WizardBack,
    WizardOutcome,
    _method_job_label,
)
from .wizard_prepared import (
    _prepared_moving_intrinsics,
)
from .wizard_prompts import (
    _checkerboard_sources,
    _choice,
    _prompt_intrinsic_scan_settings,
    _select_checkerboard_source,
    _show_video_geometry,
)
from .wizard_real_input import (
    _real_data_input,
)
from .wizard_simulation import (
    _simulation_input_queue,
)
from .wizard_bindings import current_wizard_bindings

def _base_project(
    repository_root: Path,
    run_label: str = "baseline",
    execution_mode: str = "complete",
) -> ProjectSettings:
    return ProjectSettings(
        workspace_root=repository_root / "workspace",
        dataset_cache_root=repository_root / "workspace" / "preparation_cache",
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
    if video is not None:
        _show_video_geometry(console, "Intrinsic video geometry", video)
    profile_id = typer.prompt(
        "New intrinsics profile ID", default=safe_id(source.stem)
    ).strip()
    columns = 8
    rows = 6
    maximum_views = 80
    minimum_gap = 0 if images is not None else 5
    minimum_detections = 20
    scan = _prompt_intrinsic_scan_settings()
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
        target_hz = 3.0
        preview = 1920
        if scan.mode == "balanced":
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
            mode=scan.mode,
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
            / "src/camera_rig_calibration/input/intrinsics_calibration.py"
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
        for directory in ("workspace", "results")
    )
    if not exists:
        return suggested
    return f"{suggested}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _aruco_experiment_id(dataset_id: str, detection_mode: str) -> str:
    base_id = dataset_id
    for mode in ("baseline", "subpixel_refined", "high_sensitivity"):
        suffix = f"__aruco_{mode}"
        if base_id.endswith(suffix):
            base_id = base_id[: -len(suffix)]
            break
    if detection_mode == "baseline":
        return safe_id(base_id)
    suffix = f"__aruco_{detection_mode}"
    return safe_id(base_id + suffix)


def _rekey_method_contexts(
    jobs: list[MethodQueueJob],
    old_key: str,
    new_key: str,
) -> None:
    if old_key == new_key:
        return
    for job in jobs:
        for mapping in (
            job.context_methods,
            job.context_selections,
            job.context_deferred_selection_keys,
        ):
            if old_key in mapping:
                mapping[new_key] = mapping.pop(old_key)


def _build_simulation_batch_outcome(
    repository_root: Path,
    console: Console,
    experiment_jobs: list[SimulationQueueJob],
) -> WizardOutcome:
    hooks = current_wizard_bindings()
    _choice = hooks.choice
    _job_methods = hooks.job_methods
    _job_selection = hooks.job_selection
    _method_job_label = hooks.method_job_label
    _method_queue = hooks.method_queue
    _save_wizard_queue = hooks.save_wizard_queue
    action = _choice(
        "After simulation input selection",
        {
            "1": "run the complete calibration pipeline for every experiment",
            "2": "prepare and validate every input only (no AP methods)",
        },
        "1",
    )
    execution_mode = "prepare_only" if action == "2" else "complete"
    if execution_mode == "prepare_only":
        console.print(
            "Prepare-only mode creates each canonical input exactly once and "
            "does not schedule AP preflight or calibration methods."
        )
        method_jobs: list[MethodQueueJob] = []
    else:
        method_jobs = _method_queue(
            console,
            tuple(
                SelectionDatasetContext(
                    key=experiment.experiment_id,
                    display_name=experiment.experiment_id,
                    dataset_root=experiment.prepared_root,
                    static_cameras=experiment.cameras,
                )
                for experiment in experiment_jobs
            ),
        )
    default_batch_id = (
        f"simulation_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    batch_id = safe_id(
        typer.prompt("Simulation batch ID", default=default_batch_id).strip()
    )
    batch_root = repository_root / "workspace" / "batches" / batch_id
    queued_runs: list[QueuedRun] = []
    queues: list[tuple[str, Path]] = []
    used_dataset_ids: set[str] = set()
    for experiment_index, experiment in enumerate(experiment_jobs, 1):
        dataset_id = safe_id(experiment.experiment_id)
        if method_jobs:
            dataset_id = _aruco_experiment_id(
                dataset_id, method_jobs[0].markers.detection_mode
            )
        suffix = 2
        while dataset_id in used_dataset_ids:
            dataset_id = safe_id(f"{experiment.experiment_id}_{suffix}")
            suffix += 1
        used_dataset_ids.add(dataset_id)
        dataset = DatasetSettings(
            id=dataset_id,
            category=DatasetCategory.SIMULATION,
            source_kind=InputSourceKind.PREPARED,
            scene_type=SceneType.SIMULATION,
            prepared_root=experiment.prepared_root,
            input_root=(
                experiment.prepared_root
                if experiment.prepared_root is not None
                else (
                    experiment.simulation.world.parent
                    if experiment.simulation.world is not None
                    else None
                )
            ),
        )
        common = {
            "dataset": dataset,
            "static_cameras": list(experiment.cameras),
            "moving_camera": experiment.moving_camera,
            "simulation": experiment.simulation,
            "sampling": SamplingSettings(target_hz=None),
        }
        experiment_queue_root = (
            batch_root
            / f"{experiment_index:02d}_{dataset_id}"
            / "queue"
        )
        experiment_runs: list[QueuedRun] = []
        if execution_mode == "prepare_only":
            project = _base_project(
                repository_root,
                run_label="prepare_input",
                execution_mode="prepare_only",
            ).model_copy(update={"experiment_id": dataset_id})
            config = RigConfig(
                project=project,
                methods=MethodSettings(),
                evaluation=EvaluationSettings(enabled=False),
                **common,
            )
            path = experiment_queue_root / "01_prepare_input.yaml"
            save_user_config(config, path)
            experiment_runs.append(QueuedRun(config, path))
        else:
            for method_index, job in enumerate(method_jobs, 1):
                methods = _job_methods(job, experiment.experiment_id)
                selection = _job_selection(job, experiment.experiment_id)
                label = _method_job_label(
                    job, experiment.experiment_id
                )
                project = _base_project(
                    repository_root,
                    run_label=label,
                ).model_copy(update={"experiment_id": dataset_id})
                config = RigConfig(
                    project=project,
                    markers=job.markers,
                    observation_quality=job.observation_quality,
                    colmap=job.colmap,
                    methods=methods,
                    evaluation=job.evaluation,
                    selection=selection,
                    **common,
                )
                path = (
                    experiment_queue_root
                    / f"{method_index:02d}_{label}.yaml"
                )
                save_user_config(config, path)
                experiment_runs.append(QueuedRun(config, path))
        queue_path = _save_wizard_queue(
            experiment_queue_root,
            f"{batch_id}__{dataset_id}",
            experiment_runs,
        )
        queues.append((dataset_id, queue_path))
        queued_runs.extend(experiment_runs)
    batch_path = save_batch(
        batch_id,
        queues,
        batch_root / "batch.yaml",
    )
    first, *rest = queued_runs
    return WizardOutcome(
        first.config,
        first.path,
        tuple(rest),
        batch_path=batch_path,
        queue_paths=tuple(path for _, path in queues),
    )


def new_calibration_wizard(
    repository_root: Path, console: Console
) -> WizardOutcome | None:
    hooks = current_wizard_bindings()
    _build_simulation_batch_outcome = (
        hooks.build_simulation_batch_outcome
    )
    _choice = hooks.choice
    _job_methods = hooks.job_methods
    _job_selection = hooks.job_selection
    _method_job_label = hooks.method_job_label
    _method_queue = hooks.method_queue
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
                "0": "back to main menu",
            },
            "1",
        )
        if mode == "0":
            return None
        prepared_root = None
        prepared_markers = None
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
                    ) = _real_data_input(repository_root, console)
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
                    prepared_markers = _stored_prepared_marker_settings(
                        prepared_root
                    )
                    input_root = prepared_root
                    source_kind = InputSourceKind.PREPARED
            elif mode == "2":
                experiment_jobs = _simulation_input_queue(
                    repository_root, console
                )
                return _build_simulation_batch_outcome(
                    repository_root,
                    console,
                    experiment_jobs,
                )
            break
        except WizardBack:
            continue

    default_dataset_id = (
        suggested_id
        if scene_type is SceneType.SIMULATION or prepared_root is not None
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
            selection_contexts = (
                (
                    SelectionDatasetContext(
                        key=dataset_id,
                        display_name=dataset_id,
                        dataset_root=prepared_root,
                        static_cameras=tuple(cameras),
                    ),
                )
                if prepared_root is not None
                else ()
            )
            jobs = _method_queue(
                console,
                selection_contexts,
                initial_markers=prepared_markers,
            )
            original_dataset_id = dataset_id
            dataset_id = _aruco_experiment_id(
                dataset_id, jobs[0].markers.detection_mode
            )
            _rekey_method_contexts(
                jobs, original_dataset_id, dataset_id
            )
            if dataset_id != original_dataset_id:
                console.print(
                    "Non-baseline ArUco observations use the distinct "
                    f"experiment ID: [cyan]{dataset_id}[/cyan]"
                )
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
        save_user_config(config, path)
        return WizardOutcome(config, path)

    queued: list[QueuedRun] = []
    queue_root = repository_root / "workspace" / dataset_id / "queue"
    for index, job in enumerate(jobs, 1):
        methods = _job_methods(job, dataset_id)
        selection = _job_selection(job, dataset_id)
        label = _method_job_label(job, dataset_id)
        config = RigConfig(
            project=_base_project(repository_root, run_label=label),
            markers=job.markers,
            observation_quality=job.observation_quality,
            colmap=job.colmap,
            methods=methods,
            evaluation=job.evaluation,
            selection=selection,
            **common,
        )
        path = queue_root / f"{index:02d}_{label}.yaml"
        save_user_config(config, path)
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
                "id": (
                    f"{queued_run.config.dataset.id}__"
                    f"{queued_run.config.methods.enabled[0]}__"
                    f"{queued_run.config.project.run_label}__{index:02d}"
                ),
                "config": queued_run.path.name,
                "depends_on": [],
            }
            for index, queued_run in enumerate(queued, 1)
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
                    f"{run.config.methods.enabled[0]}__"
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


__all__ = [
    '_base_project',
    '_create_intrinsic_profile_only',
    '_new_dataset_id',
    '_aruco_experiment_id',
    '_rekey_method_contexts',
    '_build_simulation_batch_outcome',
    'new_calibration_wizard',
    '_save_wizard_queue',
]
