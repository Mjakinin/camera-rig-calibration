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



from .wizard_method_jobs import (
    GUIDED_SELECTION_KEYS,
    _configure_guided_selection,
    _job_methods,
    _new_method_job,
    _parse_ids,
    _refresh_job_selection_mode,
    _selection_value,
    _show_method_queue,
    _sync_context_methods,
    _validate_prepared_job_selections,
)
from .wizard_models import (
    MethodQueueJob,
    SelectionDatasetContext,
    WizardBack,
    _refresh_method_job_label,
)
from .wizard_prompts import (
    _choice,
    _clear_terminal,
    _format_setting_value,
    _optional_positive_int,
    _prompt_enum_choice,
    _prompt_index,
    _public_policy_name,
    _show_input_error,
)
from .wizard_bindings import current_wizard_bindings

METHOD_JOB_GROUPS = frozenset(
    {
        "OBSERVATION QUALITY OVERRIDE",
        "METHOD-SPECIFIC SETTINGS",
        "COLMAP SETTINGS",
    }
)


def _setting_rows(
    job: MethodQueueJob,
    groups: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, str, str, object, object, str]]:
    # Keep this public compatibility hook in the facade: product policies wrap
    # it, while the declarative row construction lives in a focused UI module.
    from .method_settings import build_setting_rows

    hooks = current_wizard_bindings()
    return build_setting_rows(
        job,
        groups,
        defaults_factory=hooks.new_method_job,
        selection_value=_selection_value,
    )



def _edit_method_job(
    console: Console,
    job: MethodQueueJob,
    *,
    groups: set[str] | frozenset[str] = METHOD_JOB_GROUPS,
    title: str | None = None,
    selection_contexts: tuple[SelectionDatasetContext, ...] = (),
) -> MethodQueueJob:
    from .method_editor import EditorBindings, edit_method_job

    hooks = current_wizard_bindings()
    return edit_method_job(
        console,
        job,
        groups=groups,
        title=title,
        selection_contexts=selection_contexts,
        bindings=EditorBindings(
            setting_rows=hooks.setting_rows,
            format_setting_value=_format_setting_value,
            public_policy_name=_public_policy_name,
            clear_terminal=_clear_terminal,
            show_input_error=_show_input_error,
            configure_guided_selection=hooks.configure_guided_selection,
            prompt_enum_choice=_prompt_enum_choice,
            job_methods=_job_methods,
            optional_positive_int=_optional_positive_int,
            parse_ids=_parse_ids,
            refresh_job_selection_mode=_refresh_job_selection_mode,
            refresh_method_job_label=hooks.refresh_method_job_label,
            sync_context_methods=_sync_context_methods,
            wizard_back=WizardBack,
            guided_selection_keys=GUIDED_SELECTION_KEYS,
        ),
    )


def _clone_method_job(job: MethodQueueJob, label: str) -> MethodQueueJob:
    del label
    clone = MethodQueueJob(
        method_id=job.method_id,
        label=job.label,
        methods=job.methods.model_copy(deep=True),
        markers=job.markers.model_copy(deep=True),
        observation_quality=job.observation_quality.model_copy(deep=True),
        colmap=job.colmap.model_copy(deep=True),
        evaluation=job.evaluation.model_copy(deep=True),
        selection=job.selection.model_copy(deep=True),
        context_methods={
            key: value.model_copy(deep=True)
            for key, value in job.context_methods.items()
        },
        context_selections={
            key: value.model_copy(deep=True)
            for key, value in job.context_selections.items()
        },
        deferred_selection_keys=set(job.deferred_selection_keys),
        context_deferred_selection_keys={
            key: set(value)
            for key, value in job.context_deferred_selection_keys.items()
        },
    )
    _refresh_method_job_label(clone)
    return clone


def _method_queue(
    console: Console,
    selection_contexts: tuple[SelectionDatasetContext, ...] = (),
    *,
    initial_markers: MarkerSettings | None = None,
) -> list[MethodQueueJob]:
    hooks = current_wizard_bindings()
    _choice = hooks.choice
    _edit_method_job = hooks.edit_method_job
    _new_method_job = hooks.new_method_job
    _prompt_index = hooks.prompt_index
    _refresh_method_job_label = hooks.refresh_method_job_label
    _show_input_error = hooks.show_input_error
    _show_method_queue = hooks.show_method_queue
    _validate_prepared_job_selections = (
        hooks.validate_prepared_job_selections
    )
    recommended_ids = ["ap01", "ap02", "ap03"]
    methods = sorted(
        calibration_methods.all(),
        key=lambda method: (
            (
                recommended_ids.index(method.id)
                if method.id in recommended_ids
                else len(recommended_ids)
            ),
            method.id,
        ),
    )
    explanations = {
        "ap01": "experimental baseline; marker-direct and moving-COLMAP relay",
        "ap02": "primary candidate; static-only diagnostic and combined bundle adjustment",
        "ap03": "primary candidate; one COLMAP reconstruction, single and multi scale",
    }

    def show_choices() -> None:
        for index, method in enumerate(methods, 1):
            typer.echo(
                f"  {index}. {method.id.upper()} — "
                f"{explanations.get(method.id, method.display_name)}"
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
            numbers = [
                int(value.strip())
                for value in raw.split(",")
                if value.strip()
            ]
        except ValueError:
            _show_input_error(
                "Use comma-separated method numbers, for example 1,2,3."
            )
            continue
        if not numbers or min(numbers) < 1 or max(numbers) > len(methods):
            typer.echo(f"Choose method numbers between 1 and {len(methods)}.")
            continue
        jobs = []
        method_counts: dict[str, int] = {}
        for number in numbers:
            method_id = methods[number - 1].id
            method_counts[method_id] = method_counts.get(method_id, 0) + 1
            job = _new_method_job(
                method_id,
                prompt_for_single_marker=True,
                markers=initial_markers,
            )
            jobs.append(job)
        while True:
            _show_method_queue(console, jobs)
            action = _choice(
                "Queue action",
                {
                    "1": "accept this queue and continue",
                    "2": "add another method job",
                    "3": "duplicate a job (best for ablations/parameter comparisons)",
                    "4": "edit one method job (quality, method and COLMAP settings)",
                    "5": "edit queue-wide ArUco input",
                    "6": "edit queue-wide common evaluation",
                    "7": "remove jobs (comma-separated or all)",
                    "8": "edit queue-wide observation-quality baseline",
                    "0": "back to method selection",
                },
                "1",
            )
            if action == "1":
                try:
                    _validate_prepared_job_selections(
                        jobs, selection_contexts
                    )
                except ValueError as exc:
                    _show_input_error(
                        f"{exc}. Re-open that method row and choose from the "
                        "updated candidates."
                    )
                    continue
                if any(
                    job.markers != jobs[0].markers
                    or job.evaluation != jobs[0].evaluation
                    or job.observation_quality
                    != jobs[0].observation_quality
                    for job in jobs[1:]
                ):
                    typer.echo(
                        "ArUco input, observation-quality baseline and common "
                        "evaluation belong to the queue. Use the queue-wide "
                        "actions to edit them."
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
                new_job.observation_quality = (
                    jobs[0].observation_quality.model_copy(deep=True)
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
                jobs.append(_clone_method_job(source, source.label))
                typer.echo(
                    "Copied configuration. Edit the new row to create a distinct "
                    "automatic result name; unchanged duplicates are skipped."
                )
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
                        console,
                        candidate,
                        selection_contexts=selection_contexts,
                    )
                except (
                    ValidationError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    yaml.YAMLError,
                ) as exc:
                    _show_input_error(
                        f"Invalid method setting: {exc}. "
                        "The previous row was kept unchanged."
                    )
            elif action == "5":
                candidate = _clone_method_job(jobs[0], jobs[0].label)
                try:
                    source = _edit_method_job(
                        console,
                        candidate,
                        groups={"QUEUE-WIDE ARUCO"},
                        title="Queue-wide ArUco input",
                        selection_contexts=selection_contexts,
                    )
                except (
                    ValidationError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    yaml.YAMLError,
                ) as exc:
                    _show_input_error(
                        f"Invalid ArUco setting: {exc}. "
                        "Queue values were kept unchanged."
                    )
                    continue
                for target in jobs:
                    target.markers = source.markers.model_copy(deep=True)
                    _refresh_method_job_label(target)
                typer.echo(
                    "Applied the ArUco input to every queue job."
                )
            elif action == "6":
                candidate = _clone_method_job(jobs[0], jobs[0].label)
                try:
                    source = _edit_method_job(
                        console,
                        candidate,
                        groups={"COMMON EVALUATION"},
                        title="Queue-wide common evaluation",
                        selection_contexts=selection_contexts,
                    )
                except (
                    ValidationError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    yaml.YAMLError,
                ) as exc:
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
            elif action == "7":
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
            elif action == "8":
                candidate = _clone_method_job(jobs[0], jobs[0].label)
                try:
                    source = _edit_method_job(
                        console,
                        candidate,
                        groups={"OBSERVATION QUALITY BASELINE"},
                        title="Queue-wide observation-quality baseline",
                        selection_contexts=selection_contexts,
                    )
                except (
                    ValidationError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    yaml.YAMLError,
                ) as exc:
                    _show_input_error(
                        f"Invalid quality baseline: {exc}. "
                        "Queue values were kept unchanged."
                    )
                    continue
                for target in jobs:
                    target.observation_quality = (
                        source.observation_quality.model_copy(deep=True)
                    )
                    _refresh_method_job_label(target)
                typer.echo(
                    "Applied the observation-quality baseline to every "
                    "queue job; method overrides remain unchanged."
                )


__all__ = [
    'METHOD_JOB_GROUPS',
    '_setting_rows',
    '_edit_method_job',
    '_clone_method_job',
    '_method_queue',
]
