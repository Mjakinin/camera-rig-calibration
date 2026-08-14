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
from .wizard_bindings import current_wizard_bindings
from ..input.simulation_profiles import (
    SimulationCameraProfile,
    SimulationWorldProfile,
    bus_world_profile,
)
from ..input.simulation_routes import SimulationRouteAsset





@dataclass(frozen=True)
class WizardOutcome:
    config: RigConfig
    path: Path
    queued_runs: tuple["QueuedRun", ...] = ()
    batch_path: Path | None = None
    queue_paths: tuple[Path, ...] = ()

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
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    context_methods: dict[str, MethodSettings] = field(default_factory=dict)
    context_selections: dict[str, SelectionSettings] = field(
        default_factory=dict
    )
    deferred_selection_keys: set[str] = field(default_factory=set)
    context_deferred_selection_keys: dict[str, set[str]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SelectionDatasetContext:
    """A dataset whose existing observations may support an immediate choice."""

    key: str
    display_name: str
    dataset_root: Path | None
    static_cameras: tuple[StaticCameraSettings, ...]

    @property
    def observations_csv(self) -> Path | None:
        if self.dataset_root is None:
            return None
        candidate = (
            self.dataset_root
            / "observations"
            / "shared_all_aruco_observations.csv"
        )
        return candidate if candidate.is_file() else None


def _refresh_method_job_label(job: MethodQueueJob) -> str:
    job.label = current_wizard_bindings().method_job_label(job)
    return job.label


_MANUAL_SELECTION_LABELS = {
    "root_camera": "root_manual",
    "ap02_reference": "ref_manual",
    "single_marker": "single_manual",
    "multi_markers": "multi_manual",
}


def _pending_selection_keys(
    job: MethodQueueJob, context_key: str | None = None
) -> set[str]:
    if context_key is not None:
        contextual = job.context_deferred_selection_keys
        if context_key in contextual:
            return set(contextual[context_key])
    pending = set(job.deferred_selection_keys)
    if context_key is None:
        for contextual in job.context_deferred_selection_keys.values():
            pending.update(contextual)
    return pending


def _method_job_label(
    job: MethodQueueJob, context_key: str | None = None
) -> str:
    methods = (
        job.context_methods[context_key]
        if context_key is not None and context_key in job.context_methods
        else job.methods
    )
    baseline_label = automatic_method_label(
        job.method_id,
        methods=methods,
        markers=job.markers,
        observation_quality=job.observation_quality,
        colmap=job.colmap,
    )
    manual_tokens = [
        label
        for key, label in _MANUAL_SELECTION_LABELS.items()
        if key in _pending_selection_keys(job, context_key)
    ]
    if not manual_tokens:
        return baseline_label
    return safe_id("__".join((baseline_label, *manual_tokens)))


def _method_job_identity(job: MethodQueueJob) -> str:
    """Identify the complete requested job, including deferred selections."""

    payload = {
        "method_id": job.method_id,
        "methods": job.methods.model_dump(mode="json", exclude_none=True),
        "markers": job.markers.model_dump(mode="json", exclude_none=True),
        "observation_quality": job.observation_quality.model_dump(
            mode="json", exclude_none=True
        ),
        "colmap": job.colmap.model_dump(mode="json", exclude_none=True),
        "evaluation": job.evaluation.model_dump(mode="json", exclude_none=True),
        "selection": job.selection.model_dump(mode="json", exclude_none=True),
        "deferred_selection_keys": sorted(job.deferred_selection_keys),
        "context_methods": {
            key: value.model_dump(mode="json", exclude_none=True)
            for key, value in sorted(job.context_methods.items())
        },
        "context_selections": {
            key: value.model_dump(mode="json", exclude_none=True)
            for key, value in sorted(job.context_selections.items())
        },
        "context_deferred_selection_keys": {
            key: sorted(value)
            for key, value in sorted(
                job.context_deferred_selection_keys.items()
            )
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SimulationQueueJob:
    experiment_id: str
    parameters: dict[str, object]
    cameras: tuple[StaticCameraSettings, ...]
    moving_camera: MovingCameraSettings
    simulation: SimulationSettings
    prepared_root: Path | None
    source: str

    @property
    def input_mode(self) -> str:
        return (
            "reuse local dataset"
            if self.prepared_root is not None
            else "new capture required"
        )


_BusCamera = SimulationCameraProfile
_BusRoute = SimulationRouteAsset
_BusDefinition = SimulationWorldProfile


def _bus_definition(repository_root: Path) -> SimulationWorldProfile:
    """Compatibility facade for the reviewed built-in world profile."""

    return bus_world_profile(repository_root)


class WizardBack(Exception):
    """Return from a nested selection to the previous wizard menu."""


__all__ = [
    'WizardOutcome',
    'QueuedRun',
    'MethodQueueJob',
    'SelectionDatasetContext',
    '_refresh_method_job_label',
    '_MANUAL_SELECTION_LABELS',
    '_pending_selection_keys',
    '_method_job_label',
    '_method_job_identity',
    'SimulationQueueJob',
    '_BusCamera',
    '_BusRoute',
    '_BusDefinition',
    '_bus_definition',
    'WizardBack',
]
