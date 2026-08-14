from __future__ import annotations

import json
import hashlib
import importlib.util
import csv
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.table import Table

from .components import register_builtin_components
from .config import config_fingerprint, load_config, save_config
from .config.models import RigConfig, effective_observation_quality
from .contracts import CommandSpec, RunContext
from .dataset.manifest import AutoSelection, load_dataset_manifest, save_dataset_manifest
from .dataset.validation import validate_dataset
from .input.preparation import build_preparation_plan, finalize_dataset
from .input.topics import resolve_rosbag_source
from .intrinsics_profiles import resolve_intrinsic_profile
from .methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from .experiments import (
    colmap_artifact_fingerprint,
    evaluation_fingerprint,
    experiment_paths,
    input_fingerprint,
    method_config_diff,
    method_fingerprint,
    method_result_label,
    write_experiment_manifest,
)
from .observations import (
    ResolvedSelections,
    freeze_selections,
    resolve_selections,
)
from .observation_quality import ObservationQualityError, filter_observations
from .progress import ProgressClock, progress_text, terminal_lines
from .pipeline import StageContract, validate_stage_dag
from .registry import calibration_methods, evaluators, input_adapters
from .results import write_comparison


from .runtime_services.common import (
    T,
    COMMAND_HEARTBEAT_SECONDS,
    BASE_RUN_DIRECTORIES,
    METHOD_DIRECTORIES,
    TERMINAL_PREFIXES,
    _now,
    _write_json,
    _read_json,
    _run_id,
    _run_directories,
    _automatic_scientific_selections,
    _materialize_tree,
    planned_stages,
    observation_id,
)
from .runtime_services.environment import EnvironmentMixin
from .runtime_services.observations import ObservationMixin
from .runtime_services.artifacts import ArtifactMixin
from .runtime_services.commands import CommandMixin
from .runtime_services.execution import ExecutionMixin


class PipelineOrchestrator(
    EnvironmentMixin,
    ObservationMixin,
    ArtifactMixin,
    CommandMixin,
    ExecutionMixin,
):
    def __init__(
        self,
        repository_root: Path,
        console: Console | None = None,
        selection_reviewer: Callable[
            [RigConfig, ResolvedSelections, Path], dict[str, Any]
        ]
        | None = None,
        defer_evaluation: bool = False,
        job_id: str | None = None,
        job_index: int = 1,
        job_count: int = 1,
        queue_started_monotonic: float | None = None,
        batch_started_monotonic: float | None = None,
        transaction_root: Path | None = None,
        reuse_intermediates_from: Path | None = None,
        rerun_metadata: dict[str, Any] | None = None,
        explicit_method_rerun: bool = False,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.console = console or Console()
        self.selection_reviewer = selection_reviewer
        self.defer_evaluation = defer_evaluation
        self.transaction_root = (
            transaction_root.resolve()
            if transaction_root is not None
            else None
        )
        self.reuse_intermediates_from = (
            reuse_intermediates_from.resolve()
            if reuse_intermediates_from is not None
            else None
        )
        self.reused_method_stages: tuple[str, ...] = ()
        self.rerun_metadata = dict(rerun_metadata or {})
        self.explicit_method_rerun = explicit_method_rerun
        self.progress = ProgressClock(
            job_id=job_id or "rigcal",
            job_index=job_index,
            job_count=job_count,
            queue_started_monotonic=queue_started_monotonic,
            batch_started_monotonic=batch_started_monotonic,
        )
        register_builtin_components()
        self.run_directory: Path | None = None
        self.manifest: dict[str, Any] = {}
        self.timings: dict[str, Any] = {}



def find_run(output_root: Path, run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    root = output_root.resolve()
    matches: list[Path] = []
    for manifest_path in root.rglob("run_manifest.json"):
        directory = manifest_path.parent
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if (
            directory.name == run_id_or_path
            or payload.get("run_id") == run_id_or_path
            or payload.get("execution_id") == run_id_or_path
        ):
            matches.append(directory)
    if not matches:
        raise FileNotFoundError(f"Run not found: {run_id_or_path}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Run ID is ambiguous across datasets: {run_id_or_path}; use its path"
        )
    return matches[0].resolve()

__all__ = [
    "PipelineOrchestrator",
    "planned_stages",
    "observation_id",
    "find_run",
]
