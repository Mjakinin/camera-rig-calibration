"""Queue behavior grouped by one cohesive responsibility."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import Field, model_validator
from rich.console import Console
from rich.table import Table

from ..config import config_fingerprint, load_config, save_config
from ..config.models import RigConfig, StrictModel
from ..config.models import (
    DatasetSettings,
    EvaluationSettings,
    MarkerSettings,
    ObservationQualitySettings,
)
from ..dataset.discovery import safe_id
from ..dataset_identity import build_dataset_identity
from ..experiments import (
    automatic_method_label,
    evaluation_fingerprint,
    experiment_paths,
)
from ..filesystem import promote_directory, rename_with_retry
from ..methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from ..observations import (
    ResolvedSelections,
    freeze_selections,
    write_selection_candidates_csv,
)
from ..runtime import PipelineOrchestrator, observation_id
from ..preflight import (
    PreflightJob,
    QueuePreflightResult,
    run_queue_preflight,
)
from ..observation_quality import filter_observations
from ..publication import (
    publish_preparation_transaction,
    publish_queue_transaction,
)
from ..storage_layout import queue_temporary_root


@dataclass(frozen=True)
class SelectionReviewJob:
    """One independently filtered method job awaiting an attended selection."""

    entry_id: str
    config: RigConfig
    selections: ResolvedSelections
    output_directory: Path


QueueSelectionReviewer = Callable[
    [tuple[SelectionReviewJob, ...], Path],
    dict[str, dict[str, Any]],
]


from .common import (
    QueueSelectionReviewer,
    QueueObservationReviewer,
    _now,
    _write_json,
    _read_json,
    _config_with_detection_mode,
    _write_observation_detection_config,
)
from .models import (
    QueueConfig,
)
from .bindings import current_queue_bindings


class QueueRunnerBaseMixin:
    def __init__(
        self,
        repository_root: Path,
        console: Console | None = None,
        selection_reviewer: QueueSelectionReviewer | None = None,
        observation_reviewer: QueueObservationReviewer | None = None,
        reuse_method_intermediates: dict[str, Path] | None = None,
        rerun_metadata: dict[str, dict[str, Any]] | None = None,
        explicit_method_rerun: bool = False,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.console = console or Console()
        self.selection_reviewer = selection_reviewer
        self.observation_reviewer = observation_reviewer
        self.reuse_method_intermediates = dict(
            reuse_method_intermediates or {}
        )
        self.rerun_metadata = dict(rerun_metadata or {})
        self.explicit_method_rerun = explicit_method_rerun

    def show(self, queue: QueueConfig) -> None:
        table = Table(title=f"Experiment queue: {queue.id}")
        table.add_column("#", justify="right")
        table.add_column("Entry")
        table.add_column("Dataset")
        table.add_column("Method")
        table.add_column("Config", overflow="fold")
        for index, entry in enumerate(queue.entries, 1):
            config = load_config(entry.config)
            table.add_row(
                str(index),
                entry.id,
                config.dataset.id,
                ", ".join(config.methods.enabled),
                str(entry.config),
            )
        self.console.print(table)

    def validate(self, queue: QueueConfig) -> list[RigConfig]:
        PipelineOrchestrator = (
            current_queue_bindings().pipeline_orchestrator
        )
        configs: list[RigConfig] = []
        experiment_ids: set[tuple[str, str]] = set()
        for entry in queue.entries:
            config = load_config(entry.config)
            experiment_ids.add(
                (
                    config.dataset.category.value,
                    config.project.experiment_id or config.dataset.id,
                )
            )
            if queue.common is not None:
                mismatches = []
                if config.dataset != queue.common.dataset:
                    mismatches.append("dataset")
                if config.markers != queue.common.aruco:
                    mismatches.append("aruco")
                if (
                    config.observation_quality
                    != queue.common.observation_quality
                ):
                    mismatches.append("observation_quality")
                if config.evaluation != queue.common.evaluation:
                    mismatches.append("evaluation")
                if mismatches:
                    raise RuntimeError(
                        f"Queue entry '{entry.id}' conflicts with queue-common "
                        f"fields: {', '.join(mismatches)}"
                    )
            PipelineOrchestrator(
                self.repository_root,
                self.console,
                defer_evaluation=True,
            ).validate_ready(config)
            configs.append(config)
        if len(experiment_ids) > 1:
            raise RuntimeError(
                "A schema-v5 queue contains one experiment. Use a rigcal_batch "
                f"for multiple experiments; found: {sorted(experiment_ids)}"
            )
        return configs

    @staticmethod
    def _selection_group(config: RigConfig) -> tuple[object, ...]:
        return (
            config.project.experiment_id or config.dataset.id,
            config.dataset.id,
            tuple(camera.id for camera in config.static_cameras),
            config.moving_camera.id,
        )

    @staticmethod
    def _prepared_root_from_run(run: Path) -> Path | None:
        pointer = run / "00_INPUT" / "dataset_pointer.json"
        if not pointer.is_file():
            return None
        try:
            root = Path(
                json.loads(pointer.read_text(encoding="utf-8"))[
                    "dataset_root"
                ]
            ).resolve()
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
        return root if root.is_dir() else None

    def _retry_detector_on_prepared_input(
        self,
        *,
        transaction_root: Path,
        resolved_root: Path,
        prepared_root: Path,
        preparation_path: Path,
        configs: list[RigConfig],
        detection_mode: str,
    ) -> list[RigConfig]:
        """Replace only the mutable transaction observation view."""
        PipelineOrchestrator = (
            current_queue_bindings().pipeline_orchestrator
        )
        current_mode = configs[0].markers.detection_mode
        attempt_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        attempt = (
            resolved_root
            / "detector_attempts"
            / f"{attempt_stamp}_{current_mode}"
        )
        attempt.mkdir(parents=True, exist_ok=False)
        config_archive = attempt / "previous_configs"
        for index, config in enumerate(configs, 1):
            save_config(
                config,
                config_archive / f"{index:02d}_{config.project.run_label}.yaml",
            )
        previous_preflight = resolved_root / "preflight"
        observations = transaction_root / "dataset" / "observations"
        input_id: str | None = None
        if observations.is_dir():
            detection = _read_json(observations / "detection_config.json")
            input_id = (
                str(detection.get("input_id"))
                if detection.get("input_id")
                else None
            )

        updated = [
            _config_with_detection_mode(config, detection_mode)
            for config in configs
        ]
        retry_run = (
            transaction_root
            / "jobs"
            / "queue_preflight"
            / "detector_retries"
            / f"{attempt_stamp}_{detection_mode}"
        )
        previous_execution = attempt / "previous_execution"
        previous_retry_runs = sorted(
            (
                transaction_root
                / "jobs"
                / "queue_preflight"
                / "detector_retries"
            ).glob(f"*_{current_mode}"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        execution_source = (
            previous_retry_runs[0]
            if previous_retry_runs
            else preparation_path
        )
        if (execution_source / "logs").is_dir():
            shutil.copytree(
                execution_source / "logs",
                previous_execution / "logs",
                dirs_exist_ok=True,
            )
        for name in (
            "commands.txt",
            "environment.json",
            "requested_config.yaml",
            "resolved_config.yaml",
        ):
            source = execution_source / name
            if source.is_file():
                previous_execution.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, previous_execution / name)
        generated = PipelineOrchestrator(
            self.repository_root,
            self.console,
            defer_evaluation=True,
            job_id="queue_preflight_detector_retry",
            job_index=0,
            job_count=len(updated),
            transaction_root=transaction_root,
        ).detect_observations_only(
            updated[0],
            dataset_root=prepared_root,
            run_directory=retry_run,
        )
        retry_evidence = attempt / "retry_execution"
        retry_evidence.mkdir(parents=True, exist_ok=True)
        retry_logs = retry_run / "logs"
        if retry_logs.is_dir():
            shutil.copytree(
                retry_logs,
                retry_evidence / "logs",
                dirs_exist_ok=True,
            )
        retry_commands = retry_run / "commands.txt"
        if retry_commands.is_file():
            shutil.copy2(
                retry_commands,
                retry_evidence / "commands.txt",
            )
        if previous_preflight.is_dir():
            shutil.copytree(
                previous_preflight,
                attempt / "preflight",
                dirs_exist_ok=True,
            )
            shutil.rmtree(previous_preflight)
        if observations.is_dir():
            rename_with_retry(observations, attempt / "observations")
        if observations.exists():
            raise RuntimeError(
                "Detector retry destination unexpectedly exists: "
                f"{observations}"
            )
        promotion_mode = promote_directory(generated, observations)
        _write_observation_detection_config(
            observations,
            config=updated[0],
            input_id=input_id,
        )
        _write_json(
            attempt / "ATTEMPT.json",
            {
                "schema_version": 5,
                "status": "superseded_by_detector_retry",
                "previous_detection_mode": current_mode,
                "next_detection_mode": detection_mode,
                "created_at": _now(),
                "raw_images_reused": "dataset/raw_images",
                "capture_repeated": False,
                "video_extraction_repeated": False,
                "intrinsics_repeated": False,
                "retry_log": "retry_execution/logs",
                "observation_promotion": promotion_mode,
            },
        )
        return updated

    def _recover_interrupted_detector_retry(
        self,
        *,
        transaction_root: Path,
        resolved_root: Path,
        configs: list[RigConfig],
    ) -> tuple[list[RigConfig], str, str] | None:
        """Finish a detector retry whose completed output was not promoted."""
        observations = transaction_root / "dataset" / "observations"
        if observations.exists():
            return None
        retries = (
            transaction_root
            / "jobs"
            / "queue_preflight"
            / "detector_retries"
        )
        if not retries.is_dir():
            return None
        retry_runs = sorted(
            (item for item in retries.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for retry_run in retry_runs:
            generated = retry_run / "01_OBSERVATIONS"
            effective = _read_json(
                generated / "effective_detection_config.json"
            )
            next_mode = str(effective.get("mode", ""))
            if next_mode not in {
                "baseline",
                "subpixel_refined",
                "high_sensitivity",
            }:
                continue
            if not (
                generated
                / "shared_all_aruco_observations.csv"
            ).is_file():
                continue
            if any(
                config.markers.dictionary
                != str(effective.get("dictionary", ""))
                for config in configs
            ):
                continue
            suffix = f"_{next_mode}"
            if not retry_run.name.endswith(suffix):
                continue
            attempt_stamp = retry_run.name[: -len(suffix)]
            attempts = sorted(
                resolved_root.glob(
                    f"detector_attempts/{attempt_stamp}_*"
                )
            )
            attempt = next(
                (
                    item
                    for item in attempts
                    if (item / "observations").is_dir()
                    and not (item / "ATTEMPT.json").exists()
                ),
                None,
            )
            if attempt is None:
                continue
            current_mode = attempt.name[len(attempt_stamp) + 1 :]
            archived_detection = _read_json(
                attempt / "observations" / "detection_config.json"
            )
            input_id = (
                str(archived_detection["input_id"])
                if archived_detection.get("input_id")
                else None
            )
            updated = [
                _config_with_detection_mode(config, next_mode)
                for config in configs
            ]
            promotion_mode = promote_directory(generated, observations)
            _write_observation_detection_config(
                observations,
                config=updated[0],
                input_id=input_id,
            )
            _write_json(
                attempt / "ATTEMPT.json",
                {
                    "schema_version": 5,
                    "status": "superseded_by_detector_retry",
                    "previous_detection_mode": current_mode,
                    "next_detection_mode": next_mode,
                    "created_at": _now(),
                    "raw_images_reused": "dataset/raw_images",
                    "capture_repeated": False,
                    "video_extraction_repeated": False,
                    "intrinsics_repeated": False,
                    "retry_log": "retry_execution/logs",
                    "observation_promotion": promotion_mode,
                    "recovered_after_interrupted_promotion": True,
                },
            )
            return updated, current_mode, next_mode
        return None



__all__ = ['QueueRunnerBaseMixin']
