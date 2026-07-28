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

from .config import config_fingerprint, load_config, save_config
from .config.models import RigConfig, StrictModel
from .config.models import (
    DatasetSettings,
    EvaluationSettings,
    MarkerSettings,
    ObservationQualitySettings,
)
from .dataset.discovery import safe_id
from .experiments import (
    automatic_method_label,
    evaluation_fingerprint,
    experiment_paths,
)
from .filesystem import promote_directory, rename_with_retry
from .methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from .observations import (
    ResolvedSelections,
    ap03_candidate_rank,
    freeze_selections,
)
from .runtime import PipelineOrchestrator, observation_id
from .preflight import (
    PreflightJob,
    QueuePreflightResult,
    run_queue_preflight,
)
from .publication import (
    publish_preparation_transaction,
    publish_queue_transaction,
)
from .storage_layout import queue_temporary_root


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


@dataclass(frozen=True)
class ObservationReviewDecision:
    action: Literal["retry_detector", "continue_partial", "pause"]
    detection_mode: str | None = None

    def __post_init__(self) -> None:
        if self.action == "retry_detector" and self.detection_mode not in {
            "baseline",
            "subpixel_refined",
            "high_sensitivity",
        }:
            raise ValueError(
                "retry_detector requires a registered detection_mode"
            )
        if self.action != "retry_detector" and self.detection_mode is not None:
            raise ValueError(
                "Only retry_detector accepts a detection_mode"
            )


QueueObservationReviewer = Callable[
    [QueuePreflightResult, Path],
    ObservationReviewDecision,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _base_experiment_id(value: str) -> str:
    for mode in ("baseline", "subpixel_refined", "high_sensitivity"):
        suffix = f"__aruco_{mode}"
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _config_with_detection_mode(
    config: RigConfig, detection_mode: str
) -> RigConfig:
    base_id = _base_experiment_id(
        config.project.experiment_id or config.dataset.id
    )
    experiment_id = (
        safe_id(base_id)
        if detection_mode == "baseline"
        else safe_id(f"{base_id}__aruco_{detection_mode}")
    )
    markers = config.markers.model_copy(
        update={"detection_mode": detection_mode}
    )
    method_id = config.methods.enabled[0]
    label = automatic_method_label(
        method_id,
        methods=config.methods,
        markers=markers,
        observation_quality=config.observation_quality,
        colmap=config.colmap,
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={"id": experiment_id}, deep=True
                ),
                "project": config.project.model_copy(
                    update={
                        "experiment_id": experiment_id,
                        "run_label": label,
                    },
                    deep=True,
                ),
                "markers": markers,
            },
            deep=True,
        ).model_dump(mode="python")
    )


def _write_observation_detection_config(
    destination: Path,
    *,
    config: RigConfig,
    input_id: str | None,
) -> None:
    _write_json(
        destination / "detection_config.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "input_id": input_id,
            "observation_id": observation_id(config),
            "markers": config.markers.model_dump(mode="json"),
            "effective_detector": effective_detector_config(
                config.markers.detection_mode,
                config.markers.dictionary,
            ),
            "detector_contract": DETECTOR_CONTRACT,
            "observation_input_contract": "all_quality_passed_v1",
        },
    )


def _queue_job_fingerprint(config: RigConfig) -> str:
    payload = config.model_dump(mode="json", exclude_none=True)
    project = dict(payload.get("project", {}))
    project.pop("run_label", None)
    project.pop("duplicate_policy", None)
    payload["project"] = project
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _format_runtime(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_left = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_left:02d}s"
    if minutes:
        return f"{minutes}m{seconds_left:02d}s"
    return f"{seconds_left}s"


def _method_selection_summary(
    config: RigConfig, selections: ResolvedSelections
) -> str:
    method_id = config.methods.enabled[0]
    if method_id == "ap01":
        return f"root camera={selections.root_camera}"
    if method_id == "ap02":
        return (
            "reference marker="
            f"{selections.ap02_reference_marker_id}"
        )
    if method_id == "ap03":
        multi = ",".join(
            str(value) for value in selections.ap03_multi_marker_ids
        )
        return (
            f"single marker={selections.ap03_single_scale_marker_id}; "
            f"multi markers={multi}"
        )
    return "no guided root/marker selection"


def _configured_selection_summary(config: RigConfig) -> str:
    method_id = config.methods.enabled[0]
    if method_id == "ap01":
        return f"root camera={config.methods.ap01.root_camera}"
    if method_id == "ap02":
        return (
            "reference marker="
            f"{config.methods.ap02.reference_marker_id}"
        )
    if method_id == "ap03":
        marker_ids = config.methods.ap03.multi.marker_ids
        multi = (
            marker_ids
            if marker_ids == "auto"
            else ",".join(str(value) for value in marker_ids)
        )
        return (
            f"single marker={config.methods.ap03.single.scale_marker_id}; "
            f"multi markers={multi}"
        )
    return "no guided root/marker selection"


def _method_preflight_coverage(
    config: RigConfig, report: Any
) -> tuple[str, str]:
    selections = report.selections
    if selections is None:
        return "-", "; ".join(report.errors) or "selection unavailable"
    method_id = config.methods.enabled[0]
    expected = len(config.static_cameras)
    if method_id == "ap01":
        selected = next(
            (
                item
                for item in selections.payload["ap01_root_camera"][
                    "candidates"
                ]
                if str(item["id"]) == selections.root_camera
            ),
            {},
        )
        reachable = list(selected.get("reachable_cameras", []))
        missing = list(selected.get("unreachable_cameras", []))
        return (
            f"relay graph {len(reachable)}/{expected}",
            "missing: " + (", ".join(missing) if missing else "none"),
        )
    if method_id == "ap02" and report.ap02_graph_diagnosis is not None:
        diagnosis = report.ap02_graph_diagnosis
        causes = ", ".join(diagnosis.cause_codes)
        return (
            (
                f"Combined {len(diagnosis.reached_static_cameras)}/"
                f"{expected}; {len(diagnosis.components)} components"
            ),
            (
                "missing: "
                + (
                    ", ".join(diagnosis.missing_static_cameras)
                    if diagnosis.missing_static_cameras
                    else "none"
                )
                + f"; cause: {causes}"
            ),
        )
    if method_id == "ap03":
        candidates = {
            int(item["id"]): item
            for item in selections.payload["ap03_single_scale_marker"][
                "candidates"
            ]
        }
        supported = {
            str(camera)
            for marker_id in selections.ap03_multi_marker_ids
            for camera in candidates.get(marker_id, {}).get(
                "static_cameras", []
            )
        }
        missing = sorted(
            {camera.id for camera in config.static_cameras} - supported
        )
        return (
            f"scale support {len(supported)}/{expected}",
            (
                "COLMAP coverage is determined during reconstruction; "
                "without direct scale support: "
                + (", ".join(missing) if missing else "none")
            ),
        )
    return "registered method", "method-specific coverage unavailable"


def _selection_source(mode: str, *, reviewed: bool = False) -> str:
    if reviewed:
        return "manual review"
    if mode == "auto":
        return "automatic"
    if mode == "review_once":
        return "automatic proposal; manual review pending"
    return "validated explicit"


def _method_result_summary(path: Path) -> tuple[str, str]:
    """Return compact scientific metrics and the complete-log location."""
    details: list[str] = []
    result_payload: dict[str, Any] = {}
    result_path = path / "RESULT.json"
    if result_path.is_file():
        try:
            result_payload = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            result_payload = {}
    has_public_result = bool(result_payload)
    if result_payload.get("runtime_seconds") is not None:
        details.append(
            "method="
            + _format_runtime(float(result_payload["runtime_seconds"]))
        )
    timings_path = path / "provenance" / "timings.json"
    try:
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        timings = {}
    structured = timings.get("_structured", {})
    method_times = [
        float(item.get("stage_elapsed_seconds", 0.0))
        for stage_id, item in structured.items()
        if str(stage_id).startswith("method_") and isinstance(item, dict)
    ] if isinstance(structured, dict) else []
    if method_times and result_payload.get("runtime_seconds") is None:
        details.append(f"method={_format_runtime(max(method_times))}")

    if result_payload.get("primary_result"):
        details.append(f"primary={result_payload['primary_result']}")
    if result_payload.get("static_camera_count") is not None:
        details.append(f"cameras={result_payload['static_camera_count']}")
    marker = result_payload.get("reference_marker_id")
    if marker is not None:
        details.append(f"marker={marker}")
    status_paths = sorted((path / "diagnostics").rglob("METHOD_STATUS.json"))
    if status_paths and not has_public_result:
        try:
            status = json.loads(
                status_paths[0].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            status = {}
        if isinstance(status, dict):
            primary = status.get("primary_result")
            if primary:
                details.append(f"primary={primary}")
            cameras = status.get("available_static_cameras")
            if isinstance(cameras, list):
                details.append(f"cameras={len(cameras)}")
            marker = status.get("reference_marker_id")
            if marker is not None:
                details.append(f"marker={marker}")

    logs = path / "logs"
    log_path = str(logs if logs.is_dir() else path)
    return ", ".join(details) or "metrics unavailable", log_path


def _bind_prepared_dataset(
    config: RigConfig, prepared_root: Path
) -> RigConfig:
    """Make the once-prepared queue input authoritative for method jobs."""
    moving = config.moving_camera.model_copy(
        update={
            "intrinsics": None,
            "intrinsics_profile": None,
            "intrinsic_calibration_video": None,
            "intrinsic_calibration_images": None,
        },
        deep=True,
    )
    simulation = config.simulation.model_copy(
        update={"enabled": False},
        deep=True,
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "dataset": config.dataset.model_copy(
                    update={"prepared_root": prepared_root.resolve()},
                    deep=True,
                ),
                "moving_camera": moving,
                "simulation": simulation,
            },
            deep=True,
        ).model_dump(mode="python")
    )


def _print_queue_completion(
    console: Console,
    config: RigConfig,
    results: dict[str, dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> None:
    """Show one canonical, experiment-wide hand-off after all queue rows."""
    paths = experiment_paths(config)
    table = Table(
        title="Calibration queue completed",
        caption=(
            f"Experiment time: {_format_runtime(elapsed_seconds)} | "
            f"Results: {paths.root / 'RESULTS.txt'} | "
            f"Machine comparison: {paths.root / 'COMPARISON.json'}"
        ),
        expand=True,
    )
    table.add_column("Experiment")
    table.add_column("Method variant")
    table.add_column("Status")
    table.add_column("Method time")
    table.add_column("Key metrics", overflow="fold")
    table.add_column("Canonical result", overflow="fold")
    for entry_id, row in results.items():
        result = Path(str(row.get("result", "")))
        payload_path = result / "RESULT.json"
        payload: dict[str, Any] = {}
        if payload_path.is_file():
            try:
                payload = json.loads(
                    payload_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                payload = {}
        runtime = payload.get("runtime_seconds")
        metrics = []
        if payload.get("primary_result"):
            metrics.append(f"primary={payload['primary_result']}")
        if payload.get("static_camera_count") is not None:
            metrics.append(f"cameras={payload['static_camera_count']}")
        marker = payload.get("reference_marker_id")
        if marker is not None:
            metrics.append(f"marker={marker}")
        table.add_row(
            config.project.experiment_id or config.dataset.id,
            (
                f"{payload.get('method')}/{payload.get('label')}"
                if payload
                else entry_id
            ),
            (
                "available"
                if row.get("status") in {"completed", "duplicate_skipped"}
                else str(
                    row.get("failure", {}).get(
                        "cause_code", row.get("status", "unknown")
                    )
                )
            ),
            _format_runtime(float(runtime)) if runtime is not None else "-",
            ", ".join(metrics) or "-",
            str(result),
        )
    console.print(table)


class QueueEntry(StrictModel):
    id: str
    config: Path
    depends_on: list[str] = Field(default_factory=list)


class QueueCommon(StrictModel):
    dataset: DatasetSettings
    aruco: MarkerSettings = Field(default_factory=MarkerSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)


class QueueConfig(StrictModel):
    kind: Literal["rigcal_queue"] = "rigcal_queue"
    schema_version: Literal[5] = 5
    id: str
    continue_independent: bool = True
    common: QueueCommon | None = None
    entries: list[QueueEntry] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def require_schema_v5(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version")
        if version is None:
            return value
        if version != 5:
            raise ValueError(
                "Only schema_version 5 queues are supported. Recreate the "
                "queue with the current rigcal wizard."
            )
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> "QueueConfig":
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("queue entry IDs must be unique")
        known: set[str] = set()
        for entry in self.entries:
            missing = set(entry.depends_on) - set(ids)
            if missing:
                raise ValueError(
                    f"queue entry '{entry.id}' has unknown dependencies: "
                    f"{sorted(missing)}"
                )
            forward = set(entry.depends_on) - known
            if forward:
                raise ValueError(
                    f"queue dependencies must precede '{entry.id}': "
                    f"{sorted(forward)}"
                )
            known.add(entry.id)
        return self


class BatchEntry(StrictModel):
    experiment_id: str
    queue: Path


class BatchConfig(StrictModel):
    kind: Literal["rigcal_batch"] = "rigcal_batch"
    schema_version: Literal[1] = 1
    id: str
    continue_independent: bool = True
    queues: list[BatchEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entries(self) -> "BatchConfig":
        experiment_ids = [entry.experiment_id for entry in self.queues]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("batch experiment IDs must be unique")
        return self


def is_queue_config(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("kind") == "rigcal_queue"
        and payload.get("schema_version") == 5
    )


def is_batch_config(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("kind") == "rigcal_batch"


def load_batch(path: Path) -> BatchConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    batch = BatchConfig.model_validate(payload)
    return batch.model_copy(
        update={
            "queues": [
                entry.model_copy(
                    update={
                        "queue": (
                            entry.queue.resolve()
                            if entry.queue.is_absolute()
                            else (source.parent / entry.queue).resolve()
                        )
                    }
                )
                for entry in batch.queues
            ]
        },
        deep=True,
    )


def save_batch(
    batch_id: str,
    queues: list[tuple[str, Path]],
    destination: Path,
) -> Path:
    if not queues:
        raise ValueError("A batch must contain at least one experiment queue")
    payload = BatchConfig(
        id=batch_id,
        queues=[
            BatchEntry(
                experiment_id=experiment_id,
                queue=(
                    queue_path.resolve().relative_to(
                        destination.parent.resolve()
                    )
                    if queue_path.resolve().is_relative_to(
                        destination.parent.resolve()
                    )
                    else queue_path.resolve()
                ),
            )
            for experiment_id, queue_path in queues
        ],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(
            payload.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _load_queue_unpartitioned(path: Path) -> QueueConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Queue root must be a mapping: {source}")
    if payload.get("schema_version") != 5:
        raise ValueError(
            f"Only schema_version 5 queues are supported: {source}"
        )
    queue = QueueConfig.model_validate(payload)
    resolved_queue = queue.model_copy(
        update={
            "entries": [
                entry.model_copy(
                    update={
                        "config": (
                            entry.config.resolve()
                            if entry.config.is_absolute()
                            else (source.parent / entry.config).resolve()
                        )
                    }
                )
                for entry in queue.entries
            ]
        },
        deep=True,
    )
    return resolved_queue


def load_queue_partitions(path: Path) -> tuple[QueueConfig, ...]:
    """Load one strict schema-v5, single-experiment queue."""
    queue = _load_queue_unpartitioned(path)
    configs = [load_config(entry.config) for entry in queue.entries]
    identities = {
        (
            config.dataset.category.value,
            config.project.experiment_id or config.dataset.id,
        )
        for config in configs
    }
    if len(identities) != 1:
        raise ValueError(
            "A schema-v5 queue contains exactly one experiment. Use a "
            "rigcal_batch for multiple experiments."
        )
    return (queue,)


def load_queue(path: Path) -> QueueConfig:
    return load_queue_partitions(path)[0]


def save_queue(
    queue_id: str, configs: list[tuple[str, Path]], destination: Path
) -> Path:
    if not configs:
        raise ValueError("A queue must contain at least one method job")
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry_id, config_path in configs:
        try:
            relative = config_path.resolve().relative_to(destination.parent.resolve())
        except ValueError:
            relative = config_path.resolve()
        entries.append(
            {"id": entry_id, "config": str(relative), "depends_on": []}
        )
    loaded = [load_config(path) for _, path in configs]
    dataset_ids = {config.dataset.id for config in loaded}
    if len(dataset_ids) != 1:
        raise ValueError(
            "A schema-v5 queue contains exactly one dataset; create one queue "
            "per dataset"
        )
    first = loaded[0]
    payload = {
        "kind": "rigcal_queue",
        "schema_version": 5,
        "id": queue_id,
        "continue_independent": True,
        "common": {
            "dataset": first.dataset.model_dump(mode="json", exclude_none=True),
            "aruco": first.markers.model_dump(mode="json", exclude_none=True),
            "evaluation": first.evaluation.model_dump(
                mode="json", exclude_none=True
            ),
        },
        "entries": entries,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


class QueueRunner:
    def __init__(
        self,
        repository_root: Path,
        console: Console | None = None,
        selection_reviewer: QueueSelectionReviewer | None = None,
        observation_reviewer: QueueObservationReviewer | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.console = console or Console()
        self.selection_reviewer = selection_reviewer
        self.observation_reviewer = observation_reviewer

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

    def run(
        self,
        queue: QueueConfig,
        *,
        dry_run: bool = False,
        batch_started_monotonic: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        configs = self.validate(queue)
        preparation_modes = {
            config.project.execution_mode for config in configs
        }
        if "prepare_only" in preparation_modes:
            if preparation_modes != {"prepare_only"} or len(configs) != 1:
                raise RuntimeError(
                    "Prepare-only is one dedicated input job per experiment "
                    "and cannot be mixed with calibration methods."
                )
            entry = queue.entries[0]
            config = configs[0]
            transaction_root = queue_temporary_root(config, queue.id)
            orchestrator = PipelineOrchestrator(
                self.repository_root,
                self.console,
                defer_evaluation=True,
                job_id=entry.id,
                job_index=1,
                job_count=1,
                batch_started_monotonic=batch_started_monotonic,
                transaction_root=transaction_root,
            )
            if dry_run:
                orchestrator.show_dry_run(config)
                return {entry.id: {"status": "dry_run"}}
            preparation = orchestrator.run(config)
            path = publish_preparation_transaction(
                transaction_root,
                queue_id=queue.id,
                config=config,
                preparation=preparation,
            )
            if transaction_root.is_dir():
                shutil.rmtree(transaction_root)
            return {
                entry.id: {
                    "status": "completed",
                    "result": str(path),
                    "execution_mode": "prepare_only",
                }
            }
        source_fingerprints = {
            entry.id: config_fingerprint(config)
            for entry, config in zip(queue.entries, configs, strict=True)
        }
        transaction_root = queue_temporary_root(configs[0], queue.id)
        state_path = transaction_root / "queue_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        requested_queue_path = transaction_root / "requested_queue.yaml"
        if not requested_queue_path.is_file() and not dry_run:
            snapshot = queue.model_dump(mode="json", exclude_none=True)
            for item, source in zip(
                snapshot["entries"], queue.entries, strict=True
            ):
                item["config"] = str(source.config.resolve())
            temporary = requested_queue_path.with_suffix(".yaml.tmp")
            temporary.write_text(
                yaml.safe_dump(
                    snapshot, sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
            temporary.replace(requested_queue_path)
        previous_state: dict[str, Any] = {}
        if state_path.is_file() and not dry_run:
            try:
                candidate = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
                if candidate.get("queue_id") == queue.id:
                    previous_state = candidate
            except (OSError, json.JSONDecodeError):
                previous_state = {}
        previous_fingerprints = previous_state.get(
            "source_fingerprints", {}
        )
        results: dict[str, dict[str, Any]] = {
            entry.id: dict(previous_state.get("entries", {}).get(entry.id, {}))
            for entry in queue.entries
            if previous_fingerprints.get(entry.id)
            == source_fingerprints[entry.id]
        }
        resolved_configs: dict[str, str] = dict(
            previous_state.get("resolved_configs", {})
        )
        for index, entry in enumerate(queue.entries):
            resolved_path = Path(resolved_configs.get(entry.id, ""))
            if (
                resolved_path.is_file()
                and previous_fingerprints.get(entry.id)
                == source_fingerprints[entry.id]
            ):
                configs[index] = load_config(resolved_path)
            else:
                resolved_configs.pop(entry.id, None)

        resolved_root = transaction_root / "resolved"
        selection_cache: dict[
            tuple[object, ...], dict[str, Any]
        ] = {}
        preflight_preparation = str(
            previous_state.get("preflight_preparation", "")
        )
        preflight_reports: dict[str, Any] = {}
        observation_coverage_override = bool(
            previous_state.get("observation_coverage_override", False)
        )
        observation_review: dict[str, Any] = dict(
            previous_state.get("observation_review") or {}
        )

        def save_state() -> None:
            temporary = state_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": 5,
                        "queue_id": queue.id,
                        "updated_at": _now(),
                        "entries": results,
                        "source_fingerprints": source_fingerprints,
                        "resolved_configs": resolved_configs,
                        "preflight_preparation": preflight_preparation or None,
                        "observation_coverage_override": (
                            observation_coverage_override
                        ),
                        "observation_review": observation_review or None,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(state_path)
            statuses = {
                str(row.get("status", "pending"))
                for row in results.values()
            }
            _write_json(
                transaction_root / "queue_transaction.json",
                {
                    "schema_version": 5,
                    "queue_id": queue.id,
                    "status": (
                        "running" if "running" in statuses else "incomplete"
                    ),
                    "updated_at": _now(),
                    "requested_queue": str(requested_queue_path.resolve()),
                    "entries": {
                        key: value.get("status", "pending")
                        for key, value in results.items()
                    },
                },
            )

        def publish_terminal_outcome(entry_id: str) -> bool:
            """Publish one method outcome without waiting for later queue rows."""
            row = results[entry_id]
            original_status = str(row.get("status", ""))
            if original_status not in {"completed", "failed"}:
                return True
            try:
                published = publish_queue_transaction(
                    transaction_root,
                    queue_id=queue.id,
                    configs=configs,
                    results={entry_id: dict(row)},
                    finalize=False,
                )[entry_id]
            except Exception as exc:
                row.update(
                    {
                        "status": "publication_failed",
                        "method_status": original_status,
                        "publication_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )
                save_state()
                self.console.print(
                    f"[red]Publication failed for {entry_id}; the queue "
                    f"remains resumable: {exc}[/red]"
                )
                return False
            row.update(published)
            row.pop("method_status", None)
            row.pop("publication_error", None)
            if original_status == "completed":
                row["status"] = "completed"
                row["published"] = True
                row["published_at"] = _now()
            save_state()
            outcome = (
                "available"
                if row["status"] == "completed"
                else str(
                    row.get("failure", {}).get(
                        "cause_code", "failed attempt"
                    )
                )
            )
            summary, log_path = _method_result_summary(
                Path(str(row.get("result", "")))
            )
            self.console.print(
                f"[bold]{entry_id}: {outcome}[/bold] | "
                f"{summary} | logs: {log_path}"
            )
            return True

        def close_terminal_transaction() -> bool:
            if not all(
                row.get("status")
                in {
                    "completed",
                    "duplicate_skipped",
                    "failed_published",
                }
                for row in results.values()
            ):
                return False
            receipt = (
                configs[0].project.workspace_root.resolve()
                / "queues"
                / f"{queue.id}.published.json"
            )
            failed = any(
                row.get("status") == "failed_published"
                for row in results.values()
            )
            successful = any(
                row.get("status")
                in {"completed", "published", "duplicate_skipped"}
                for row in results.values()
            )
            _write_json(
                receipt,
                {
                    "schema_version": 5,
                    "queue_id": queue.id,
                    "status": "published",
                    "scientific_status": (
                        "partial"
                        if failed and successful
                        else "failed"
                        if failed
                        else "available"
                    ),
                    "published_at": _now(),
                    "entries": results,
                },
            )
            shutil.rmtree(transaction_root)
            return True

        def save_resolved_queue() -> None:
            if not resolved_configs:
                return
            entries = []
            for entry in queue.entries:
                path = resolved_configs.get(entry.id)
                if path is None:
                    continue
                entries.append(
                    {
                        "id": entry.id,
                        "config": str(Path(path).resolve()),
                        "depends_on": entry.depends_on,
                    }
                )
            payload = {
                "kind": "rigcal_queue",
                "schema_version": 5,
                "id": f"{queue.id}_resolved",
                "continue_independent": queue.continue_independent,
                # Resolved configs may point at the canonical prepared dataset.
                # The resolved queue must therefore snapshot those resolved
                # common values instead of repeating the requested queue block.
                "common": {
                    "dataset": configs[0].dataset.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "aruco": configs[0].markers.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "evaluation": configs[0].evaluation.model_dump(
                        mode="json", exclude_none=True
                    ),
                },
                "entries": entries,
                "source_queue_id": queue.id,
            }
            # The strict public queue omits provenance-only metadata.
            public = dict(payload)
            public.pop("source_queue_id")
            destination = resolved_root / "queue.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".yaml.tmp")
            temporary.write_text(
                yaml.safe_dump(
                    public, sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
            temporary.replace(destination)

        def persist_group(
            group: tuple[object, ...],
            *,
            prepared_root: Path | None,
            resolved: ResolvedSelections | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> None:
            for config_index, (entry, candidate) in enumerate(
                zip(queue.entries, configs, strict=True)
            ):
                if self._selection_group(candidate) != group:
                    continue
                updated = candidate
                if prepared_root is not None:
                    updated = _bind_prepared_dataset(
                        updated, prepared_root
                    )
                if resolved is not None:
                    updated = freeze_selections(
                        updated, resolved, overrides
                    )
                updated = RigConfig.model_validate(
                    updated.model_dump(mode="python")
                )
                configs[config_index] = updated
                destination = (
                    resolved_root
                    / f"{config_index + 1:02d}_{entry.id}_resolved.yaml"
                )
                save_config(updated, destination)
                resolved_configs[entry.id] = str(destination.resolve())
            save_resolved_queue()
            save_state()

        def review_and_freeze(
            config: RigConfig,
            resolved: ResolvedSelections,
            run_directory: Path,
        ) -> dict[str, Any]:
            if self.selection_reviewer is None:
                raise RuntimeError(
                    "Selection review requires an interactive terminal"
                )
            group = self._selection_group(config)
            if group not in selection_cache:
                group_methods = list(
                    dict.fromkeys(
                        method_id
                        for candidate in configs
                        if self._selection_group(candidate) == group
                        for method_id in candidate.methods.enabled
                    )
                )
                review_config = config.model_copy(
                    update={
                        "methods": config.methods.model_copy(
                            update={"enabled": group_methods},
                            deep=True,
                        )
                    },
                    deep=True,
                )
                selection_cache[group] = self.selection_reviewer(
                    review_config, resolved, run_directory
                )
            prepared_root = self._prepared_root_from_run(run_directory)
            persist_group(
                group,
                prepared_root=prepared_root,
                resolved=resolved,
                overrides=selection_cache[group],
            )
            self.console.print(
                f"[green]Frozen prompt-free queue: "
                f"{resolved_root / 'queue.yaml'}[/green]"
            )
            return selection_cache[group]

        if not dry_run:
            save_state()
        queue_started = time.monotonic()
        if not dry_run:
            preparation_path = Path(preflight_preparation)
            preparation_manifest_path = (
                preparation_path / "run_manifest.json"
            )
            if not preparation_manifest_path.is_file():
                base = configs[0]
                prep_config = RigConfig.model_validate(
                    base.model_copy(
                        update={
                            "project": base.project.model_copy(
                                update={
                                    "execution_mode": "prepare_only",
                                    "run_label": "queue_preflight",
                                    "duplicate_policy": "skip",
                                }
                            ),
                            "selection": base.selection.model_copy(
                                update={"mode": "auto"}
                            ),
                            "markers": base.markers.model_copy(
                                update={"accepted_ids": "all_detected"}
                            ),
                            "observation_quality": ObservationQualitySettings(),
                            "evaluation": base.evaluation.model_copy(
                                update={"enabled": False}
                            ),
                        },
                        deep=True,
                    ).model_dump(mode="python")
                )
                self.console.print(
                    "\n[bold]QUEUE PREFLIGHT — prepare input and raw "
                    "observations once[/bold]"
                )
                preparation_path = PipelineOrchestrator(
                    self.repository_root,
                    self.console,
                    defer_evaluation=True,
                    job_id="queue_preflight",
                    job_index=0,
                    job_count=len(queue.entries),
                    queue_started_monotonic=queue_started,
                    batch_started_monotonic=batch_started_monotonic,
                    transaction_root=transaction_root,
                ).run(prep_config)
                preflight_preparation = str(preparation_path)
                save_state()
                preparation_manifest_path = (
                    preparation_path / "run_manifest.json"
                )
            preparation_manifest = json.loads(
                preparation_manifest_path.read_text(encoding="utf-8")
            )
            raw_observations_root = Path(
                str(preparation_manifest["observations_root"])
            )
            prepared_root = self._prepared_root_from_run(preparation_path)
            if prepared_root is None:
                raise RuntimeError(
                    "Queue preflight preparation has no reusable dataset pointer"
                )
            recovered_retry = self._recover_interrupted_detector_retry(
                transaction_root=transaction_root,
                resolved_root=resolved_root,
                configs=configs,
            )
            if recovered_retry is not None:
                configs, previous_mode, recovered_mode = recovered_retry
                attempted_modes = list(
                    observation_review.get("attempted_modes", [])
                )
                for mode in (previous_mode, recovered_mode):
                    if mode not in attempted_modes:
                        attempted_modes.append(mode)
                observation_review.update(
                    {
                        "status": "retrying_detector_recovered",
                        "attempted_modes": attempted_modes,
                        "current_detection_mode": recovered_mode,
                        "capture_repeated": False,
                        "video_extraction_repeated": False,
                        "intrinsics_repeated": False,
                        "recovered_after_interrupted_promotion": True,
                        "updated_at": _now(),
                    }
                )
                for index, (entry, config) in enumerate(
                    zip(queue.entries, configs, strict=True), 1
                ):
                    destination = (
                        resolved_root
                        / f"{index:02d}_{entry.id}_detector_retry.yaml"
                    )
                    save_config(config, destination)
                    resolved_configs[entry.id] = str(
                        destination.resolve()
                    )
                    results[entry.id] = {
                        "status": "retrying_observations",
                        "detection_mode": recovered_mode,
                        "capture_reused": True,
                    }
                save_state()
                self.console.print(
                    "[green]Recovered the completed detector retry after "
                    "a filesystem lock. Capture, frames, intrinsics and "
                    "ArUco detection were reused.[/green]"
                )
            preflight_result = run_queue_preflight(
                (
                    PreflightJob(entry.id, config)
                    for entry, config in zip(
                        queue.entries, configs, strict=True
                    )
                ),
                raw_observations_csv=(
                    raw_observations_root
                    / "shared_all_aruco_observations.csv"
                ),
                dataset_root=prepared_root,
                output_directory=resolved_root / "preflight",
                repository_root=self.repository_root,
            )
            required_total = sum(
                item.required
                and item.camera_id != configs[0].moving_camera.id
                for item in preflight_result.camera_coverage
            )
            required_observed = sum(
                item.required
                and item.camera_id != configs[0].moving_camera.id
                and item.raw_detection_count > 0
                for item in preflight_result.camera_coverage
            )
            gate_text = (
                "OBSERVATION REVIEW REQUIRED"
                if preflight_result.review_required
                else "READY"
            )
            self.console.print(
                f"\n[bold]Queue observation status:[/bold] {gate_text} | "
                f"required static cameras {required_observed}/"
                f"{required_total} | detector "
                f"{configs[0].markers.detection_mode}"
            )
            if preflight_result.review_reasons:
                self.console.print(
                    "Review reasons: "
                    + ", ".join(preflight_result.review_reasons)
                )
            table = Table(title="Method readiness")
            table.add_column("Job")
            table.add_column("Method")
            table.add_column("Status")
            table.add_column("Accepted", justify="right")
            table.add_column("Selection after preflight", overflow="fold")
            table.add_column("Method coverage", overflow="fold")
            table.add_column("Missing / reason", overflow="fold")
            for entry, config, report in zip(
                queue.entries, configs, preflight_result.jobs, strict=True
            ):
                coverage, reason = _method_preflight_coverage(
                    config, report
                )
                table.add_row(
                    entry.id,
                    config.methods.enabled[0],
                    (
                        "WAITING_REVIEW"
                        if preflight_result.review_required
                        else report.status
                    ),
                    (
                        str(report.filter_result.accepted_count)
                        if report.filter_result is not None
                        else "0"
                    ),
                    (
                        (
                            _method_selection_summary(
                                config, report.selections
                            )
                            + " ["
                            + _selection_source(config.selection.mode)
                            + "]"
                        )
                        if report.selections is not None
                        else "-"
                    ),
                    coverage,
                    reason,
                )
            self.console.print(table)
            coverage_table = Table(
                title=(
                    "ArUco camera coverage — "
                    f"{configs[0].markers.detection_mode}"
                )
            )
            coverage_table.add_column("Camera")
            coverage_table.add_column("Required")
            coverage_table.add_column("Raw detections", justify="right")
            for entry in queue.entries:
                coverage_table.add_column(
                    f"{entry.id} accepted",
                    justify="right",
                )
            coverage_table.add_column("Marker IDs", overflow="fold")
            reports_by_id = {
                entry.id: report
                for entry, report in zip(
                    queue.entries,
                    preflight_result.jobs,
                    strict=True,
                )
            }
            for camera in preflight_result.camera_coverage:
                coverage_table.add_row(
                    camera.camera_id,
                    "yes" if camera.required else "no",
                    str(camera.raw_detection_count),
                    *[
                        str(
                            next(
                                (
                                    item.accepted_observation_count
                                    for item in reports_by_id[
                                        entry.id
                                    ].camera_coverage
                                    if item.camera_id == camera.camera_id
                                ),
                                0,
                            )
                        )
                        for entry in queue.entries
                    ],
                    ",".join(map(str, camera.marker_ids)) or "-",
                )
            self.console.print(coverage_table)
            preflight_reports = {
                entry.id: report
                for entry, report in zip(
                    queue.entries, preflight_result.jobs, strict=True
                )
            }
            for entry, report in zip(
                queue.entries, preflight_result.jobs, strict=True
            ):
                if not report.runnable:
                    results[entry.id] = {
                        "status": "failed_preflight",
                        "config": str(entry.config),
                        "preflight": str(report.output_directory),
                        "errors": list(report.errors),
                        "warnings": list(report.warnings),
                    }
            if (
                preflight_result.review_required
                and not observation_coverage_override
            ):
                reviewed = (
                    self.observation_reviewer(
                        preflight_result,
                        resolved_root / "preflight",
                    )
                    if self.observation_reviewer is not None
                    else ObservationReviewDecision("pause")
                )
                decision = (
                    ObservationReviewDecision(
                        "continue_partial" if reviewed else "pause"
                    )
                    if isinstance(reviewed, bool)
                    else reviewed
                )
                attempted_modes = list(
                    observation_review.get("attempted_modes", [])
                )
                current_mode = configs[0].markers.detection_mode
                if current_mode not in attempted_modes:
                    attempted_modes.append(current_mode)
                observation_review.update(
                    {
                        "status": decision.action,
                        "review_reasons": list(
                            preflight_result.review_reasons
                        ),
                        "missing_required_cameras": list(
                            preflight_result.missing_required_cameras
                        ),
                        "attempted_modes": attempted_modes,
                        "current_detection_mode": current_mode,
                        "updated_at": _now(),
                    }
                )
                reviews = list(observation_review.get("reviews", []))
                reviews.append(
                    {
                        "reviewed_at": _now(),
                        "detection_mode": current_mode,
                        "decision": decision.action,
                        "next_detection_mode": decision.detection_mode,
                        "review_reasons": list(
                            preflight_result.review_reasons
                        ),
                        "missing_required_cameras": list(
                            preflight_result.missing_required_cameras
                        ),
                        "ap02_combined_graphs": {
                            report.job_id: (
                                report.ap02_graph_diagnosis.model_dump()
                            )
                            for report in preflight_result.jobs
                            if report.ap02_graph_diagnosis is not None
                            and not report.ap02_graph_diagnosis.complete
                        },
                    }
                )
                observation_review["reviews"] = reviews
                if decision.action == "retry_detector":
                    assert decision.detection_mode is not None
                    if decision.detection_mode == current_mode:
                        raise RuntimeError(
                            "Detector retry must select a different mode"
                        )
                    configs = self._retry_detector_on_prepared_input(
                        transaction_root=transaction_root,
                        resolved_root=resolved_root,
                        prepared_root=prepared_root,
                        preparation_path=preparation_path,
                        configs=configs,
                        detection_mode=decision.detection_mode,
                    )
                    if decision.detection_mode not in attempted_modes:
                        attempted_modes.append(decision.detection_mode)
                    observation_review.update(
                        {
                            "status": "retrying_detector",
                            "attempted_modes": attempted_modes,
                            "current_detection_mode": (
                                decision.detection_mode
                            ),
                            "capture_repeated": False,
                            "video_extraction_repeated": False,
                            "intrinsics_repeated": False,
                        }
                    )
                    for index, (entry, config) in enumerate(
                        zip(queue.entries, configs, strict=True), 1
                    ):
                        destination = (
                            resolved_root
                            / f"{index:02d}_{entry.id}_detector_retry.yaml"
                        )
                        save_config(config, destination)
                        resolved_configs[entry.id] = str(
                            destination.resolve()
                        )
                        results[entry.id] = {
                            "status": "retrying_observations",
                            "detection_mode": decision.detection_mode,
                            "capture_reused": True,
                        }
                    preflight_reports.clear()
                    save_state()
                    self.console.print(
                        "[green]Detector retry completed on the existing "
                        "normalized frames. Re-running quality and graph "
                        "preflight now.[/green]"
                    )
                    return self.run(
                        queue,
                        dry_run=dry_run,
                        batch_started_monotonic=batch_started_monotonic,
                    )
                if decision.action == "pause":
                    observation_review["status"] = "waiting"
                    for entry, report in zip(
                        queue.entries,
                        preflight_result.jobs,
                        strict=True,
                    ):
                        results[entry.id] = {
                            "status": "waiting_for_observation_review",
                            "preflight_status": report.status,
                            "preflight": str(
                                resolved_root / "preflight"
                            ),
                            "review_reasons": list(
                                preflight_result.review_reasons
                            ),
                            "missing_required_cameras": list(
                                preflight_result.missing_required_cameras
                            ),
                            "detection_mode": current_mode,
                            "errors": list(report.errors),
                            "warnings": list(report.warnings),
                        }
                    save_state()
                    return results
                observation_coverage_override = True
                observation_review.update(
                    {
                        "status": "confirmed_diagnostic_partial",
                        "confirmed_at": _now(),
                    }
                )
                override_payload = {
                    "schema_version": 5,
                    "status": "confirmed_diagnostic_override",
                    "quality_status": "partial_coverage",
                    "detection_mode": current_mode,
                    "review_reasons": list(
                        preflight_result.review_reasons
                    ),
                    "missing_required_cameras": list(
                        preflight_result.missing_required_cameras
                    ),
                    "ap02_combined_graphs": {
                        report.job_id: (
                            report.ap02_graph_diagnosis.model_dump()
                        )
                        for report in preflight_result.jobs
                        if report.ap02_graph_diagnosis is not None
                        and not report.ap02_graph_diagnosis.complete
                    },
                    "confirmed_at": _now(),
                    "warning": (
                        "The operator explicitly continued with incomplete "
                        "observation coverage. Results are diagnostic; "
                        "cross-component camera relationships are not "
                        "observable."
                    ),
                }
                for target in (
                    resolved_root
                    / "preflight"
                    / "OBSERVATION_REVIEW_OVERRIDE.json",
                    raw_observations_root
                    / "OBSERVATION_REVIEW_OVERRIDE.json",
                ):
                    _write_json(target, override_payload)
                if preflight_result.missing_required_cameras:
                    for target in (
                        resolved_root
                        / "preflight"
                        / "REQUIRED_CAMERA_OVERRIDE.json",
                        raw_observations_root
                        / "REQUIRED_CAMERA_OVERRIDE.json",
                    ):
                        _write_json(target, override_payload)
                for report in preflight_result.jobs:
                    _write_json(
                        report.output_directory
                        / "OBSERVATION_REVIEW_OVERRIDE.json",
                        override_payload,
                    )
                save_state()
            if not preflight_result.ready:
                save_state()
                self.console.print(
                    "[red]No calibration method is runnable. Failed jobs remain "
                    "available as non-authoritative scientific attempts.[/red]"
                )
                results = publish_queue_transaction(
                    transaction_root,
                    queue_id=queue.id,
                    configs=configs,
                    results=results,
                )
                close_terminal_transaction()
                return results
            failed_count = sum(
                1 for report in preflight_result.jobs if not report.runnable
            )
            if failed_count:
                self.console.print(
                    f"[yellow]{failed_count} queue job(s) failed preflight and "
                    "will be skipped; independent runnable jobs continue.[/yellow]"
                )

            overrides_by_job: dict[str, dict[str, Any]] = {}
            review_jobs = [
                SelectionReviewJob(
                    entry_id=entry.id,
                    config=config,
                    selections=report.selections,
                    output_directory=report.output_directory,
                )
                for entry, config, report in zip(
                    queue.entries,
                    configs,
                    preflight_result.jobs,
                    strict=True,
                )
                if report.runnable
                and config.selection.mode == "review_once"
                and report.selections is not None
            ]
            if review_jobs:
                if self.selection_reviewer is None:
                    manual_entries = {
                        review.entry_id for review in review_jobs
                    }
                    for entry in queue.entries:
                        results[entry.id] = {
                            "status": (
                                "waiting_for_selection"
                                if entry.id in manual_entries
                                else "ready_after_preflight"
                            ),
                            "preflight": str(
                                resolved_root / "preflight"
                            ),
                        }
                    save_state()
                    return results
                overrides_by_job = self.selection_reviewer(
                    tuple(review_jobs),
                    resolved_root / "preflight",
                )
                expected_reviews = {
                    review.entry_id for review in review_jobs
                }
                if set(overrides_by_job) != expected_reviews:
                    missing = sorted(
                        expected_reviews - set(overrides_by_job)
                    )
                    unexpected = sorted(
                        set(overrides_by_job) - expected_reviews
                    )
                    raise RuntimeError(
                        "Queue selection review returned an incomplete job "
                        f"mapping; missing={missing}, unexpected={unexpected}"
                    )

            selection_errors: dict[str, str] = {}
            frozen_selection_rows: list[
                tuple[str, str, str, str]
            ] = []
            for index, (entry, config, report) in enumerate(
                zip(
                    queue.entries,
                    configs,
                    preflight_result.jobs,
                    strict=True,
                )
            ):
                if not report.runnable or report.selections is None:
                    continue
                updated = _bind_prepared_dataset(config, prepared_root)
                try:
                    updated = freeze_selections(
                        updated,
                        report.selections,
                        overrides_by_job.get(entry.id),
                    )
                except ValueError as exc:
                    selection_errors[entry.id] = str(exc)
                    continue
                updated = RigConfig.model_validate(
                    updated.model_dump(mode="python")
                )
                configs[index] = updated
                destination = (
                    resolved_root
                    / f"{index + 1:02d}_{entry.id}_resolved.yaml"
                )
                save_config(updated, destination)
                resolved_configs[entry.id] = str(destination.resolve())
                frozen_selection_rows.append(
                    (
                        entry.id,
                        config.methods.enabled[0],
                        _selection_source(
                            config.selection.mode,
                            reviewed=entry.id in overrides_by_job,
                        ),
                        _configured_selection_summary(updated),
                    )
                )
            if frozen_selection_rows:
                selection_table = Table(
                    title="Selections frozen before calibration"
                )
                selection_table.add_column("Job")
                selection_table.add_column("Method")
                selection_table.add_column("Source")
                selection_table.add_column("Final root / marker selection")
                for row in frozen_selection_rows:
                    selection_table.add_row(*row)
                self.console.print(selection_table)
            if selection_errors:
                for entry in queue.entries:
                    own_error = selection_errors.get(entry.id)
                    if own_error is None:
                        continue
                    results[entry.id] = {
                        "status": "failed_preflight",
                        "config": str(entry.config),
                        "preflight": str(resolved_root / "preflight"),
                        "errors": [own_error],
                    }
                _write_json(
                    resolved_root
                    / "preflight"
                    / "selection_validation_failure.json",
                    {
                        "schema_version": 5,
                        "status": "FAILED_PREFLIGHT",
                        "errors": selection_errors,
                        "methods_may_start": any(
                            entry.id not in selection_errors
                            and preflight_reports.get(entry.id) is not None
                            and preflight_reports[entry.id].runnable
                            for entry in queue.entries
                        ),
                    },
                )
                save_state()
                self.console.print(
                    "[yellow]Incompatible selections failed only their own "
                    "jobs; independent runnable jobs continue.[/yellow]"
                )
            canonical_observations = (
                transaction_root / "dataset" / "observations"
            )
            authoritative_report = next(
                (
                    report
                    for report in preflight_result.jobs
                    if report.runnable
                    and report.filter_result is not None
                    and report.selections is not None
                ),
                None,
            )
            if authoritative_report is None:
                raise RuntimeError(
                    "No runnable preflight job can finalize the shared "
                    "observation evidence"
                )
            authoritative_observations = (
                authoritative_report.filter_result.filtered_observations_root
            )
            for name in (
                "SELECTION_CANDIDATES.json",
                "REFERENCE_SELECTIONS.json",
                "REFERENCE_MARKER_ID.txt",
            ):
                source = authoritative_observations / name
                if not source.is_file():
                    raise RuntimeError(
                        "Preflight selection evidence is missing: "
                        f"{source}"
                    )
                shutil.copy2(source, canonical_observations / name)
            quality_destination = (
                canonical_observations / "quality" / "queue"
            )
            shutil.copytree(
                resolved_root / "preflight",
                quality_destination,
                dirs_exist_ok=True,
            )
            _write_json(
                canonical_observations / "QUEUE_SELECTIONS.json",
                {
                    "schema_version": 5,
                    "layout_version": 2,
                    "selection_mode": configs[0].selection.mode,
                    "reviewed_once": bool(review_jobs),
                    "jobs": [
                        {
                            "job_id": entry.id,
                            "method": config.methods.enabled[0],
                            "root_camera": config.methods.ap01.root_camera,
                            "ap02_reference_marker_id": (
                                config.methods.ap02.reference_marker_id
                            ),
                            "ap03_single_scale_marker_id": (
                                config.methods.ap03_single.scale_marker_id
                            ),
                            "ap03_multi_marker_ids": (
                                config.methods.ap03_multi.marker_ids
                            ),
                            "evaluation_anchor_marker_id": (
                                config.evaluation.anchor_marker_id
                            ),
                        }
                        for entry, config in zip(
                            queue.entries, configs, strict=True
                        )
                    ],
                },
            )
            _write_json(
                canonical_observations / "PUBLICATION_COMPLETE.json",
                {
                    "schema_version": 5,
                    "layout_version": 2,
                    "status": "complete",
                    "selection_files": [
                        "SELECTION_CANDIDATES.json",
                        "REFERENCE_SELECTIONS.json",
                        "REFERENCE_MARKER_ID.txt",
                    ],
                    "quality_directory": "quality",
                    "debug_images": "debug_images",
                    "detection_mode": configs[0].markers.detection_mode,
                    "observation_id": observation_id(configs[0]),
                    "finalized_at": _now(),
                },
            )
            detector_attempts = resolved_root / "detector_attempts"
            if detector_attempts.is_dir():
                shutil.copytree(
                    detector_attempts,
                    transaction_root
                    / "dataset"
                    / "metadata"
                    / "detector_attempts",
                    dirs_exist_ok=True,
                )
            save_resolved_queue()
            save_state()
            published_dataset = publish_preparation_transaction(
                transaction_root,
                queue_id=queue.id,
                config=configs[0],
                preparation=preparation_path,
            )
            self.console.print(
                "[green]Complete immutable dataset published before method "
                f"execution: {published_dataset}[/green]"
            )

        seen_jobs: dict[str, str] = {}
        for index, (entry, config) in enumerate(
            zip(queue.entries, configs, strict=True), 1
        ):
            report = preflight_reports.get(entry.id)
            if report is not None and not report.runnable:
                self.console.print(
                    f"[yellow]QUEUE {index}/{len(queue.entries)} — {entry.id}: "
                    "skipped after its own failed preflight[/yellow]"
                )
                continue
            previous = results.get(entry.id, {})
            previous_result = Path(str(previous.get("result", "")))
            if (
                previous.get("status") == "publication_failed"
                and previous.get("method_status") in {"completed", "failed"}
                and (previous_result / "run_manifest.json").is_file()
            ):
                previous["status"] = str(previous["method_status"])
                self.console.print(
                    f"[yellow]QUEUE {index}/{len(queue.entries)} — "
                    f"{entry.id}: retrying publication of the completed "
                    "method; calibration is not rerun[/yellow]"
                )
                if not publish_terminal_outcome(entry.id):
                    break
                seen_jobs.setdefault(
                    _queue_job_fingerprint(config), entry.id
                )
                continue
            if (
                previous.get("status")
                in {"completed", "published", "duplicate_skipped"}
                and (previous_result / "run_manifest.json").is_file()
            ):
                seen_jobs.setdefault(
                    _queue_job_fingerprint(config), entry.id
                )
                self.console.print(
                    f"[dim]QUEUE {index}/{len(queue.entries)} — "
                    f"{entry.id}: already completed; skipped[/dim]"
                )
                continue
            failed_dependencies = [
                dependency
                for dependency in entry.depends_on
                if results.get(dependency, {}).get("status")
                not in {"completed", "published", "duplicate_skipped"}
            ]
            if failed_dependencies:
                results[entry.id] = {
                    "status": "skipped_dependency",
                    "dependencies": failed_dependencies,
                }
                save_state()
                continue
            job_fingerprint = _queue_job_fingerprint(config)
            duplicate_of = seen_jobs.get(job_fingerprint)
            if duplicate_of is not None:
                results[entry.id] = {
                    "status": "duplicate_skipped",
                    "duplicate_of": duplicate_of,
                    "finished_at": _now(),
                    "reason": (
                        "identical method/input configuration already exists "
                        "earlier in this queue"
                    ),
                }
                save_state()
                self.console.print(
                    f"[yellow]QUEUE {index}/{len(queue.entries)} — "
                    f"{entry.id}: exact duplicate of {duplicate_of}; skipped[/yellow]"
                )
                continue
            seen_jobs[job_fingerprint] = entry.id
            self.console.print(
                f"\n[bold]QUEUE {index}/{len(queue.entries)} — "
                f"{entry.id}[/bold]"
            )
            if dry_run:
                PipelineOrchestrator(
                    self.repository_root, self.console
                ).show_dry_run(config)
                results[entry.id] = {"status": "dry_run"}
                continue
            results[entry.id] = {
                "status": "running",
                "started_at": _now(),
                "config": str(entry.config),
                "source_config_fingerprint": source_fingerprints[
                    entry.id
                ],
            }
            save_state()
            orchestrator = PipelineOrchestrator(
                self.repository_root,
                self.console,
                selection_reviewer=(
                    review_and_freeze
                    if self.selection_reviewer is not None
                    else None
                ),
                defer_evaluation=True,
                job_id=entry.id,
                job_index=index,
                job_count=len(queue.entries),
                queue_started_monotonic=queue_started,
                batch_started_monotonic=batch_started_monotonic,
                transaction_root=transaction_root,
            )
            try:
                resume = (
                    previous_result
                    if previous.get("status")
                    in {
                        "failed",
                        "interrupted",
                        "waiting_for_selection",
                    }
                    and (previous_result / "run_manifest.json").is_file()
                    else None
                )
                path = orchestrator.run(
                    config if resume is None else None,
                    resume_directory=resume,
                )
                manifest_path = path / "run_manifest.json"
                manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file()
                    else {}
                )
                queue_job_preflight = (
                    resolved_root / "preflight" / "jobs" / entry.id
                )
                if queue_job_preflight.is_dir() and path.is_dir():
                    snapshot = path / "preflight" / "queue_snapshot"
                    snapshot.mkdir(parents=True, exist_ok=True)
                    for name in (
                        "preflight_summary.json",
                        "observation_filter_summary.json",
                        "accepted_observations.csv",
                        "rejected_observations.csv",
                        "REQUIRED_CAMERA_OVERRIDE.json",
                        "OBSERVATION_REVIEW_OVERRIDE.json",
                        "AP02_COMBINED_GRAPH.json",
                        "AP02_COMBINED_GRAPH.txt",
                    ):
                        source = queue_job_preflight / name
                        if source.is_file():
                            shutil.copy2(source, snapshot / name)
                status = str(manifest.get("status", "completed"))
                if status == "completed" and not path.resolve().is_relative_to(
                    transaction_root.resolve()
                ):
                    status = "duplicate_skipped"
                results[entry.id].update(
                    {
                        "status": status,
                        "finished_at": _now(),
                        "result": str(path),
                    }
                )
                save_state()
                prepared_root = self._prepared_root_from_run(path)
                if prepared_root is not None:
                    persist_group(
                        self._selection_group(config),
                        prepared_root=prepared_root,
                    )
                if status == "completed" and not publish_terminal_outcome(
                    entry.id
                ):
                    break
                if status == "waiting_for_selection":
                    # A non-interactive review checkpoint is intentional and
                    # should not start later method jobs with unresolved choices.
                    break
            except KeyboardInterrupt:
                orchestrator.mark_interrupted()
                results[entry.id].update(
                    {"status": "interrupted", "finished_at": _now()}
                )
                save_state()
                raise
            except Exception as exc:
                results[entry.id].update(
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if orchestrator.run_directory is not None:
                    results[entry.id]["result"] = str(
                        orchestrator.run_directory
                    )
                save_state()
                self.console.print(
                    f"[red]Queue entry failed: {entry.id}: {exc}[/red]"
                )
                if orchestrator.run_directory is not None:
                    prepared_root = self._prepared_root_from_run(
                        orchestrator.run_directory
                    )
                    if prepared_root is not None:
                        persist_group(
                            self._selection_group(config),
                            prepared_root=prepared_root,
                        )
                if not publish_terminal_outcome(entry.id):
                    break
                if not queue.continue_independent:
                    break
        if not dry_run:
            self._run_common_evaluations(queue, results, configs)
            results = publish_queue_transaction(
                transaction_root,
                queue_id=queue.id,
                configs=configs,
                results=results,
            )
            if close_terminal_transaction():
                _print_queue_completion(
                    self.console,
                    configs[0],
                    results,
                    elapsed_seconds=time.monotonic() - queue_started,
                )
                return results
        save_state()
        return results

    def _run_common_evaluations(
        self,
        queue: QueueConfig,
        results: dict[str, dict[str, Any]],
        requested_configs: list[RigConfig],
    ) -> None:
        """Evaluate each experiment with one anchor shared by every method.

        This stage never re-runs a calibration method. Candidate anchors are
        tried in deterministic observation-quality order; unsuccessful
        evaluation artifacts remain visible for diagnosis.
        """
        groups: dict[
            tuple[str, str],
            list[tuple[Path, dict[str, Any], RigConfig]],
        ] = {}
        for entry, requested_config in zip(
            queue.entries, requested_configs, strict=True
        ):
            row = results.get(entry.id, {})
            if row.get("status") not in {
                "completed",
                "published",
                "duplicate_skipped",
            }:
                continue
            result_path = Path(str(row.get("result", "")))
            manifest_path = (
                result_path / "provenance" / "run_manifest.json"
            )
            config_path = (
                result_path / "provenance" / "resolved_config.yaml"
            )
            if not manifest_path.is_file() or not config_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_root = str(experiment_paths(requested_config).root)
            input_id = str(manifest.get("input_id", ""))
            if not experiment_root or not input_id:
                continue
            groups.setdefault((experiment_root, input_id), []).append(
                (result_path, manifest, requested_config)
            )

        label_by_method = {
            "ap01": "AP01",
            "ap02": "AP02",
            "ap03": "AP03_MULTI",
        }
        directory_by_method = {
            "ap01": "diagnostics/method",
            "ap02": "diagnostics/method",
            "ap03": "diagnostics/method/scale_multi",
        }
        for (experiment_text, input_id), group in groups.items():
            experiment = Path(experiment_text)
            first_path, first_manifest, first_config = group[0]
            enabled_group = [
                item for item in group if item[2].evaluation.enabled
            ]
            if not enabled_group:
                continue
            group = enabled_group
            first_path, first_manifest, first_config = group[0]
            transaction_dataset = (
                queue_temporary_root(first_config, queue.id) / "dataset"
            )
            dataset_root = (
                transaction_dataset
                if transaction_dataset.is_dir()
                else experiment_paths(first_config).dataset_root
            )
            observations = dataset_root / "observations"
            candidate_path = observations / "SELECTION_CANDIDATES.json"
            if not candidate_path.is_file():
                unavailable = (
                    queue_temporary_root(first_config, queue.id)
                    / "results"
                    / "evaluations"
                    / "COMMON_EVALUATION_UNAVAILABLE.json"
                )
                _write_json(
                    unavailable,
                    {
                        "schema_version": 5,
                        "layout_version": 2,
                        "status": "unavailable",
                        "reason": (
                            "The complete dataset has no "
                            "SELECTION_CANDIDATES.json; common evaluation was "
                            "not silently skipped."
                        ),
                        "dataset_root": str(dataset_root),
                    },
                )
                self.console.print(
                    "[yellow]Common evaluation unavailable: the complete "
                    "selection-candidate evidence is missing.[/yellow]"
                )
                continue
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            eligible = set(
                int(value)
                for value in payload["evaluation_anchor"][
                    "observation_candidates"
                ]
            )
            ranked = [
                item
                for item in payload["ap03_single_scale_marker"]["candidates"]
                if int(item["id"]) in eligible
            ]
            ranked.sort(
                key=lambda item: (
                    ap03_candidate_rank(item),
                    -int(item["id"]),
                ),
                reverse=True,
            )
            explicit_anchors = {
                int(config.evaluation.anchor_marker_id)
                for _, _, config in group
                if config.evaluation.anchor_marker_id != "auto_common"
            }
            if len(explicit_anchors) > 1:
                final = (
                    experiment
                    / "evaluations"
                    / "COMMON_EVALUATION_UNAVAILABLE.json"
                )
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_text(
                    json.dumps(
                        {
                            "status": "unavailable",
                            "reason": (
                                "Compared queue entries request conflicting "
                                "explicit evaluation anchors"
                            ),
                            "configured_anchor_marker_ids": sorted(
                                explicit_anchors
                            ),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue
            if explicit_anchors:
                requested_anchor = next(iter(explicit_anchors))
                ranked = [
                    item
                    for item in ranked
                    if int(item["id"]) == requested_anchor
                ]
            methods: list[tuple[str, Path]] = []
            for result_path, manifest, _ in group:
                method_id = str(manifest.get("method_id", ""))
                if method_id not in directory_by_method:
                    continue
                if method_id == "ap02":
                    status_path = (
                        result_path
                        / "diagnostics"
                        / "method"
                        / "METHOD_STATUS.json"
                    )
                    if status_path.is_file():
                        try:
                            method_status = json.loads(
                                status_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            method_status = {}
                        if not method_status.get(
                            "comparison_eligible", True
                        ):
                            self.console.print(
                                "[yellow]AP02 diagnostic partial result is "
                                "excluded from common primary-method "
                                "evaluation.[/yellow]"
                            )
                            continue
                variant = result_path.name
                methods.append(
                    (
                        f"{label_by_method[method_id]}__{variant}",
                        result_path / directory_by_method[method_id],
                    )
                )
            if not methods or not ranked:
                continue

            selection: dict[str, Any] | None = None
            for item in ranked:
                anchor = int(item["id"])
                eval_sha = evaluation_fingerprint(first_config, anchor)[:8]
                method_identity = [
                    {
                        "method_id": manifest.get("method_id"),
                        "variant": manifest.get("variant"),
                        "method_fingerprint": manifest.get(
                            "method_fingerprint"
                        ),
                    }
                    for _, manifest, _ in group
                ]
                job_fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "evaluation": first_config.evaluation.model_dump(
                                mode="json"
                            ),
                            "anchor": anchor,
                            "methods": method_identity,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                transaction_evaluations = (
                    queue_temporary_root(first_config, queue.id)
                    / "results"
                    / "evaluations"
                )
                output = (
                    transaction_evaluations
                    / f"anchor_marker_{anchor}_{eval_sha}"
                )
                previous_status = output / "COMMON_ANCHOR_STATUS.json"
                if previous_status.is_file() and not any(
                    config.project.duplicate_policy == "force"
                    for _, _, config in group
                ):
                    previous = json.loads(
                        previous_status.read_text(encoding="utf-8")
                    )
                    if (
                        previous.get("success_for_every_method")
                        and previous.get("evaluation_job_fingerprint")
                        == job_fingerprint
                    ):
                        selection = previous
                        self.console.print(
                            f"[dim]Exact common evaluation already exists; "
                            f"skipped: {output}[/dim]"
                        )
                        break
                argv = [
                    sys.executable,
                    str(
                        self.repository_root
                        / "src/camera_rig_calibration/evaluation/marker_consistency.py"
                    ),
                    "--dataset",
                    str(dataset_root),
                    "--results-root",
                    str(experiment),
                    "--observations-root",
                    str(observations),
                    "--output-root",
                    str(output),
                    "--anchor-marker-id",
                    str(anchor),
                    "--marker-length-m",
                    str(first_config.markers.length_m),
                    "--reprojection-threshold-px",
                    str(first_config.evaluation.reprojection_threshold_px),
                    "--min-inliers",
                    str(first_config.evaluation.minimum_inliers),
                    "--ransac-iters",
                    str(first_config.evaluation.ransac_iterations),
                    "--min-triangulation-angle-deg",
                    str(
                        first_config.evaluation.minimum_triangulation_angle_deg
                    ),
                    "--max-moving-observations-per-marker",
                    str(
                        first_config.evaluation.maximum_moving_observations_per_marker
                    ),
                    "--cameras",
                    ",".join(
                        camera.id for camera in first_config.static_cameras
                    ),
                ]
                for label, directory in methods:
                    argv += ["--method", f"{label}={directory.resolve()}"]
                self.console.print(
                    f"[cyan]Common evaluation candidate marker {anchor} "
                    f"for {len(methods)} methods[/cyan]"
                )
                started = time.monotonic()
                completed = subprocess.run(
                    argv,
                    cwd=self.repository_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                runtime_seconds = time.monotonic() - started
                self.console.print(
                    f"[cyan]Common evaluation finished in "
                    f"{runtime_seconds:.1f} s[/cyan]"
                )
                (output / "evaluation.log").parent.mkdir(
                    parents=True, exist_ok=True
                )
                (output / "evaluation.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                summary_path = (
                    output
                    / "marker_consistency"
                    / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json"
                )
                summaries = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.is_file()
                    else []
                )
                success = (
                    completed.returncode == 0
                    and len(summaries) == len(methods)
                    and all(
                        not str(row.get("status", "")).startswith(
                            "NOT_AVAILABLE"
                        )
                        for row in summaries
                    )
                )
                _status = {
                    "anchor_marker_id": anchor,
                    "success_for_every_method": success,
                    "evaluation_job_fingerprint": job_fingerprint,
                    "runtime_seconds": runtime_seconds,
                    "method_statuses": {
                        str(row.get("method")): str(row.get("status"))
                        for row in summaries
                    },
                    "output": str(output),
                }
                (output / "COMMON_ANCHOR_STATUS.json").write_text(
                    json.dumps(_status, indent=2) + "\n",
                    encoding="utf-8",
                )
                if success:
                    selection = _status
                    comparison = output / "comparison"
                    comparison.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(
                        summary_path,
                        comparison
                        / "COMMON_METHOD_EVALUATION_SUMMARY.json",
                    )
                    support_path = (
                        output
                        / "marker_consistency"
                        / "COMMON_SUPPORT_REPORT.json"
                    )
                    if support_path.is_file():
                        shutil.copy2(
                            support_path,
                            comparison / "COMMON_SUPPORT_REPORT.json",
                        )
                    # AP03 Single is a diagnostic scale on the same COLMAP
                    # reconstruction. Evaluate it separately with the selected
                    # common anchor, but never let this diagnostic replace or
                    # block AP03 Multi in the common method comparison.
                    for result_path, manifest, config in group:
                        if manifest.get("method_id") != "ap03":
                            continue
                        single_output = (
                            output
                            / "diagnostics"
                            / f"ap03_single_{result_path.name}"
                        )
                        single_argv = [
                            sys.executable,
                            str(
                                self.repository_root
                                / "src/camera_rig_calibration/evaluation/"
                                "marker_consistency.py"
                            ),
                            "--dataset",
                            str(dataset_root),
                            "--results-root",
                            str(result_path),
                            "--observations-root",
                            str(observations),
                            "--output-root",
                            str(single_output),
                            "--anchor-marker-id",
                            str(anchor),
                            "--marker-length-m",
                            str(config.markers.length_m),
                            "--reprojection-threshold-px",
                            str(config.evaluation.reprojection_threshold_px),
                            "--min-inliers",
                            str(config.evaluation.minimum_inliers),
                            "--ransac-iters",
                            str(config.evaluation.ransac_iterations),
                            "--min-triangulation-angle-deg",
                            str(
                                config.evaluation.minimum_triangulation_angle_deg
                            ),
                            "--max-moving-observations-per-marker",
                            str(
                                config.evaluation.maximum_moving_observations_per_marker
                            ),
                            "--cameras",
                            ",".join(
                                camera.id for camera in config.static_cameras
                            ),
                            "--method",
                            (
                                "AP03_SINGLE="
                                + str(
                                    (
                                        result_path
                                        / "diagnostics"
                                        / "method"
                                        / "scale_single"
                                    ).resolve()
                                )
                            ),
                        ]
                        diagnostic = subprocess.run(
                            single_argv,
                            cwd=self.repository_root,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        single_output.mkdir(parents=True, exist_ok=True)
                        (single_output / "evaluation.log").write_text(
                            diagnostic.stdout, encoding="utf-8"
                        )
                        _write_json(
                            single_output / "DIAGNOSTIC_STATUS.json",
                            {
                                "role": "diagnostic",
                                "method": "AP03_SINGLE",
                                "anchor_marker_id": anchor,
                                "returncode": diagnostic.returncode,
                                "success": diagnostic.returncode == 0,
                            },
                        )
                    break
            final = (
                queue_temporary_root(first_config, queue.id)
                / "results"
                / "evaluations"
                / (
                "SELECTED_COMMON_EVALUATION.json"
                if selection is not None
                else "COMMON_EVALUATION_UNAVAILABLE.json"
                )
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_text(
                json.dumps(
                    selection
                    or {
                        "status": "unavailable",
                        "reason": (
                            "No single observation candidate was reconstructable "
                            "for every completed method. No per-method anchor "
                            "substitution was performed."
                        ),
                        "candidate_marker_ids": [
                            int(item["id"]) for item in ranked
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
