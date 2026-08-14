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
            "observation_input_contract": (
                "raw_detection_with_dimensions_and_area_ratio_v2"
            ),
        },
    )


def _freeze_queue_preflight_dataset_evidence(
    *,
    transaction_root: Path,
    resolved_root: Path,
    configs: list[RigConfig],
    preflight: QueuePreflightResult,
    raw_observations_csv: Path,
) -> None:
    """Freeze queue-global evidence once without later method mutation."""

    observations = transaction_root / "dataset" / "observations"
    observations.mkdir(parents=True, exist_ok=True)
    baseline = filter_observations(
        raw_observations_csv,
        resolved_root / "preflight" / "global_baseline",
        job_id="queue_global_baseline",
        marker_settings=configs[0].markers,
        quality=configs[0].observation_quality,
    )
    quality = observations / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    for name in (
        "accepted_observations.csv",
        "rejected_observations.csv",
        "observation_filter_summary.json",
        "marker_inventory.csv",
        "marker_inventory.json",
    ):
        source = baseline.output_directory / name
        if source.is_file():
            shutil.copy2(source, quality / name)
    _write_json(
        quality / "preflight_summary.json",
        {
            "schema_version": 5,
            "status": (
                "READY"
                if baseline.accepted_count
                else "FAILED_PREFLIGHT"
            ),
            "scope": "queue_global_observation_quality_baseline",
            "effective_observation_quality": (
                configs[0].observation_quality.model_dump(mode="json")
            ),
            "observation_quality_sources": {
                field_name: "global"
                for field_name in configs[
                    0
                ].observation_quality.model_dump(mode="json")
            },
            "accepted_observations": baseline.accepted_count,
            "rejected_observations": baseline.rejected_count,
        },
    )

    runnable = next(
        (
            result
            for result in preflight.jobs
            if result.runnable
            and result.selections is not None
            and result.filter_result is not None
        ),
        None,
    )
    if runnable is not None:
        selection_root = (
            runnable.filter_result.filtered_observations_root
        )
        for name in (
            "SELECTION_CANDIDATES.json",
            "SELECTION_CANDIDATES.csv",
            "REFERENCE_SELECTIONS.json",
            "REFERENCE_MARKER_ID.txt",
        ):
            shutil.copy2(selection_root / name, observations / name)
    _write_json(
        observations / "QUEUE_SELECTIONS.json",
        {
            "schema_version": 5,
            "scope": "per_method_preflight",
            "common_evaluation_anchor_marker_id": (
                preflight.common_evaluation_anchor_marker_id
            ),
            "jobs": {
                result.job_id: result.selections.payload
                for result in preflight.jobs
                if result.selections is not None
            },
        },
    )
    completion = observations / "PUBLICATION_COMPLETE.json"
    payload = _read_json(completion)
    payload.update(
        {
            "schema_version": 5,
            "layout_version": 2,
            "status": "complete",
            "quality_scope": "queue_global_baseline",
            "method_quality_evidence": (
                "stored with each method diagnostic"
            ),
            "selection_scope": "per_method_preflight",
            "queue_selections": "QUEUE_SELECTIONS.json",
            "common_evaluation_anchor_marker_id": (
                preflight.common_evaluation_anchor_marker_id
            ),
            "frozen_before_methods": True,
        }
    )
    _write_json(completion, payload)


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
    observation_quality: ObservationQualitySettings = Field(
        default_factory=ObservationQualitySettings
    )
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)



__all__ = [
    'SelectionReviewJob',
    'QueueSelectionReviewer',
    'ObservationReviewDecision',
    'QueueObservationReviewer',
    '_now',
    '_write_json',
    '_read_json',
    '_base_experiment_id',
    '_config_with_detection_mode',
    '_write_observation_detection_config',
    '_freeze_queue_preflight_dataset_evidence',
    '_queue_job_fingerprint',
    '_format_runtime',
    '_method_selection_summary',
    '_configured_selection_summary',
    '_method_preflight_coverage',
    '_selection_source',
    '_method_result_summary',
    '_bind_prepared_dataset',
    '_print_queue_completion',
    'QueueEntry',
    'QueueCommon',
]
