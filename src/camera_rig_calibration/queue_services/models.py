"""Queue schemas and YAML serialization."""

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
    QueueEntry,
    QueueCommon,
)


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
    common = queue.common
    if common is not None:
        dataset_updates: dict[str, Path] = {}
        for field_name in ("prepared_root", "input_root"):
            configured = getattr(common.dataset, field_name)
            if configured is None:
                continue
            dataset_updates[field_name] = (
                configured.resolve()
                if configured.is_absolute()
                else (source.parent / configured).resolve()
            )
        if dataset_updates:
            common = common.model_copy(
                update={
                    "dataset": common.dataset.model_copy(
                        update=dataset_updates
                    )
                },
                deep=True,
            )
    resolved_queue = queue.model_copy(
        update={
            "common": common,
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
            "observation_quality": first.observation_quality.model_dump(
                mode="json", exclude_none=True
            ),
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



__all__ = [
    'QueueConfig',
    'BatchEntry',
    'BatchConfig',
    'is_queue_config',
    'is_batch_config',
    'load_batch',
    'save_batch',
    '_load_queue_unpartitioned',
    'load_queue_partitions',
    'load_queue',
    'save_queue',
]
