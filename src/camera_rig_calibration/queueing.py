from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import time
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
from .experiments import evaluation_fingerprint
from .observations import (
    ResolvedSelections,
    ap03_candidate_rank,
    freeze_selections,
)
from .runtime import PipelineOrchestrator
from .preflight import PreflightJob, run_queue_preflight
from .publication import publish_queue_transaction
from .storage_layout import queue_temporary_root


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
    def migrate_legacy_queue(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("schema_version", 1) in {1, 2, 3, 4, 5}:
            payload["schema_version"] = 5
        return payload

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


def is_queue_config(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and (
        payload.get("kind") == "rigcal_queue"
        or isinstance(payload.get("entries"), list)
        or isinstance(payload.get("runs"), list)
        and "dataset" not in payload
    )


def _load_queue_unpartitioned(path: Path) -> QueueConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Queue root must be a mapping: {source}")
    if "entries" not in payload and isinstance(payload.get("runs"), list):
        payload = {
            "kind": "rigcal_queue",
            "schema_version": 5,
            "id": str(payload.get("queue_id") or payload.get("dataset_id") or source.stem),
            "continue_independent": True,
            "entries": [
                {
                    "id": str(
                        item.get("id")
                        or item.get("label")
                        or f"entry_{index:02d}"
                    ),
                    "config": item["config"],
                    "depends_on": item.get("depends_on", []),
                }
                for index, item in enumerate(payload["runs"], 1)
            ],
        }
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
    # Schema-v2 stored AP03 single and multi as separate rows. Once both files
    # migrate to an identical combined AP03 snapshot, keep exactly one job.
    raw_entries = list(payload.get("entries", []))
    split_indices = {
        index
        for index, item in enumerate(raw_entries)
        if _is_v2_split_ap03_path(
            (
                Path(item["config"])
                if Path(item["config"]).is_absolute()
                else source.parent / Path(item["config"])
            )
        )
    }
    kept: list[QueueEntry] = []
    seen_ap03: set[str] = set()
    for index, entry in enumerate(resolved_queue.entries):
        if index not in split_indices:
            kept.append(entry)
            continue
        config = load_config(entry.config)
        payload_for_hash = {
            "dataset": config.dataset.model_dump(mode="json"),
            "static_cameras": [
                item.model_dump(mode="json") for item in config.static_cameras
            ],
            "moving_camera": config.moving_camera.model_dump(mode="json"),
            "markers": config.markers.model_dump(mode="json"),
            "observation_quality": config.observation_quality.model_dump(mode="json"),
            "colmap": config.colmap.model_dump(mode="json"),
            "ap03": config.methods.ap03.model_dump(mode="json"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload_for_hash, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_ap03:
            continue
        seen_ap03.add(fingerprint)
        kept.append(entry)
    return resolved_queue.model_copy(update={"entries": kept}, deep=True)


def load_queue_partitions(path: Path) -> tuple[QueueConfig, ...]:
    """Load a queue and partition legacy multi-dataset rows in input order."""
    queue = _load_queue_unpartitioned(path)
    groups: dict[str, list[QueueEntry]] = {}
    configs: dict[str, RigConfig] = {}
    for entry in queue.entries:
        config = load_config(entry.config)
        dataset_id = config.dataset.id
        groups.setdefault(dataset_id, []).append(entry)
        configs.setdefault(dataset_id, config)
    partitions: list[QueueConfig] = []
    for index, (dataset_id, entries) in enumerate(groups.items(), 1):
        config = configs[dataset_id]
        partitions.append(
            QueueConfig(
                id=(
                    queue.id
                    if len(groups) == 1
                    else f"{queue.id}__{index:02d}_{dataset_id}"
                ),
                continue_independent=queue.continue_independent,
                common=QueueCommon(
                    dataset=config.dataset,
                    aruco=config.markers,
                    evaluation=config.evaluation,
                ),
                entries=entries,
            )
        )
    return tuple(partitions)


def load_queue(path: Path) -> QueueConfig:
    partitions = load_queue_partitions(path)
    if not partitions:
        raise ValueError(f"Queue contains no entries: {path}")
    if len(partitions) > 1:
        ids = ", ".join(partition.id for partition in partitions)
        raise ValueError(
            "Legacy queue contains multiple datasets and was partitioned into "
            f"ordered dataset subqueues: {ids}. Use load_queue_partitions() or "
            "run the queue through the rigcal CLI."
        )
    return partitions[0]


def _is_v2_split_ap03_path(path: Path) -> bool:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        methods = payload.get("methods", {})
        enabled = methods.get("enabled", [])
        return (
            int(payload.get("schema_version", 1)) <= 2
            and any(value in {"ap03_single", "ap03_multi"} for value in enabled)
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return False


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
        selection_reviewer: Callable[
            [RigConfig, ResolvedSelections, Path], dict[str, Any]
        ]
        | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.console = console or Console()
        self.selection_reviewer = selection_reviewer

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
        dataset_ids: set[str] = set()
        for entry in queue.entries:
            config = load_config(entry.config)
            dataset_ids.add(config.dataset.id)
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
        if len(dataset_ids) > 1:
            raise RuntimeError(
                "A schema-v5 queue contains one dataset. Partition this legacy "
                f"queue before execution; found: {sorted(dataset_ids)}"
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

    def run(
        self, queue: QueueConfig, *, dry_run: bool = False
    ) -> dict[str, dict[str, Any]]:
        configs = self.validate(queue)
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
                    updated = updated.model_copy(
                        update={
                            "dataset": updated.dataset.model_copy(
                                update={"prepared_root": prepared_root}
                            )
                        },
                        deep=True,
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
            table = Table(title="Queue preflight")
            table.add_column("Job")
            table.add_column("Method")
            table.add_column("Status")
            table.add_column("Accepted", justify="right")
            table.add_column("Readiness details", overflow="fold")
            for entry, config, report in zip(
                queue.entries, configs, preflight_result.jobs, strict=True
            ):
                table.add_row(
                    entry.id,
                    config.methods.enabled[0],
                    report.status,
                    (
                        str(report.filter_result.accepted_count)
                        if report.filter_result is not None
                        else "0"
                    ),
                    "; ".join(
                        (*report.details, *report.warnings, *report.errors)
                    )
                    or "-",
                )
            self.console.print(table)
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
                        "preflight": str(report.output_directory),
                        "errors": list(report.errors),
                        "warnings": list(report.warnings),
                    }
            if not preflight_result.ready:
                save_state()
                self.console.print(
                    "[red]No calibration method is runnable. Failed jobs remain "
                    "available with their individual preflight reports.[/red]"
                )
                return results
            failed_count = sum(
                1 for report in preflight_result.jobs if not report.runnable
            )
            if failed_count:
                self.console.print(
                    f"[yellow]{failed_count} queue job(s) failed preflight and "
                    "will be skipped; independent runnable jobs continue.[/yellow]"
                )

            overrides: dict[str, Any] | None = None
            review_jobs = [
                (config, report)
                for config, report in zip(
                    configs, preflight_result.jobs, strict=True
                )
                if report.runnable
                and config.selection.mode == "review_once"
                and not (
                    config.methods.ap01.root_camera == "auto"
                    and config.methods.ap02.reference_marker_id == "auto"
                    and config.methods.ap03_single.scale_marker_id == "auto"
                    and config.methods.ap03_multi.marker_ids == "auto"
                    and config.evaluation.anchor_marker_id == "auto_common"
                )
            ]
            if review_jobs:
                if self.selection_reviewer is None:
                    for entry in queue.entries:
                        results[entry.id] = {
                            "status": "waiting_for_selection",
                            "preflight": str(
                                resolved_root / "preflight"
                            ),
                        }
                    save_state()
                    return results
                review_config, review_report = review_jobs[0]
                review_config = review_config.model_copy(
                    update={
                        "methods": review_config.methods.model_copy(
                            update={
                                "enabled": list(
                                    dict.fromkeys(
                                        method_id
                                        for candidate, _ in review_jobs
                                        for method_id in candidate.methods.enabled
                                    )
                                )
                            },
                            deep=True,
                        )
                    },
                    deep=True,
                )
                assert review_report.selections is not None
                overrides = self.selection_reviewer(
                    review_config,
                    review_report.selections,
                    resolved_root / "preflight",
                )

            selection_errors: dict[str, str] = {}
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
                updated = config.model_copy(
                    update={
                        "dataset": config.dataset.model_copy(
                            update={"prepared_root": prepared_root}
                        )
                    },
                    deep=True,
                )
                try:
                    updated = freeze_selections(
                        updated, report.selections, overrides
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
            if selection_errors:
                for entry in queue.entries:
                    own_error = selection_errors.get(entry.id)
                    if own_error is None:
                        continue
                    results[entry.id] = {
                        "status": "failed_preflight",
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
            save_resolved_queue()
            save_state()

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
                previous.get("status")
                in {"completed", "duplicate_skipped"}
                and (previous_result / "run_manifest.json").is_file()
            ):
                self.console.print(
                    f"[dim]QUEUE {index}/{len(queue.entries)} — "
                    f"{entry.id}: already completed; skipped[/dim]"
                )
                continue
            failed_dependencies = [
                dependency
                for dependency in entry.depends_on
                if results.get(dependency, {}).get("status")
                not in {"completed", "duplicate_skipped"}
            ]
            if failed_dependencies:
                results[entry.id] = {
                    "status": "skipped_dependency",
                    "dependencies": failed_dependencies,
                }
                save_state()
                continue
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
                    ):
                        source = queue_job_preflight / name
                        if source.is_file():
                            shutil.copy2(source, snapshot / name)
                status = str(manifest.get("status", "completed"))
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
            if all(
                row.get("status") in {"completed", "duplicate_skipped"}
                for row in results.values()
            ):
                receipt = (
                    configs[0].project.workspace_root.resolve()
                    / "queues"
                    / f"{queue.id}.published.json"
                )
                _write_json(
                    receipt,
                    {
                        "schema_version": 5,
                        "queue_id": queue.id,
                        "status": "published",
                        "published_at": _now(),
                        "entries": results,
                    },
                )
                shutil.rmtree(transaction_root)
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
            if row.get("status") not in {"completed", "duplicate_skipped"}:
                continue
            result_path = Path(str(row.get("result", "")))
            manifest_path = result_path / "run_manifest.json"
            config_path = result_path / "resolved_config.yaml"
            if not manifest_path.is_file() or not config_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_root = str(manifest.get("experiment_root", ""))
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
            "ap01": "02_AP01",
            "ap02": "03_AP02",
            "ap03": "04_AP03/scale_multi",
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
            observations = Path(str(first_manifest["observations_root"]))
            candidate_path = observations / "SELECTION_CANDIDATES.json"
            pointer_path = first_path / "00_INPUT" / "dataset_pointer.json"
            if not candidate_path.is_file() or not pointer_path.is_file():
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
            dataset_root = Path(
                json.loads(pointer_path.read_text(encoding="utf-8"))[
                    "dataset_root"
                ]
            )
            methods: list[tuple[str, Path]] = []
            for result_path, manifest, _ in group:
                method_id = str(manifest.get("method_id", ""))
                if method_id not in directory_by_method:
                    continue
                if method_id == "ap02":
                    status_path = result_path / "03_AP02/METHOD_STATUS.json"
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
                variant = str(manifest.get("variant", "baseline"))
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
                output = (
                    experiment
                    / "evaluations"
                    / f"anchor_marker_{anchor}_{eval_sha}"
                    / f"queue_{queue.id}_{input_id}"
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
                        / "run/real_vehicle_data/12_evaluate_real_marker_consistency.py"
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
                    comparison = (
                        experiment
                        / "comparisons"
                        / f"{queue.id}_{input_id}"
                        / f"anchor_marker_{anchor}_{eval_sha}"
                    )
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
                            result_path / "04_AP03" / "evaluation_single"
                        )
                        single_argv = [
                            sys.executable,
                            str(
                                self.repository_root
                                / "run/real_vehicle_data/"
                                "12_evaluate_real_marker_consistency.py"
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
                                        / "04_AP03"
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
            final = experiment / "evaluations" / (
                "SELECTED_COMMON_EVALUATION.json"
                if selection is not None
                else "COMMON_EVALUATION_UNAVAILABLE.json"
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
