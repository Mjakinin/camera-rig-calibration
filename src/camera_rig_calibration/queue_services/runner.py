"""Queue transaction phase extracted from the public runner."""

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
    _now,
    _write_json,
    _queue_job_fingerprint,
    _method_result_summary,
    _bind_prepared_dataset,
    _print_queue_completion,
)
from .models import (
    QueueConfig,
)
from .bindings import current_queue_bindings
from .state import QueuePreflightState


class QueueRunnerMixin:
    def run(
        self,
        queue: QueueConfig,
        *,
        dry_run: bool = False,
        batch_started_monotonic: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        bindings = current_queue_bindings()
        PipelineOrchestrator = bindings.pipeline_orchestrator
        publish_preparation_transaction = (
            bindings.publish_preparation_transaction
        )
        publish_queue_transaction = bindings.publish_queue_transaction
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
                reuse_intermediates_from=(
                    self.reuse_method_intermediates.get(entry.id)
                ),
                rerun_metadata=self.rerun_metadata.get(entry.id),
                explicit_method_rerun=self.explicit_method_rerun,
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
        preflight_state = QueuePreflightState(
            preparation=str(
                previous_state.get("preflight_preparation", "")
            ),
            coverage_override=bool(
                previous_state.get("observation_coverage_override", False)
            ),
            observation_review=dict(
                previous_state.get("observation_review") or {}
            ),
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
                        "preflight_preparation": preflight_state.preparation or None,
                        "observation_coverage_override": (
                            preflight_state.coverage_override
                        ),
                        "observation_review": preflight_state.observation_review or None,
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
                    "observation_quality": configs[
                        0
                    ].observation_quality.model_dump(
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
            early_result = self._prepare_queue_for_execution(
                queue=queue,
                configs=configs,
                results=results,
                source_fingerprints=source_fingerprints,
                transaction_root=transaction_root,
                resolved_root=resolved_root,
                resolved_configs=resolved_configs,
                preflight_state=preflight_state,
                save_state=save_state,
                save_resolved_queue=save_resolved_queue,
                close_terminal_transaction=close_terminal_transaction,
                queue_started=queue_started,
                batch_started_monotonic=batch_started_monotonic,
                dry_run=dry_run,
            )
            if early_result is not None:
                return early_result
        seen_jobs: dict[str, str] = {}
        for index, (entry, config) in enumerate(
            zip(queue.entries, configs, strict=True), 1
        ):
            report = preflight_state.reports.get(entry.id)
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
                reuse_intermediates_from=(
                    self.reuse_method_intermediates.get(entry.id)
                ),
                rerun_metadata=self.rerun_metadata.get(entry.id),
                explicit_method_rerun=self.explicit_method_rerun,
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
                        "marker_inventory.csv",
                        "marker_inventory.json",
                        "ap02_frame_selection.csv",
                        "ap02_frame_selection.json",
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


__all__ = ['QueueRunnerMixin']
