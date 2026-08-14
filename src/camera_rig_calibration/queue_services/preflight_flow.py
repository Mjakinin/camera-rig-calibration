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
    SelectionReviewJob,
    ObservationReviewDecision,
    _now,
    _write_json,
    _freeze_queue_preflight_dataset_evidence,
    _method_selection_summary,
    _configured_selection_summary,
    _method_preflight_coverage,
    _selection_source,
    _bind_prepared_dataset,
)
from .models import (
    QueueConfig,
)
from .bindings import current_queue_bindings
from .state import QueuePreflightState


class QueuePreflightFlowMixin:
    def _prepare_queue_for_execution(
        self,
        *,
        queue: QueueConfig,
        configs: list[RigConfig],
        results: dict[str, dict[str, Any]],
        source_fingerprints: dict[str, str],
        transaction_root: Path,
        resolved_root: Path,
        resolved_configs: dict[str, str],
        preflight_state: QueuePreflightState,
        save_state,
        save_resolved_queue,
        close_terminal_transaction,
        queue_started: float,
        batch_started_monotonic: float | None,
        dry_run: bool,
    ) -> dict[str, dict[str, Any]] | None:
        bindings = current_queue_bindings()
        PipelineOrchestrator = bindings.pipeline_orchestrator
        freeze_selections = bindings.freeze_selections
        _method_preflight_coverage = bindings.method_preflight_coverage
        publish_preparation_transaction = (
            bindings.publish_preparation_transaction
        )
        publish_queue_transaction = bindings.publish_queue_transaction
        run_queue_preflight = bindings.run_queue_preflight
        preparation_path = Path(preflight_state.preparation)
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
                        # Queue preparation must retain every observation
                        # that any later method override could accept.
                        # Scientific filtering is performed per queue row.
                        "observation_quality": ObservationQualitySettings(
                            minimum_marker_area_ratio=0.0,
                            maximum_pnp_reprojection_error_px="disabled",
                            require_positive_depth=False,
                            maximum_marker_distance_m="disabled",
                        ),
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
            preflight_state.preparation = str(preparation_path)
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
        # Freeze the immutable acquisition identity once for the whole
        # queue.  Method-resolved configs may differ, but every row inherits
        # this exact content contract.
        queue_dataset_identity = build_dataset_identity(prepared_root)
        if not queue_dataset_identity.get("content_files"):
            raise RuntimeError(
                "The prepared dataset has no hashable raw images or "
                "camera-info files; its queue identity cannot be frozen."
            )
        _write_json(
            transaction_root / "queue_dataset_identity.json",
            {
                **queue_dataset_identity,
                "queue_id": queue.id,
                "prepared_root": str(prepared_root.resolve()),
                "scope": "queue_shared_immutable_dataset",
            },
        )
        recovered_retry = self._recover_interrupted_detector_retry(
            transaction_root=transaction_root,
            resolved_root=resolved_root,
            configs=configs,
        )
        if recovered_retry is not None:
            configs, previous_mode, recovered_mode = recovered_retry
            attempted_modes = list(
                preflight_state.observation_review.get("attempted_modes", [])
            )
            for mode in (previous_mode, recovered_mode):
                if mode not in attempted_modes:
                    attempted_modes.append(mode)
            preflight_state.observation_review.update(
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
        raw_observations_csv = (
            raw_observations_root
            / "shared_all_aruco_observations.csv"
        )
        preflight_result = run_queue_preflight(
            (
                PreflightJob(entry.id, config)
                for entry, config in zip(
                    queue.entries, configs, strict=True
                )
            ),
            raw_observations_csv=raw_observations_csv,
            dataset_root=prepared_root,
            output_directory=resolved_root / "preflight",
            repository_root=self.repository_root,
        )
        _freeze_queue_preflight_dataset_evidence(
            transaction_root=transaction_root,
            resolved_root=resolved_root,
            configs=configs,
            preflight=preflight_result,
            raw_observations_csv=raw_observations_csv,
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
        preflight_state.reports = {
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
            and not preflight_state.coverage_override
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
                preflight_state.observation_review.get("attempted_modes", [])
            )
            current_mode = configs[0].markers.detection_mode
            if current_mode not in attempted_modes:
                attempted_modes.append(current_mode)
            preflight_state.observation_review.update(
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
            reviews = list(preflight_state.observation_review.get("reviews", []))
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
            preflight_state.observation_review["reviews"] = reviews
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
                preflight_state.observation_review.update(
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
                preflight_state.reports.clear()
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
                preflight_state.observation_review["status"] = "waiting"
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
            preflight_state.coverage_override = True
            preflight_state.observation_review.update(
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
            and (
                config.selection.mode == "review_once"
                or (
                    config.evaluation.enabled
                    and config.evaluation.anchor_selection_mode
                    == "review_once"
                )
            )
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
                        and preflight_state.reports.get(entry.id) is not None
                        and preflight_state.reports[entry.id].runnable
                        for entry in queue.entries
                    ),
                },
            )
            save_state()
            self.console.print(
                "[yellow]Incompatible selections failed only their own "
                "jobs; independent runnable jobs continue.[/yellow]"
            )
        self._publish_preflight_dataset(
            queue=queue,
            configs=configs,
            results=results,
            transaction_root=transaction_root,
            resolved_root=resolved_root,
            preflight_result=preflight_result,
            review_jobs=review_jobs,
            overrides_by_job=overrides_by_job,
            preparation_path=preparation_path,
            save_resolved_queue=save_resolved_queue,
            save_state=save_state,
        )


__all__ = ['QueuePreflightFlowMixin']
