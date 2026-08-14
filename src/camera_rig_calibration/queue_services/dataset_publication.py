"""Publication of immutable queue-preflight evidence."""

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

class QueueDatasetPublicationMixin:
    def _publish_preflight_dataset(
        self,
        *,
        queue: QueueConfig,
        configs: list[RigConfig],
        results: dict[str, dict[str, Any]],
        transaction_root: Path,
        resolved_root: Path,
        preflight_result,
        review_jobs: list[SelectionReviewJob],
        overrides_by_job: dict[str, dict[str, Any]],
        preparation_path: Path,
        save_resolved_queue,
        save_state,
    ) -> None:
        publish_preparation_transaction = (
            current_queue_bindings().publish_preparation_transaction
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
            "SELECTION_CANDIDATES.csv",
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
        final_anchor = (
            int(configs[0].evaluation.anchor_marker_id)
            if isinstance(configs[0].evaluation.anchor_marker_id, int)
            else None
        )
        selection_payload_path = (
            canonical_observations / "SELECTION_CANDIDATES.json"
        )
        selection_payload = json.loads(
            selection_payload_path.read_text(encoding="utf-8")
        )
        selection_payload["evaluation_anchor"].update(
            {
                "selected": final_anchor,
                "selection_mode": (
                    configs[0].evaluation.anchor_selection_mode
                ),
                "reason": (
                    "manual post-preflight selection from every raw "
                    "detected marker ID"
                    if review_jobs
                    and any(
                        "evaluation_anchor_marker_id" in values
                        for values in overrides_by_job.values()
                    )
                    else selection_payload["evaluation_anchor"].get(
                        "reason"
                    )
                ),
                "warning_confirmed": any(
                    bool(
                        values.get(
                            "evaluation_anchor_warning_confirmed"
                        )
                    )
                    for values in overrides_by_job.values()
                ),
            }
        )
        for name in (
            "SELECTION_CANDIDATES.json",
            "REFERENCE_SELECTIONS.json",
        ):
            _write_json(canonical_observations / name, selection_payload)
        write_selection_candidates_csv(
            canonical_observations, selection_payload
        )
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
                "evaluation_anchor_selection_mode": (
                    configs[0].evaluation.anchor_selection_mode
                ),
                "reviewed_once": bool(review_jobs),
                "evaluation_anchor_warning_confirmed": any(
                    bool(
                        values.get(
                            "evaluation_anchor_warning_confirmed"
                        )
                    )
                    for values in overrides_by_job.values()
                ),
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
                    "SELECTION_CANDIDATES.csv",
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


__all__ = ["QueueDatasetPublicationMixin"]
