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
from .dataset_identity import build_dataset_identity
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
    freeze_selections,
    write_selection_candidates_csv,
)
from .runtime import PipelineOrchestrator, observation_id
from .preflight import (
    PreflightJob,
    QueuePreflightResult,
    run_queue_preflight,
)
from .observation_quality import filter_observations
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


from .queue_services.common import (
    SelectionReviewJob,
    QueueSelectionReviewer,
    ObservationReviewDecision,
    QueueObservationReviewer,
    _now,
    _write_json,
    _read_json,
    _base_experiment_id,
    _config_with_detection_mode,
    _write_observation_detection_config,
    _freeze_queue_preflight_dataset_evidence,
    _queue_job_fingerprint,
    _format_runtime,
    _method_selection_summary,
    _configured_selection_summary,
    _method_preflight_coverage,
    _selection_source,
    _method_result_summary,
    _bind_prepared_dataset,
    _print_queue_completion,
    QueueEntry,
    QueueCommon,
)
from .queue_services.models import (
    QueueConfig,
    BatchEntry,
    BatchConfig,
    is_queue_config,
    is_batch_config,
    load_batch,
    save_batch,
    _load_queue_unpartitioned,
    load_queue_partitions,
    load_queue,
    save_queue,
)
from .queue_services.base import QueueRunnerBaseMixin
from .queue_services.runner import QueueRunnerMixin
from .queue_services.preflight_flow import QueuePreflightFlowMixin
from .queue_services.dataset_publication import QueueDatasetPublicationMixin
from .queue_services.evaluation import QueueEvaluationMixin


class QueueRunner(
    QueueRunnerBaseMixin,
    QueueRunnerMixin,
    QueuePreflightFlowMixin,
    QueueDatasetPublicationMixin,
    QueueEvaluationMixin,
):
    """Public queue facade; implementation phases live in queue_services."""


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
    'QueueRunner',
]
