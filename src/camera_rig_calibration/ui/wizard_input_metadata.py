"""Focused wizard responsibilities extracted from the compatibility facade."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_user_config
from ..config.models import (
    ColmapSettings,
    DatasetCategory,
    DatasetSettings,
    EvaluationSettings,
    McapSettings,
    MethodSettings,
    MarkerSettings,
    MovingCameraSettings,
    IntrinsicScanSettings,
    InputSourceKind,
    ObservationQualitySettings,
    ProjectSettings,
    RigConfig,
    SamplingSettings,
    SceneType,
    SelectionSettings,
    SimulationSettings,
    StaticCameraSettings,
    effective_observation_quality,
)
from ..dataset.discovery import (
    IMAGE_SUFFIXES,
    discover_image_directories,
    discover_inputs,
    inspect_prepared_dataset,
    media_path_role,
    safe_id,
)
from ..doctor import run_checks
from ..experiments import automatic_method_label
from ..input.topics import McapTopic, list_mcap_topics
from ..input.video_geometry import probe_video_geometry
from ..intrinsics_profiles import (
    IntrinsicProfile,
    discover_intrinsic_profiles,
    intrinsic_dimensions,
)
from ..inventory import (
    BASELINE_SIMULATION_PARAMETERS,
    PreparedDatasetSummary,
    RawInputSummary,
    SimulationExperimentSummary,
    discover_prepared_datasets,
    discover_raw_input_folders,
    discover_simulation_experiments,
    find_matching_simulation,
    format_simulation_parameters,
)
from ..registry import (
    calibration_methods,
    experiment_providers,
    input_adapters,
)
from ..runtime import PipelineOrchestrator
from ..observation_quality import filter_observations
from ..observations import ResolvedSelections, resolve_selections
from ..queueing import SelectionReviewJob, save_batch
# Compatibility hooks wrapped by the product policy stack. The concrete result
# browser lives under ui/, but these names remain stable until the wrappers are
# converted to explicit composition.
from ..publication import reconcile_existing_experiment
from ..visualization import launch_isolated_rviz





def _stored_prepared_marker_settings(prepared_root: Path) -> MarkerSettings:
    """Reuse the exact detector contract of a canonical prepared dataset."""

    path = prepared_root / "observations" / "detection_config.json"
    if not path.is_file():
        return MarkerSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        markers = payload["markers"]
        return MarkerSettings.model_validate(markers)
    except (OSError, json.JSONDecodeError, KeyError, ValidationError) as exc:
        raise RuntimeError(
            "Prepared dataset has an invalid ArUco detection contract: "
            f"{path}. Repair or regenerate its observations before reuse."
        ) from exc

def _stored_prepared_sampling(root: Path) -> tuple[float | None, str]:
    candidates = [
        root / "dataset_manifest.json",
        root / "metadata" / "moving_video_extraction" / "MOVING_VIDEO_EXTRACTION.json",
        root / "raw_images" / "metadata" / "moving_video_extraction" / "MOVING_VIDEO_EXTRACTION.json",
    ]
    keys = (
        "sampling_hz",
        "target_hz",
        "requested_hz",
        "output_hz",
        "moving_sampling_hz",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in keys:
            value = payload.get(key) if isinstance(payload, dict) else None
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number, "stored metadata"
    return None, "unknown"


__all__ = [
    '_stored_prepared_marker_settings',
    '_stored_prepared_sampling',
]
