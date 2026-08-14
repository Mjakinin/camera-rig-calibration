from __future__ import annotations

import json
import hashlib
import importlib.util
import csv
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_config
from ..config.models import RigConfig, effective_observation_quality
from ..contracts import CommandSpec, RunContext
from ..dataset.manifest import AutoSelection, load_dataset_manifest, save_dataset_manifest
from ..dataset.validation import validate_dataset
from ..input.preparation import build_preparation_plan, finalize_dataset
from ..input.topics import resolve_rosbag_source
from ..intrinsics_profiles import resolve_intrinsic_profile
from ..methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from ..experiments import (
    colmap_artifact_fingerprint,
    evaluation_fingerprint,
    experiment_paths,
    input_fingerprint,
    method_config_diff,
    method_fingerprint,
    method_result_label,
    write_experiment_manifest,
)
from ..observations import (
    ResolvedSelections,
    freeze_selections,
    resolve_selections,
)
from ..observation_quality import ObservationQualityError, filter_observations
from ..progress import ProgressClock, progress_text, terminal_lines
from ..pipeline import StageContract, validate_stage_dag
from ..registry import calibration_methods, evaluators, input_adapters
from ..results import write_comparison


T = TypeVar("T")
COMMAND_HEARTBEAT_SECONDS = 10.0
BASE_RUN_DIRECTORIES = (
    "00_INPUT",
    "01_OBSERVATIONS",
    "preflight",
    "06_EVALUATION",
    "07_COMPARISON",
    "99_FINAL_RESULTS",
    "logs",
)
METHOD_DIRECTORIES = {
    "ap01": "02_AP01",
    "ap02": "03_AP02",
    "ap03": "04_AP03",
}
TERMINAL_PREFIXES = (
    "[OK]",
    "[WARN]",
    "[WARNING]",
    "[ERROR]",
    "[INFO]",
    "[REUSE]",
    "ERROR:",
    "WARNING:",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_id(config: RigConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{config.project.run_label}_{config_fingerprint(config)[:8]}"


def _run_directories(config: RigConfig) -> tuple[str, ...]:
    selected = tuple(
        METHOD_DIRECTORIES[method_id]
        for method_id in config.methods.enabled
        if method_id in METHOD_DIRECTORIES
    )
    return (*BASE_RUN_DIRECTORIES[:2], *selected, *BASE_RUN_DIRECTORIES[2:])


def _automatic_scientific_selections(config: RigConfig) -> bool:
    return (
        config.methods.ap01.root_camera == "auto"
        and config.methods.ap02.reference_marker_id == "auto"
        and config.methods.ap03_single.scale_marker_id == "auto"
        and config.methods.ap03_multi.marker_ids == "auto"
        and config.evaluation.anchor_marker_id == "auto"
    )


def _materialize_tree(source: Path, destination: Path) -> dict[str, int]:
    """Place immutable input files in results, using hardlinks when possible."""
    counts = {"hardlinked": 0, "copied": 0, "existing": 0}
    if not source.is_dir():
        return counts
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item.stat().st_size:
                raise RuntimeError(
                    f"Published input conflicts with the canonical dataset: {target}"
                )
            counts["existing"] += 1
            continue
        try:
            os.link(item, target)
            counts["hardlinked"] += 1
        except OSError:
            shutil.copy2(item, target)
            counts["copied"] += 1
    return counts


def planned_stages(
    config: RigConfig, *, defer_evaluation: bool = False
) -> list[tuple[str, str]]:
    stages = [
        ("prepare_inputs", "Prepare canonical inputs and provenance"),
        ("validate_dataset", "Validate the canonical dataset"),
        ("detect_markers", "Detect shared ArUco observations and debug images"),
        (
            "observation_quality",
            "Apply immutable checks and job-specific observation quality",
        ),
        ("analyze_selections", "Analyze root-camera and method-marker candidates"),
    ]
    if config.project.execution_mode == "prepare_only":
        stages.append(("finalize", "Write input-preparation report"))
        return stages
    stages.extend(
        (f"method_{method_id}", calibration_methods.get(method_id).display_name)
        for method_id in config.methods.enabled
    )
    if config.evaluation.enabled and not defer_evaluation:
        stages.append(("evaluation", "Evaluate with the frozen preflight anchor"))
    stages.extend(
        [
            ("comparison", "Normalize and compare method results"),
            ("finalize", "Write final report"),
        ]
    )
    return stages


def observation_id(config: RigConfig) -> str:
    """Content ID for one versioned ArUco observation contract."""
    payload = {
        "dictionary": config.markers.dictionary,
        "length_m": config.markers.length_m,
        "detection": effective_detector_config(
            config.markers.detection_mode,
            config.markers.dictionary,
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (
        "detection_"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    )


__all__ = [
    'T',
    'COMMAND_HEARTBEAT_SECONDS',
    'BASE_RUN_DIRECTORIES',
    'METHOD_DIRECTORIES',
    'TERMINAL_PREFIXES',
    '_now',
    '_write_json',
    '_read_json',
    '_run_id',
    '_run_directories',
    '_automatic_scientific_selections',
    '_materialize_tree',
    'planned_stages',
    'observation_id',
]
