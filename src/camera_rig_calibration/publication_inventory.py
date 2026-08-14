"""Focused atomic-publication responsibility."""

from __future__ import annotations

import hashlib
import csv
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .anchor_export import export_method_anchor_poses
from .dataset.discovery import safe_id
from .dataset_identity import (
    build_dataset_identity,
    identities_match,
    write_dataset_identity,
)
from .evaluation.reporting import write_scientific_experiment_reports
from .experiments import (
    experiment_manifest_payload,
    experiment_paths,
    method_result_label,
)
from .filesystem import rename_with_retry
from .storage_layout import storage_manifest


from .publication_core import (
    _now,
    _read_json,
    _write_json,
)
from .publication_dataset import (
    _relative,
)

def _comparison_rows(root: Path) -> list[dict[str, Any]]:
    current: dict[tuple[str, str], dict[str, Any]] = {}
    attempts = root / "attempts"
    if attempts.is_dir():
        for failure_path in sorted(attempts.glob("*/*/*/FAILURE.json")):
            payload = _read_json(failure_path)
            if (
                payload.get("superseded") is True
                or payload.get("current_in_comparison") is False
            ):
                continue
            method = str(
                payload.get("method", failure_path.parents[2].name)
            )
            label = str(
                payload.get("label", failure_path.parents[1].name)
            )
            # Sorted paths make the newest attempt replace an older failure.
            # A later authoritative result below always replaces failures.
            current[(method, label)] = {
                "method": method,
                "label": label,
                "status": "failed",
                "runtime_seconds": None,
                "static_camera_count": None,
                "primary_result": None,
                "result_path": _relative(failure_path.parent, root),
                "warning": payload.get("explanation", ""),
                "artifact_status": "failed",
                "quality_status": "not_available",
                "warnings": [payload.get("explanation", "")],
                "config_summary": {},
                "metrics": {},
            }
    methods = root / "methods"
    if methods.is_dir():
        for result_path in sorted(methods.glob("*/*/RESULT.json")):
            payload = _read_json(result_path)
            method = str(
                payload.get("method", result_path.parents[1].name)
            )
            if (
                method == "ap03"
                and payload.get("comparison_visibility")
                == "hidden_when_scale_variants_available"
                and (
                    root
                    / "methods"
                    / "ap03_single"
                    / result_path.parent.name
                    / "RESULT.json"
                ).is_file()
                and (
                    root
                    / "methods"
                    / "ap03_multi"
                    / result_path.parent.name
                    / "RESULT.json"
                ).is_file()
            ):
                continue
            label = str(payload.get("label", result_path.parent.name))
            current[(method, label)] = {
                "method": method,
                "label": label,
                "status": payload.get("status", "available"),
                "runtime_seconds": payload.get("runtime_seconds"),
                "static_camera_count": payload.get("static_camera_count"),
                "primary_result": payload.get("primary_result"),
                "result_path": _relative(result_path.parent, root),
                "warning": "; ".join(
                    str(item)
                    for item in payload.get("warnings", [])
                    if item
                )
                or payload.get("warning")
                or payload.get("error")
                or "",
                "artifact_status": payload.get(
                    "artifact_status", payload.get("status", "available")
                ),
                "execution_status": payload.get(
                    "execution_status", "completed"
                ),
                "solver_status": payload.get(
                    "solver_status", "not_applicable"
                ),
                "quality_status": payload.get("quality_status", "unknown"),
                "calibration_status": payload.get(
                    "calibration_status",
                    payload.get("artifact_status", "available"),
                ),
                "evaluation_status": payload.get(
                    "evaluation_status", "not_run"
                ),
                "anchor_export_status": payload.get(
                    "anchor_export_status", "ANCHOR_NOT_AVAILABLE"
                ),
                "visualization_status": payload.get(
                    "visualization_status", "unavailable"
                ),
                "warnings": payload.get("warnings", []),
                "config_summary": payload.get("config_summary", {}),
                "metrics": payload.get("metrics", {}),
            }
    return [
        current[key]
        for key in sorted(current)
    ]


def _write_inventory_reports(
    root: Path,
    *,
    dataset_root: Path,
    category: str,
    experiment: str,
    sampling_rate: str,
    queue_id: str,
    queue_complete: bool,
) -> None:
    rows = _comparison_rows(root)
    successful = sum(
        row.get("artifact_status") == "available" for row in rows
    )
    failed = sum(row.get("artifact_status") == "failed" for row in rows)
    warning_count = sum(
        row.get("quality_status")
        not in {"good", "converged", "not_available"}
        for row in rows
        if row.get("artifact_status") == "available"
    )
    scientific_status = (
        "partial"
        if successful and failed
        else "failed"
        if failed
        else "available"
        if successful
        else "prepared"
    )
    comparison = {
        "schema_version": 5,
        "layout_version": 2,
        "experiment": experiment,
        "status": scientific_status,
        "methods": rows,
        "evaluation_path": "evaluations",
        "human_report": "RESULTS.txt",
        "quality_status": "warnings" if warning_count else "ok",
        "visualization": _read_json(
            root / "visualization" / "visualization_manifest.json"
        ),
        "generated_at": (
            _read_json(root / "COMPARISON.json").get("generated_at")
            or _now()
        ),
    }
    _write_json(root / "COMPARISON.json", comparison)
    fields = [
        "method",
        "label",
        "status",
        "runtime_seconds",
        "static_camera_count",
        "primary_result",
        "result_path",
        "warning",
        "artifact_status",
        "execution_status",
        "solver_status",
        "quality_status",
        "calibration_status",
        "evaluation_status",
        "anchor_export_status",
        "visualization_status",
        "config_summary",
    ]
    with (root / "COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                )
                for key in fields
            }
            for row in rows
        )
    summary = {
        "schema_version": 5,
        "layout_version": 2,
        "experiment": experiment,
        "category": category,
        "sampling_rate": sampling_rate,
        "status": scientific_status,
        "queue_id": queue_id,
        "queue_complete": queue_complete,
        "dataset_path": str(dataset_root.resolve()),
        "results_path": str((root / "RESULTS.txt").resolve()),
        "comparison_path": str((root / "COMPARISON.json").resolve()),
        "available_methods": successful,
        "failed_attempts": failed,
        "quality_warnings": warning_count,
        "visualization": _read_json(
            root / "visualization" / "visualization_manifest.json"
        ),
        "methods": rows,
        "updated_at": (
            _read_json(root / "SUMMARY.json").get("updated_at")
            or _now()
        ),
    }
    _write_json(root / "SUMMARY.json", summary)
    (root / "SUMMARY.txt").unlink(missing_ok=True)
    (root / "COMPARISON.txt").unlink(missing_ok=True)


def write_experiment_reports(
    root: Path,
    *,
    config: Any,
    queue_id: str,
    queue_complete: bool,
) -> None:
    paths = experiment_paths(config)
    category = config.dataset.category.value
    if category == "real_vehicle" and queue_complete:
        from .evaluation.reporting import run_real_marker_consistency

        run_real_marker_consistency(
            root,
            paths.dataset_root,
            force=True,
        )
    write_scientific_experiment_reports(
        root,
        dataset_root=paths.dataset_root,
        category=category,
    )
    _write_inventory_reports(
        root,
        dataset_root=paths.dataset_root,
        category=category,
        experiment=config.project.experiment_id or config.dataset.id,
        sampling_rate=(
            f"{config.sampling.target_hz:g}Hz"
            if config.sampling.target_hz is not None
            else "native_rate"
        ),
        queue_id=queue_id,
        queue_complete=queue_complete,
    )


def _native_calibration_hashes(root: Path) -> dict[str, str]:
    """Hash authoritative method outputs that reconcile must never mutate."""
    hashes: dict[str, str] = {}
    methods_root = root / "methods"
    for method in ("ap01", "ap02", "ap03"):
        method_root = methods_root / method
        if not method_root.is_dir():
            continue
        files = [
            *method_root.glob("*/camera_extrinsics.csv"),
            *method_root.glob("*/provenance/resolved_config.yaml"),
            *method_root.glob("*/diagnostics/method/**/*"),
        ]
        for path in sorted(
            (item for item in files if item.is_file()),
            key=lambda item: item.as_posix(),
        ):
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(8 * 1024 * 1024), b""
                ):
                    digest.update(chunk)
            hashes[relative] = digest.hexdigest()
    return hashes

__all__ = [
    '_comparison_rows',
    '_write_inventory_reports',
    'write_experiment_reports',
    '_native_calibration_hashes',
]
