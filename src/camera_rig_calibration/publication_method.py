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
    METHOD_DIRECTORIES,
    _atomic_replace,
    _materialize_semantic_tree,
    _materialize_tree,
    _now,
    _read_json,
    _rename_with_retry,
    _write_json,
)
from .publication_dataset import (
    _export_accepted_extrinsics,
    _export_extrinsics,
    _method_and_label,
    _method_status,
    _reference_metadata,
    _relative,
    _runtime_seconds,
)

def _publish_success(
    source: Path,
    *,
    config: Any,
    canonical_root: Path,
    queue_id: str,
) -> tuple[Path, str]:
    method_id, label, manifest = _method_and_label(source, config)
    target = canonical_root / "methods" / method_id / label
    method_fingerprint = str(manifest.get("method_fingerprint", ""))
    force_replacement = config.project.duplicate_policy == "force"
    if target.is_dir():
        existing = _read_json(target / "RESULT.json")
        if (
            existing.get("method_fingerprint") == method_fingerprint
            and method_fingerprint
            and not force_replacement
        ):
            return target, "duplicate_skipped"
        if not force_replacement:
            raise RuntimeError(
                f"Result label conflict at {target}: label '{label}' already "
                "belongs to a different configuration. Choose a new run label."
            )

    # A prior method may have completed fully but encountered a transient
    # Windows/WSL directory lock during the final rename. Reuse that validated
    # staging directory so resuming publication never recomputes the method.
    prefix = f".incoming_{label}_"
    staged = sorted(
        (
            item
            for item in target.parent.glob(f"{prefix}*")
            if item.is_dir()
        ),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in staged:
        candidate_result = _read_json(candidate / "RESULT.json")
        if (
            method_fingerprint
            and candidate_result.get("method_fingerprint")
            == method_fingerprint
            and candidate_result.get("method") == method_id
            and candidate_result.get("label") == label
            and (candidate / "RESULT.txt").is_file()
            and (candidate / "camera_extrinsics.csv").is_file()
            and (candidate / "provenance").is_dir()
        ):
            _atomic_replace(candidate, target)
            return target, "completed"

    incoming = target.with_name(
        f".incoming_{label}_{os.getpid()}_{time.time_ns()}"
    )
    incoming.mkdir(parents=True)
    (incoming / "diagnostics" / "method").mkdir(parents=True)
    (incoming / "logs").mkdir()
    (incoming / "provenance").mkdir()
    status = _method_status(source, method_id)
    method_source = source / METHOD_DIRECTORIES.get(method_id, Path(method_id))
    _materialize_semantic_tree(
        method_source, incoming / "diagnostics" / "method"
    )
    _materialize_tree(source / "preflight", incoming / "diagnostics" / "preflight")
    _materialize_tree(
        source / "06_EVALUATION", incoming / "diagnostics" / "evaluation"
    )
    _materialize_tree(source / "logs", incoming / "logs")
    provenance = incoming / "provenance"
    for name in (
        "requested_config.yaml",
        "resolved_config.yaml",
        "method_config_diff.json",
        "run_manifest.json",
        "commands.txt",
        "environment.json",
        "timings.json",
    ):
        item = source / name
        if item.is_file():
            provenance.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, provenance / name)
    observation_config = (
        source / "01_OBSERVATIONS" / "detection_config.json"
    )
    if observation_config.is_file():
        shutil.copy2(
            observation_config,
            provenance / "observation_detection_config.json",
        )

    extrinsics = incoming / "camera_extrinsics.csv"
    has_extrinsics = _export_extrinsics(
        source, extrinsics, method_id, status
    )
    accepted_extrinsics = incoming / "camera_extrinsics_accepted.csv"
    has_accepted_extrinsics = (
        _export_accepted_extrinsics(source, accepted_extrinsics)
        if method_id == "ap01"
        else False
    )
    reference, convention = _reference_metadata(method_id, status)
    if not has_extrinsics:
        extrinsics.write_text(
            "reference_frame,transform_convention\n",
            encoding="utf-8",
        )
    runtime = _runtime_seconds(source, method_id)
    cameras = status.get("available_static_cameras")
    camera_count = len(cameras) if isinstance(cameras, list) else None
    result_payload = {
        "schema_version": 5,
        "layout_version": 2,
        "status": "available",
        "artifact_status": "available",
        "quality_status": status.get("quality_status", "converged"),
        "method": method_id,
        "label": label,
        "primary_result": status.get(
            "primary_result",
            "combined" if method_id == "ap02" else "multi" if method_id == "ap03" else "baseline",
        ),
        "method_fingerprint": method_fingerprint,
        "input_fingerprint": manifest.get("input_id"),
        "runtime_seconds": runtime,
        "static_camera_count": camera_count,
        "available_static_cameras": cameras or [],
        "reference_frame": reference,
        "transform_convention": convention,
        "camera_extrinsics": "camera_extrinsics.csv",
        "camera_extrinsics_available": has_extrinsics,
        "diagnostics": "diagnostics",
        "logs": "logs",
        "provenance": "provenance",
        "queue_id": queue_id,
        "published_at": _now(),
    }
    if method_id == "ap01":
        if not has_accepted_extrinsics:
            accepted_extrinsics.write_text(
                "reference_frame,transform_convention\n",
                encoding="utf-8",
            )
        result_payload.update(
            {
                "camera_extrinsics_accepted": (
                    "camera_extrinsics_accepted.csv"
                ),
                "estimate_status": status.get(
                    "estimate_status", "available"
                ),
                "deployment_eligible": status.get(
                    "deployment_eligible", False
                ),
                "evaluation_status": status.get(
                    "evaluation_status", "pending"
                ),
                "camera_statuses": status.get("camera_statuses", {}),
                "deployment_eligible_cameras": status.get(
                    "deployment_eligible_cameras", []
                ),
            }
        )
    for key in (
        "reference_marker_id",
        "root_camera",
        "success",
        "warning",
        "error",
        "full_rig_result_available",
        "comparison_eligible",
        "diagnostic_partial",
        "missing_static_cameras",
        "graph_component_count",
        "primary_component_id",
        "cross_component_extrinsics",
    ):
        if key in status:
            result_payload[key] = status[key]
    _write_json(incoming / "RESULT.json", result_payload)
    lines = [
        "RIGCAL CALIBRATION RESULT",
        "=" * 72,
        "",
        f"Method: {method_id}",
        f"Variant: {label}",
        "Status: available",
        f"Primary result: {result_payload['primary_result']}",
        f"Runtime: {runtime:.1f} s" if runtime is not None else "Runtime: n/a",
        f"Static cameras: {camera_count}" if camera_count is not None else "Static cameras: n/a",
        f"Reference frame: {reference}",
        f"Transform convention: {convention}",
        "",
        "Final camera poses: camera_extrinsics.csv"
        if has_extrinsics
        else "Final camera poses: unavailable (see diagnostics)",
        "Diagnostics: diagnostics/",
        "Complete logs: logs/",
        "Reproducibility: provenance/",
        "",
    ]
    (incoming / "RESULT.txt").write_text("\n".join(lines), encoding="utf-8")
    # Derive the common-anchor export only from the already completed primary
    # method artifacts. This never invokes or modifies a calibration method.
    export_method_anchor_poses(incoming)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    attempts_root = canonical_root / "attempts" / method_id / label
    superseded_result: Path | None = None
    if target.is_dir():
        superseded_result = attempts_root / f"{stamp}_superseded_result"
        superseded_result.parent.mkdir(parents=True, exist_ok=True)
        _rename_with_retry(target, superseded_result)
        _write_json(
            superseded_result / "ATTEMPT.json",
            {
                "schema_version": 5,
                "layout_version": 2,
                "status": "superseded_success",
                "superseded": True,
                "current_in_comparison": False,
                "method": method_id,
                "label": label,
                "superseded_by": f"methods/{method_id}/{label}",
                "archived_at": _now(),
            },
        )
    try:
        _rename_with_retry(incoming, target)
    except Exception:
        if (
            superseded_result is not None
            and superseded_result.exists()
            and not target.exists()
        ):
            _rename_with_retry(superseded_result, target)
        raise
    for failure_path in sorted(
        attempts_root.glob("*/FAILURE.json")
    ):
        failure = _read_json(failure_path)
        if not failure or failure.get("superseded"):
            continue
        failure.update(
            {
                "superseded": True,
                "current_in_comparison": False,
                "superseded_by": f"methods/{method_id}/{label}",
                "superseded_at": _now(),
            }
        )
        _write_json(failure_path, failure)
    success_attempt = attempts_root / f"{stamp}_successful"
    success_attempt.mkdir(parents=True, exist_ok=False)
    attempt_payload = {
        "schema_version": 5,
        "layout_version": 2,
        "status": "successful",
        "method": method_id,
        "label": label,
        "current_in_comparison": True,
        "current_result": f"methods/{method_id}/{label}",
        "method_fingerprint": method_fingerprint,
        "dataset_identity": manifest.get("dataset_identity"),
        "supersedes_attempt": manifest.get("supersedes_attempt"),
        "reused_stages": manifest.get("reused_stages", []),
        "rerun_stages": manifest.get("rerun_stages", []),
        "algorithm_version": manifest.get("algorithm_version"),
        "completed_at": _now(),
    }
    _write_json(success_attempt / "ATTEMPT.json", attempt_payload)
    for name in ("run_manifest.json", "resolved_config.yaml", "timings.json"):
        source_file = target / "provenance" / name
        if source_file.is_file():
            shutil.copy2(source_file, success_attempt / name)
    return target, "completed"


def _failure_summary(source: Path, error: str) -> dict[str, Any]:
    evidence = error.strip()
    log_evidence = ""
    for log_path in sorted((source / "logs").glob("*.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "failed to create sparse model" in text.lower():
            log_evidence = "ERROR: failed to create sparse model"
            break
        relevant = [
            line.strip()
            for line in text.splitlines()
            if "error" in line.lower() or "failed" in line.lower()
        ]
        if relevant:
            log_evidence = relevant[-1]
    combined = f"{evidence}\n{log_evidence}".lower()
    if (
        "different immutable dataset" in combined
        or "immutable publication conflict" in combined
        or "different rig/capture parameter contract" in combined
    ):
        code = "internal_dataset_publication_conflict"
        explanation = (
            "An internal prepare/publication step attempted to bind the run "
            "to a different immutable dataset identity. The calibration method "
            "did not start."
        )
        stage = "prepare_publication"
    elif "failed to create sparse model" in combined:
        code = "colmap_sparse_model_failed"
        explanation = (
            "COLMAP could not register enough images into a sparse model "
            "under the selected configuration."
        )
        stage = "method_estimation"
    elif "preflight" in combined:
        code = "preflight_failed"
        explanation = "Input or observation-quality preflight rejected this job."
        stage = "preflight"
    elif "timeout" in combined:
        code = "timeout"
        explanation = "A required process exceeded its configured timeout."
        stage = "external_process"
    elif "validation" in combined or "configuration" in combined:
        code = "configuration_validation_failed"
        explanation = "The resolved configuration violated a rigcal contract."
        stage = "configuration"
    elif "least_squares" in combined or "optimizer" in combined:
        code = "optimizer_failed"
        explanation = "The numerical optimizer did not complete successfully."
        stage = "method_estimation"
    else:
        code = "method_failed"
        explanation = "The method failed; complete evidence is retained in this attempt."
        stage = "method_estimation"
    return {
        "schema_version": 5,
        "layout_version": 2,
        "status": "failed",
        "scientific_validity": "incomplete/non-authoritative",
        "cause_code": code,
        "stage": stage,
        "explanation": explanation,
        "evidence": log_evidence or evidence or "No concise evidence line available",
        "original_error": error,
        "superseded": False,
        "current_in_comparison": True,
    }


def _publish_failure(
    source: Path,
    *,
    config: Any,
    canonical_root: Path,
    entry_id: str,
    error: str,
) -> tuple[Path, dict[str, Any]]:
    if source.is_dir():
        method_id, label, manifest = _method_and_label(source, config)
        run_id = safe_id(str(manifest.get("run_id") or source.name))
    else:
        method_id = next(iter(config.methods.enabled), "unknown")
        label = method_result_label(config, method_id)
        run_id = safe_id(entry_id)
    target = canonical_root / "attempts" / method_id / label / run_id
    suffix = 2
    while target.exists():
        target = target.with_name(f"{run_id}_{suffix}")
        suffix += 1
    target.mkdir(parents=True)
    if source.is_dir():
        _materialize_tree(source, target / "diagnostics")
    summary = _failure_summary(source, error)
    summary.update(
        {
            "method": method_id,
            "label": label,
            "attempt": _relative(target, canonical_root),
        }
    )
    _write_json(target / "FAILURE.json", summary)
    (target / "FAILURE.txt").write_text(
        "INCOMPLETE / NON-AUTHORITATIVE CALIBRATION ATTEMPT\n"
        + "=" * 64
        + f"\n\nMethod: {method_id}\nVariant: {label}\n"
        + f"Cause: {summary['cause_code']}\n"
        + f"Explanation: {summary['explanation']}\n"
        + f"Evidence: {summary['evidence']}\n",
        encoding="utf-8",
    )
    return target, summary

__all__ = [
    '_publish_success',
    '_failure_summary',
    '_publish_failure',
]
