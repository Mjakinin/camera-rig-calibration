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
    PRIMARY_POSES,
    _atomic_replace,
    _dataset_fingerprint,
    _finalize_dataset_front_door,
    _materialize_tree,
    _read_json,
    _validate_dataset,
)

def _publish_dataset(
    source: Path, canonical: Path, *, config: Any
) -> Path:
    """Publish the complete dataset exactly once, before any method result."""
    _validate_dataset(source)
    incoming_fingerprint = _dataset_fingerprint(source)
    if not incoming_fingerprint:
        raise RuntimeError(
            f"Prepared dataset has no immutable content fingerprint: {source}"
        )
    if canonical.is_dir():
        existing_fingerprint = _dataset_fingerprint(canonical)
        if existing_fingerprint != incoming_fingerprint:
            raise RuntimeError(
                f"Experiment '{canonical.name}' already contains a different "
                "dataset. Choose a new experiment ID."
            )
        incoming_identity = build_dataset_identity(source)
        existing_identity = build_dataset_identity(canonical)
        if not identities_match(incoming_identity, existing_identity):
            raise RuntimeError(
                f"Experiment '{canonical.name}' contains different immutable "
                "image, camera-info, World-snapshot, route or capture content. "
                "Choose a new experiment ID."
            )
        # A complete canonical dataset is immutable and immediately reusable.
        # Queue retries may regenerate equivalent quality reports containing
        # different transient job paths or timestamps; those are not allowed
        # to mutate the already published scientific input.
        try:
            _validate_dataset(canonical)
        except RuntimeError:
            pass
        else:
            _finalize_dataset_front_door(canonical, config)
            write_dataset_identity(canonical)
            return canonical
        # Complete an older layout-v2 front door only with missing or
        # byte-identical late-preflight evidence. Conflicts remain fatal.
        _materialize_tree(
            source / "observations",
            canonical / "observations",
            keep_existing=("PUBLICATION_COMPLETE.json",),
        )
        _materialize_tree(
            source / "metadata" / "simulation",
            canonical / "metadata" / "simulation",
        )
        _validate_dataset(canonical)
        _finalize_dataset_front_door(canonical, config)
        write_dataset_identity(canonical)
        return canonical

    incoming = canonical.with_name(
        f".incoming_{canonical.name}_{os.getpid()}_{time.time_ns()}"
    )
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    _materialize_tree(source, incoming)
    (incoming / "README.txt").unlink(missing_ok=True)
    _atomic_replace(incoming, canonical)
    _finalize_dataset_front_door(canonical, config)
    write_dataset_identity(canonical)
    return canonical


def _method_and_label(source: Path, config: Any) -> tuple[str, str, dict[str, Any]]:
    manifest = _read_json(source / "run_manifest.json")
    method_id = str(
        manifest.get("method_id")
        or next(iter(manifest.get("enabled_methods", [])), "")
        or next(iter(config.methods.enabled), "unknown")
    )
    label = str(manifest.get("variant") or method_result_label(config, method_id))
    return method_id, label, manifest


def _method_status(source: Path, method_id: str) -> dict[str, Any]:
    method_root = source / METHOD_DIRECTORIES.get(method_id, Path(method_id))
    candidates = [
        method_root / "METHOD_STATUS.json",
        *sorted(method_root.rglob("METHOD_STATUS.json")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _read_json(candidate)
    return {}


def _runtime_seconds(source: Path, method_id: str) -> float | None:
    timings = _read_json(source / "timings.json")
    structured = timings.get("_structured", {})
    if isinstance(structured, dict):
        entry = structured.get(f"method_{method_id}", {})
        if isinstance(entry, dict):
            value = entry.get("stage_elapsed_seconds")
            if value is not None:
                return float(value)
    value = timings.get(f"method_{method_id}")
    return float(value) if isinstance(value, (int, float)) else None


def _reference_metadata(method_id: str, status: dict[str, Any]) -> tuple[str, str]:
    if method_id == "ap01":
        reference = str(
            status.get("root_camera")
            or status.get("reference_camera")
            or "resolved root camera"
        )
    elif method_id == "ap02":
        marker = status.get("reference_marker_id", "resolved")
        reference = f"ArUco marker {marker}"
    else:
        reference = "COLMAP gauge with metric marker scale"
    return reference, "T_reference_camera (camera pose expressed in reference frame)"


def _export_extrinsics(
    source: Path,
    destination: Path,
    method_id: str,
    status: dict[str, Any],
) -> bool:
    pose_source = source / PRIMARY_POSES.get(method_id, Path("__missing__"))
    if not pose_source.is_file():
        return False
    reference, convention = _reference_metadata(method_id, status)
    with pose_source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])
    fields = ["reference_frame", "transform_convention", *original_fields]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reference_frame": reference,
                    "transform_convention": convention,
                    **row,
                }
            )
    return True


def _export_accepted_extrinsics(source: Path, destination: Path) -> bool:
    """Export only AP01 poses explicitly approved for deployment."""

    pose_source = (
        source
        / "02_AP01/03_static_extrinsics/"
        "AP01_STATIC_CAMERA_POSES_ACCEPTED.csv"
    )
    if not pose_source.is_file():
        return False
    reference = "resolved root camera"
    convention = "T_reference_camera (camera pose expressed in reference frame)"
    with pose_source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reference_frame", "transform_convention", *fields],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reference_frame": reference,
                    "transform_convention": convention,
                    **row,
                }
            )
    return True


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())

__all__ = [
    '_publish_dataset',
    '_method_and_label',
    '_method_status',
    '_runtime_seconds',
    '_reference_metadata',
    '_export_extrinsics',
    '_export_accepted_extrinsics',
    '_relative',
]
