from __future__ import annotations

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
from .dataset.discovery import safe_id
from .evaluation.reporting import write_scientific_experiment_reports
from .experiments import (
    experiment_manifest_payload,
    experiment_paths,
    method_result_label,
)
from .filesystem import rename_with_retry
from .storage_layout import storage_manifest


METHOD_DIRECTORIES = {
    "ap01": Path("02_AP01"),
    "ap02": Path("03_AP02"),
    "ap03": Path("04_AP03"),
}
PRIMARY_POSES = {
    "ap01": Path(
        "02_AP01/03_static_extrinsics/"
        "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
    ),
    "ap02": Path(
        "03_AP02/07_graph_ba/with_moving/"
        "optimized_static_camera_poses_ref_marker.csv"
    ),
    "ap03": Path(
        "04_AP03/scale_multi/"
        "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_tree(
    source: Path,
    destination: Path,
    *,
    keep_existing: tuple[str, ...] = (),
) -> None:
    """Copy/hardlink a tree and reject any differing existing file."""
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if relative.as_posix() in keep_existing:
                continue
            if (
                target.stat().st_size != item.stat().st_size
                or _sha256(target) != _sha256(item)
            ):
                raise RuntimeError(
                    f"Immutable publication conflict at {target}"
                )
            continue
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


def _materialize_semantic_tree(source: Path, destination: Path) -> None:
    """Publish method internals with descriptive names, not stage numbers."""
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        semantic = Path(
            *(
                re.sub(r"^\d+_", "", part) or part
                for part in relative.parts
            )
        )
        target = destination / semantic
        if item.is_symlink():
            continue
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(
                    f"Semantic diagnostic-name collision at {target}"
                )
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def _rename_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 8,
) -> None:
    """Backward-compatible alias for the shared filesystem helper."""
    rename_with_retry(source, target, attempts=attempts)


def _atomic_replace(incoming: Path, target: Path) -> None:
    backup = target.with_name(
        f".previous_{target.name}_{os.getpid()}_{time.time_ns()}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _rename_with_retry(target, backup)
    try:
        _rename_with_retry(incoming, target)
    except Exception:
        if backup.exists() and not target.exists():
            _rename_with_retry(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_fingerprint(root: Path) -> str:
    descriptor = _read_json(root / "dataset.json")
    value = descriptor.get("input_fingerprint")
    if value:
        return str(value)
    source = _read_json(root / "metadata" / "source.json")
    return str(source.get("input_id", ""))


def _validate_dataset(root: Path) -> None:
    if not (root / "raw_images").is_dir():
        raise RuntimeError(
            f"Prepared transaction has no raw_images directory: {root}"
        )
    required = (
        "shared_static_aruco_observations.csv",
        "shared_moving_aruco_observations.csv",
        "shared_all_aruco_observations.csv",
        "SELECTION_CANDIDATES.json",
        "REFERENCE_SELECTIONS.json",
        "REFERENCE_MARKER_ID.txt",
        "PUBLICATION_COMPLETE.json",
    )
    missing = [
        name for name in required if not (root / "observations" / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            "The canonical dataset is not publishable because observation "
            "generation is incomplete: " + ", ".join(missing)
        )
    if not (root / "observations" / "quality").is_dir():
        raise RuntimeError(
            "The canonical dataset is not publishable because observation "
            "quality evidence is missing."
        )
    if not (root / "observations" / "debug_images").is_dir():
        raise RuntimeError(
            "The canonical dataset is not publishable because the ArUco debug "
            "images are missing."
        )


def _refresh_dataset_descriptor(root: Path, config: Any) -> None:
    """Bind a prepared dataset descriptor to the final experiment config."""
    descriptor_path = root / "dataset.json"
    descriptor = _read_json(descriptor_path)
    if descriptor:
        input_id = str(descriptor.get("input_fingerprint", "")).strip()
        if not input_id:
            raise RuntimeError(
                f"Prepared dataset descriptor has no input fingerprint: "
                f"{descriptor_path}"
            )
        descriptor.update(
            experiment_manifest_payload(
                config,
                experiment_paths(config),
                input_id,
                created_at=(
                    str(descriptor["created_at"])
                    if descriptor.get("created_at")
                    else None
                ),
            )
        )
        _write_json(descriptor_path, descriptor)


def _finalize_dataset_front_door(
    canonical: Path, config: Any
) -> None:
    """Replace transaction-only paths with stable canonical metadata."""
    _refresh_dataset_descriptor(canonical, config)
    source_path = canonical / "metadata" / "source.json"
    source = _read_json(source_path)
    if source:
        source.pop("canonical_source_roots", None)
        source["canonical_dataset_root"] = str(canonical.resolve())
        _write_json(source_path, source)


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


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


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
    if target.is_dir():
        existing = _read_json(target / "RESULT.json")
        if (
            existing.get("method_fingerprint") == method_fingerprint
            and method_fingerprint
        ):
            return target, "duplicate_skipped"
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
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(incoming, target)
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
                "quality_status": payload.get("quality_status", "unknown"),
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
        row.get("quality_status") not in {"converged", "not_available"}
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
        "generated_at": _now(),
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
        "quality_status",
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
        "methods": rows,
        "updated_at": _now(),
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


def reconcile_existing_experiment(
    root: Path,
    *,
    dataset_root: Path,
    category: str,
) -> dict[str, Any]:
    """Rebuild a layout-v2 front door without rerunning calibration methods."""
    from .evaluation.reporting import (
        complete_existing_dataset,
        run_real_marker_consistency,
    )

    complete_existing_dataset(dataset_root, root)
    if category == "real_vehicle":
        run_real_marker_consistency(root, dataset_root, force=True)
    payload = write_scientific_experiment_reports(
        root,
        dataset_root=dataset_root,
        category=category,
    )
    previous = _read_json(root / "SUMMARY.json")
    sampling_rate = str(previous.get("sampling_rate", "native_rate"))
    _write_inventory_reports(
        root,
        dataset_root=dataset_root,
        category=category,
        experiment=str(previous.get("experiment") or root.name),
        sampling_rate=sampling_rate,
        queue_id=str(previous.get("queue_id") or "reconciled"),
        queue_complete=True,
    )
    return payload


def publish_preparation_transaction(
    transaction_root: Path,
    *,
    queue_id: str,
    config: Any,
    preparation: Path,
) -> Path:
    """Publish a complete layout-v2 dataset without a calibration result."""
    del preparation
    source = transaction_root.resolve() / "dataset"
    # A detector retry changes the experiment ID and observation contract
    # after the normalized capture was prepared. Method jobs reuse this
    # transaction dataset, so its descriptor must carry the final identity as
    # well as the canonical copy. Otherwise method startup sees the stale
    # baseline fingerprint and rejects its own prepared input.
    _refresh_dataset_descriptor(source, config)
    canonical = experiment_paths(config).dataset_root
    published = _publish_dataset(source, canonical, config=config)
    preparation_record = published / "metadata" / "preparation.json"
    if not preparation_record.is_file():
        _write_json(
            preparation_record,
            {
                "schema_version": 5,
                "layout_version": 2,
                "queue_id": queue_id,
                "status": "prepared",
                "published_at": _now(),
            },
        )
    return published


def publish_queue_transaction(
    transaction_root: Path,
    *,
    queue_id: str,
    configs: list[Any],
    results: dict[str, dict[str, Any]],
    finalize: bool = True,
) -> dict[str, dict[str, Any]]:
    """Publish one immutable dataset and independent layout-v2 method outcomes."""
    transaction = transaction_root.resolve()
    terminal = {
        "completed",
        "duplicate_skipped",
        "failed",
        "failed_preflight",
        "skipped_dependency",
        "published",
        "failed_published",
    }
    if finalize and (
        len(results) != len(configs)
        or any(str(row.get("status")) not in terminal for row in results.values())
    ):
        return results
    if not configs:
        return results

    first = configs[0]
    paths = experiment_paths(first)
    # This intentionally happens before any method result is made visible.
    _publish_dataset(
        transaction / "dataset",
        paths.dataset_root,
        config=first,
    )
    paths.root.mkdir(parents=True, exist_ok=True)

    config_by_label = {
        safe_id(config.project.run_label): config for config in configs
    }
    for entry_id, row in results.items():
        status = str(row.get("status", "unknown"))
        if status in {"duplicate_skipped", "published", "failed_published"} or (
            status == "completed" and row.get("published") is True
        ):
            continue
        source_text = str(row.get("result", "")).strip()
        source = (
            Path(source_text)
            if source_text
            else transaction / "__missing_method_result__"
        )
        config = config_by_label.get(safe_id(entry_id), first)
        row_config = Path(str(row.get("config", "")))
        if row_config.is_file():
            config = load_config(row_config)
        if source.is_dir() and (source / "resolved_config.yaml").is_file():
            config = load_config(source / "resolved_config.yaml")
        if status == "completed":
            target, outcome = _publish_success(
                source,
                config=config,
                canonical_root=paths.root,
                queue_id=queue_id,
            )
            row.update(
                {
                    "status": "completed" if outcome == "completed" else outcome,
                    "result": str(target.resolve()),
                    "published": True,
                }
            )
        else:
            target, failure = _publish_failure(
                source,
                config=config,
                canonical_root=paths.root,
                entry_id=entry_id,
                error=(
                    f"{status}: "
                    f"{row.get('error') or row.get('errors') or status}"
                ),
            )
            row.update(
                {
                    "status": "failed_published",
                    "result": str(target.resolve()),
                    "attempt": str(target.resolve()),
                    "failure": failure,
                }
            )

    _materialize_tree(
        transaction / "results" / "evaluations", paths.root / "evaluations"
    )
    write_experiment_reports(
        paths.root,
        config=first,
        queue_id=queue_id,
        queue_complete=finalize,
    )
    _write_json(
        transaction / "queue_transaction.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "queue_id": queue_id,
            "status": "published" if finalize else "publishing_queue",
            "result_root": str(paths.root.resolve()),
            "dataset_root": str(paths.dataset_root.resolve()),
            "updated_at": _now(),
        },
    )
    return results
