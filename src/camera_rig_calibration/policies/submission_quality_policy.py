"""Submission-facing calibration validity and deployment semantics.

This module does not change any calibration estimate.  It only prevents a
successfully executed diagnostic artifact from being presented as a deployable
calibration when the method's own quality/coverage evidence rejects that use.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


_INSTALLED = False
AP03_GOOD_RELATIVE_SCALE_STD = 0.05
AP03_MAX_RELATIVE_SCALE_STD = 0.10

_STATUS_FIELDS = (
    "execution_status",
    "solver_status",
    "calibration_status",
    "quality_status",
    "evaluation_status",
    "deployment_eligible",
    "deployment_eligible_cameras",
    "full_rig_result_available",
    "comparison_eligible",
    "diagnostic_partial",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ap03_full_coverage(metadata: dict[str, Any]) -> bool:
    missing = metadata.get("missing_static_cameras")
    if isinstance(missing, list):
        return len(missing) == 0
    status = str(metadata.get("status") or "")
    return status == "OK_FULL" or status == "SCALE_WEAK_CHECK_REQUIRED"


def ap03_quality_semantics(
    metadata: dict[str, Any],
    *,
    reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map native AP03 diagnostics to explicit calibration/deployment status.

    The 10% threshold is the existing AP03 weak-scale gate.  Execution remains
    independent from calibration validity: a weak/partial result is still an
    auditable completed artifact, but it is not deployment eligible.
    """

    relative_std = _finite_float(metadata.get("used_rel_std_scale"))
    scale_available = _finite_float(
        metadata.get("scale_m_per_colmap_unit")
    ) is not None
    full_coverage = _ap03_full_coverage(metadata)

    if relative_std is None:
        scale_quality = "unavailable"
    elif relative_std <= AP03_GOOD_RELATIVE_SCALE_STD:
        scale_quality = "good"
    elif relative_std <= AP03_MAX_RELATIVE_SCALE_STD:
        scale_quality = "warning_scale_dispersion"
    else:
        scale_quality = "poor_scale_dispersion"

    reconstruction_quality = str(
        (reconstruction or {}).get("quality_status") or "unavailable"
    )
    if scale_quality == "poor_scale_dispersion":
        quality_status = "poor_scale_dispersion"
    elif scale_quality == "unavailable":
        quality_status = "unavailable"
    elif scale_quality != "good" or reconstruction_quality not in {
        "good",
        "unavailable",
    }:
        quality_status = "warning_reconstruction_or_scale"
    else:
        quality_status = "good"

    if not scale_available:
        calibration_status = "unavailable"
    elif scale_quality == "poor_scale_dispersion":
        calibration_status = "rejected_by_quality_gate"
    elif not full_coverage:
        calibration_status = "partial_coverage"
    else:
        calibration_status = "available"

    deployment_eligible = calibration_status == "available"
    available = metadata.get("available_static_cameras")
    deployment_cameras = (
        sorted(str(item) for item in available)
        if deployment_eligible and isinstance(available, list)
        else []
    )
    return {
        "calibration_status": calibration_status,
        "quality_status": quality_status,
        "scale_quality_status": scale_quality,
        "deployment_eligible": deployment_eligible,
        "deployment_eligible_cameras": deployment_cameras,
    }


def _merge_method_status(
    result: dict[str, Any], status: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(result)
    for key in _STATUS_FIELDS:
        if key in status:
            merged[key] = status[key]

    calibration = str(merged.get("calibration_status") or "")
    if calibration and calibration != "available":
        merged["deployment_eligible"] = False
        merged["deployment_eligible_cameras"] = []
    elif merged.get("full_rig_result_available") is False:
        merged["deployment_eligible"] = False
        merged["deployment_eligible_cameras"] = []
    return merged


def _repair_anchor_exports(method_root: Path) -> None:
    """Keep anchor exports consistent with the owning RESULT validity."""

    result_path = method_root / "RESULT.json"
    result = _read_json(result_path)
    if not result:
        return
    calibration = str(
        result.get("calibration_status")
        or result.get("artifact_status")
        or "available"
    )
    deployment = bool(result.get("deployment_eligible", calibration == "available"))
    quality = str(result.get("quality_status") or "unknown")

    json_path = method_root / "camera_extrinsics_anchor.json"
    payload = _read_json(json_path)
    if payload:
        payload["calibration_status"] = calibration
        cameras = payload.get("cameras", [])
        if isinstance(cameras, list):
            for camera in cameras:
                if not isinstance(camera, dict):
                    continue
                if not deployment:
                    camera["deployment_eligible"] = False
                    camera["status"] = "available_diagnostic_only"
                    if camera.get("quality_status") in {None, "", "accepted"}:
                        camera["quality_status"] = quality
        _write_json(json_path, payload)
        yaml_path = method_root / "camera_extrinsics_anchor.yaml"
        yaml_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    csv_path = method_root / "camera_extrinsics_anchor.csv"
    if csv_path.is_file() and not deployment:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        if rows:
            for row in rows:
                if "deployment_eligible" in row:
                    row["deployment_eligible"] = "False"
                if "status" in row:
                    row["status"] = "available_diagnostic_only"
                if (
                    "quality_status" in row
                    and row.get("quality_status") in {None, "", "accepted"}
                ):
                    row["quality_status"] = quality
            temporary = csv_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(csv_path)


def _repair_ap03_derived_results(experiment_root: Path) -> None:
    for method in ("ap03_single", "ap03_multi"):
        for result_path in sorted(
            (experiment_root / "methods" / method).glob("*/RESULT.json")
        ):
            result = _read_json(result_path)
            metadata = result.get("metrics", {}).get("ap03_scale", {})
            if not isinstance(metadata, dict) or not metadata:
                continue
            semantics = ap03_quality_semantics(metadata)
            result.update(semantics)
            _write_json(result_path, result)
            _repair_anchor_exports(result_path.parent)


def _install_publication_status_propagation() -> None:
    from .. import publication

    original = publication._publish_success
    if getattr(original, "_rigcal_submission_quality", False):
        return

    def publish_success(source, **kwargs):
        target, outcome = original(source, **kwargs)
        method_id = target.parent.name
        status = publication._method_status(Path(source), method_id)
        result_path = target / "RESULT.json"
        result = _read_json(result_path)
        if result and status:
            result = _merge_method_status(result, status)
            _write_json(result_path, result)
            _repair_anchor_exports(target)
        return target, outcome

    publish_success._rigcal_submission_quality = True  # type: ignore[attr-defined]
    publication._publish_success = publish_success


def _install_ap03_derived_quality() -> None:
    from ..evaluation import ap03_derived, reporting

    original = ap03_derived.ensure_ap03_derived_results
    if getattr(original, "_rigcal_submission_quality", False):
        return

    def ensure_ap03_derived_results(experiment_root):
        outcomes = original(experiment_root)
        _repair_ap03_derived_results(Path(experiment_root))
        return outcomes

    ensure_ap03_derived_results._rigcal_submission_quality = True  # type: ignore[attr-defined]
    ap03_derived.ensure_ap03_derived_results = ensure_ap03_derived_results
    # reporting imports the function directly, so rebind that consumer too.
    reporting.ensure_ap03_derived_results = ensure_ap03_derived_results


def _install_anchor_status_preservation() -> None:
    from ..anchor_export import exporter
    from .. import anchor_export, publication

    original = exporter.export_method_anchor_poses
    if getattr(original, "_rigcal_submission_quality", False):
        return

    def export_method_anchor_poses(method_root):
        root = Path(method_root)
        before = _read_json(root / "RESULT.json")
        status = original(root)
        after = _read_json(root / "RESULT.json")
        if after:
            for key in (
                "calibration_status",
                "quality_status",
                "deployment_eligible",
                "deployment_eligible_cameras",
            ):
                if key in before:
                    after[key] = before[key]
            after = _merge_method_status(after, before)
            _write_json(root / "RESULT.json", after)
            _repair_anchor_exports(root)
        return status

    export_method_anchor_poses._rigcal_submission_quality = True  # type: ignore[attr-defined]
    exporter.export_method_anchor_poses = export_method_anchor_poses
    anchor_export.export_method_anchor_poses = export_method_anchor_poses
    # publication imported this symbol directly.
    publication.export_method_anchor_poses = export_method_anchor_poses


def install_submission_quality_policy() -> None:
    """Install reporting-only validity guards; calibration math is untouched."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_anchor_status_preservation()
    _install_publication_status_propagation()
    _install_ap03_derived_quality()
    _INSTALLED = True


__all__ = [
    "AP03_GOOD_RELATIVE_SCALE_STD",
    "AP03_MAX_RELATIVE_SCALE_STD",
    "ap03_quality_semantics",
    "install_submission_quality_policy",
]
