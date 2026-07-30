from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import load_config
from .adapters import load_camera_poses, resolve_method_anchor
from .geometry import invert_transform, pose_payload, rotation_error_deg


EXPORT_CONTRACT = "rigcal_anchor_export_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_text(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _hash_files(paths: list[Path], extra: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(extra, sort_keys=True, default=str).encode("utf-8")
    )
    for path in sorted(paths):
        digest.update(path.as_posix().encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _pairwise_invariance(
    native: dict[str, np.ndarray],
    anchored: dict[str, np.ndarray],
) -> dict[str, Any]:
    maximum_translation = 0.0
    maximum_rotation = 0.0
    cameras = sorted(set(native) & set(anchored))
    for first_index, first in enumerate(cameras):
        for second in cameras[first_index + 1 :]:
            native_pair = invert_transform(native[first]) @ native[second]
            anchor_pair = invert_transform(anchored[first]) @ anchored[second]
            maximum_translation = max(
                maximum_translation,
                float(np.linalg.norm(native_pair[:3, 3] - anchor_pair[:3, 3])),
            )
            maximum_rotation = max(
                maximum_rotation,
                rotation_error_deg(native_pair, anchor_pair),
            )
    passed = maximum_translation <= 1e-8 and maximum_rotation <= 1e-5
    return {
        "passed": passed,
        "maximum_translation_difference_m": maximum_translation,
        "maximum_rotation_difference_deg": maximum_rotation,
        "camera_pair_count": len(cameras) * (len(cameras) - 1) // 2,
    }


def _camera_row(
    *,
    camera_id: str,
    transform: np.ndarray,
    method: str,
    label: str,
    anchor_marker_id: int,
    status: str,
    estimate_status: str = "available",
    quality_status: str = "accepted",
    deployment_eligible: bool = True,
    evaluation_status: str = "available",
) -> dict[str, Any]:
    parent = f"evaluation_anchor_marker_{anchor_marker_id}"
    child = f"{camera_id}_optical_frame"
    return {
        "method": method,
        "label": label,
        "anchor_marker_id": anchor_marker_id,
        "parent_frame": parent,
        "child_frame": child,
        "camera_id": camera_id,
        "status": status,
        "estimate_status": estimate_status,
        "quality_status": quality_status,
        "deployment_eligible": deployment_eligible,
        "evaluation_status": evaluation_status,
        "source": "method_primary_result",
        **pose_payload(transform),
    }


def _write_csv(path: Path, cameras: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "label",
        "anchor_marker_id",
        "parent_frame",
        "child_frame",
        "camera_id",
        "status",
        "estimate_status",
        "quality_status",
        "deployment_eligible",
        "evaluation_status",
        "source",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "qx",
        "qy",
        "qz",
        "qw",
        *[f"m{row}{column}" for row in range(4) for column in range(4)],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for camera in cameras:
            matrix = camera["matrix"]
            flattened = {
                f"m{row}{column}": matrix[row][column]
                for row in range(4)
                for column in range(4)
            }
            writer.writerow(
                {
                    key: value
                    for key, value in {**camera, **flattened}.items()
                    if key in fields
                }
            )
    temporary.replace(path)


def _config_for_result(method_root: Path):
    config_path = method_root / "provenance" / "resolved_config.yaml"
    return load_config(config_path) if config_path.is_file() else None


def export_method_anchor_poses(method_root: Path) -> dict[str, Any]:
    method_root = method_root.resolve()
    result_path = method_root / "RESULT.json"
    result = _read_json(result_path)
    method = str(result.get("method") or method_root.parent.name)
    label = str(result.get("label") or method_root.name)
    config = _config_for_result(method_root)
    if config is None or not isinstance(config.evaluation.anchor_marker_id, int):
        status = {
            "code": "ANCHOR_NOT_FROZEN",
            "available": False,
            "message": "The published result has no frozen integer evaluation anchor.",
        }
        result.update(
            {
                "calibration_status": result.get("artifact_status", "available"),
                "anchor_export_status": status["code"],
                "anchor_export_available": False,
            }
        )
        _write_json(result_path, result)
        return status
    anchor = int(config.evaluation.anchor_marker_id)
    source_files = [
        method_root / "camera_extrinsics.csv",
        method_root / "diagnostics" / "preflight" / "accepted_observations.csv",
        method_root
        / "diagnostics"
        / "method"
        / "graph_ba"
        / "with_moving"
        / "optimized_marker_poses_ref_marker.csv",
        method_root
        / "diagnostics"
        / "method"
        / "scale_multi"
        / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv",
        method_root
        / "diagnostics"
        / "method"
        / "scale_multi"
        / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json",
        method_root.parents[2]
        / "observations"
        / "SELECTION_CANDIDATES.json",
        method_root / "provenance" / "derived_result.json",
    ]
    provenance = _read_json(
        method_root / "provenance" / "derived_result.json"
    )
    experiment_root = method_root.parents[2]
    for key in (
        "shared_anchor_geometry",
        "scale_metadata",
        "camera_pose_source",
    ):
        value = provenance.get(key)
        if value:
            source_files.append(experiment_root / str(value))
    fingerprint = _hash_files(
        source_files,
        {
            "contract": EXPORT_CONTRACT,
            "method": method,
            "label": label,
            "anchor": anchor,
        },
    )
    existing = _read_json(method_root / "camera_extrinsics_anchor.json")
    if existing.get("fingerprint") == fingerprint:
        expected_calibration = result.get("artifact_status", "available")
        expected_evaluation = result.get("evaluation_status", "not_run")
        if (
            existing.get("calibration_status") != expected_calibration
            or existing.get("evaluation_status") != expected_evaluation
        ):
            existing["calibration_status"] = expected_calibration
            existing["evaluation_status"] = expected_evaluation
            _write_json(
                method_root / "camera_extrinsics_anchor.json", existing
            )
            _atomic_text(
                method_root / "camera_extrinsics_anchor.yaml",
                yaml.safe_dump(
                    existing, sort_keys=False, allow_unicode=True
                ),
            )
        existing_status = dict(
            existing.get("anchor_export_status") or {}
        )
        result.update(
            {
                "calibration_status": result.get(
                    "artifact_status", "available"
                ),
                "evaluation_status": result.get(
                    "evaluation_status", "not_run"
                ),
                "anchor_export_status": existing_status.get(
                    "code", "ANCHOR_NOT_AVAILABLE"
                ),
                "anchor_export_available": bool(
                    existing_status.get("available")
                ),
                "anchor_marker_id": anchor,
                "camera_extrinsics_anchor": (
                    "camera_extrinsics_anchor.csv"
                ),
                "camera_extrinsics_anchor_json": (
                    "camera_extrinsics_anchor.json"
                ),
                "camera_extrinsics_anchor_yaml": (
                    "camera_extrinsics_anchor.yaml"
                ),
                "anchor_alignment": "diagnostics/anchor_alignment.json",
                "anchor_export_warnings": existing_status.get(
                    "warnings", []
                ),
            }
        )
        _write_json(result_path, result)
        return existing_status

    native = load_camera_poses(method_root)
    resolution = resolve_method_anchor(
        method_root, config, method, anchor, native
    )
    anchored: dict[str, np.ndarray] = {}
    invariance: dict[str, Any] = {
        "passed": False,
        "reason": "anchor transform unavailable",
    }
    code = resolution.code
    warnings = list(resolution.warnings)
    if resolution.available and resolution.transform_method_anchor is not None:
        anchor_method = invert_transform(resolution.transform_method_anchor)
        anchored = {
            camera: anchor_method @ transform
            for camera, transform in native.items()
        }
        invariance = _pairwise_invariance(native, anchored)
        if not invariance["passed"]:
            anchored = {}
            code = "INTERNAL_INVARIANCE_FAILED"
            warnings.append(
                "Anchor conversion changed a camera-to-camera transform; export was suppressed."
            )
    expected_cameras = {camera.id for camera in config.static_cameras}
    missing = sorted(expected_cameras - set(anchored))
    if anchored and missing and code == "OK":
        code = "PARTIAL_CAMERA_COVERAGE"
        warnings.append(
            "No anchor-relative pose is available for: " + ", ".join(missing)
        )
    camera_statuses = result.get("camera_statuses", {})
    camera_rows = []
    for camera in sorted(anchored):
        own_status = (
            camera_statuses.get(camera, {})
            if isinstance(camera_statuses, dict)
            else {}
        )
        deployment = bool(own_status.get("deployment_eligible", True))
        camera_rows.append(
            _camera_row(
                camera_id=camera,
                transform=anchored[camera],
                method=method,
                label=label,
                anchor_marker_id=anchor,
                status=(
                    "available"
                    if deployment
                    else "available_diagnostic_only"
                ),
                estimate_status=str(
                    own_status.get("estimate_status", "available")
                ),
                quality_status=str(
                    own_status.get("quality_status", "accepted")
                ),
                deployment_eligible=deployment,
                evaluation_status=str(
                    own_status.get("evaluation_status", "available")
                ),
            )
        )
    selection = _read_json(
        method_root.parents[2]
        / "observations"
        / "SELECTION_CANDIDATES.json"
    )
    selection_evidence = selection.get("evaluation_anchor", {})
    manual_warning_confirmed = bool(
        selection_evidence.get("warning_confirmed")
    )
    if manual_warning_confirmed:
        warnings.append(
            "The common anchor was a deliberately confirmed manual candidate "
            "that was not compatible with every queued method during preflight."
        )
    if resolution.available and not camera_rows:
        code = "CAMERA_POSE_NOT_AVAILABLE"
        warnings.append(
            "The method anchor was reconstructed, but no primary static-camera "
            "pose is available for export."
        )
    alignment_details = dict(resolution.diagnostics)
    if resolution.transform_method_anchor is not None:
        alignment_details["transform_method_anchor"] = pose_payload(
            resolution.transform_method_anchor
        )
        alignment_details["transform_anchor_method"] = pose_payload(
            invert_transform(resolution.transform_method_anchor)
        )
    status = {
        "code": code,
        "available": bool(camera_rows),
        "camera_count": len(camera_rows),
        "expected_camera_count": len(expected_cameras),
        "missing_cameras": missing,
        "warnings": warnings,
        "manual_warning_confirmed": manual_warning_confirmed,
    }
    payload = {
        "schema_version": 1,
        "layout_version": 2,
        "contract": EXPORT_CONTRACT,
        "fingerprint": fingerprint,
        "method": method,
        "label": label,
        "anchor_marker_id": anchor,
        "anchor_selection": {
            "mode": selection_evidence.get(
                "selection_mode",
                config.evaluation.anchor_selection_mode,
            ),
            "reason": selection_evidence.get("reason"),
            "warning_confirmed": manual_warning_confirmed,
        },
        "parent_frame": f"evaluation_anchor_marker_{anchor}",
        "transform_convention": (
            "T_parent_child; p_parent = T_parent_child @ p_child"
        ),
        "camera_optical_convention": "x right, y down, z forward",
        "rpy_convention": "radians; R = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "quaternion_order": "qx,qy,qz,qw",
        "calibration_status": result.get("artifact_status", "available"),
        "evaluation_status": result.get("evaluation_status", "not_run"),
        "anchor_export_status": status,
        "anchor_alignment": alignment_details,
        "pairwise_invariance": invariance,
        "cameras": camera_rows,
    }
    _write_json(method_root / "camera_extrinsics_anchor.json", payload)
    _atomic_text(
        method_root / "camera_extrinsics_anchor.yaml",
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    _write_csv(method_root / "camera_extrinsics_anchor.csv", camera_rows)
    alignment = {
        "schema_version": 1,
        "contract": EXPORT_CONTRACT,
        "fingerprint": fingerprint,
        "method": method,
        "label": label,
        "anchor_marker_id": anchor,
        "status": status,
        "alignment": alignment_details,
        "pairwise_invariance": invariance,
    }
    _write_json(
        method_root / "diagnostics" / "anchor_alignment.json",
        alignment,
    )
    result.update(
        {
            "calibration_status": result.get("artifact_status", "available"),
            "evaluation_status": result.get("evaluation_status", "not_run"),
            "anchor_export_status": code,
            "anchor_export_available": bool(camera_rows),
            "anchor_marker_id": anchor,
            "camera_extrinsics_anchor": "camera_extrinsics_anchor.csv",
            "camera_extrinsics_anchor_json": "camera_extrinsics_anchor.json",
            "camera_extrinsics_anchor_yaml": "camera_extrinsics_anchor.yaml",
            "anchor_alignment": "diagnostics/anchor_alignment.json",
            "anchor_export_warnings": warnings,
        }
    )
    _write_json(result_path, result)
    return status


def ensure_experiment_anchor_exports(
    experiment_root: Path,
) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    methods_root = experiment_root / "methods"
    if not methods_root.is_dir():
        return outcomes
    for result_path in sorted(methods_root.glob("*/*/RESULT.json")):
        key = result_path.parent.relative_to(methods_root).as_posix()
        try:
            outcomes[key] = export_method_anchor_poses(result_path.parent)
        except Exception as exc:
            outcomes[key] = {
                "code": "INTERNAL_EXPORT_ERROR",
                "available": False,
                "warnings": [str(exc)],
            }
    return outcomes
