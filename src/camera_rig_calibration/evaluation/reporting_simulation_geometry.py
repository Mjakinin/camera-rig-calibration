"""Focused scientific reporting responsibility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.geometry import rotation_to_quaternion
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import (
    ensure_simulation_ground_truth,
    resolve_simulation_ground_truth,
)

from ..methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    invT,
    make_T,
    rot_error_deg,
    rpy_to_R,
    rvec_to_R,
)

from .reporting_configuration import (
    _baseline_contract,
    _baseline_contract_text,
    _config_text,
)
from .reporting_core import (
    PoseRecord,
    _angle_between,
    _fmt,
    _maximum,
    _mean,
    _median,
    _pose_columns,
    _pose_from_row,
    _read_json,
    _text_table,
    load_pose_records,
)
from .reporting_diagnostics import (
    _scale_comparison_rows,
    _scale_comparison_text,
)
from .reporting_bindings import current_reporting_bindings

def _repository_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return path.resolve()


def _matrix(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def _simulation_gt_maps(
    payload: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    cameras = {
        str(name): _matrix(transform)
        for name, transform in payload.get("static_cameras", {}).items()
    }
    markers = {
        int(marker): _matrix(value["T_world_marker_opencv"])
        for marker, value in payload.get("markers", {}).items()
        if isinstance(value, dict) and value.get("T_world_marker_opencv")
    }
    return cameras, markers


def _simulation_pairwise(
    method: str,
    label: str,
    estimated: dict[str, PoseRecord],
    gt_cameras: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common = sorted(set(estimated) & set(gt_cameras))
    for first, second in combinations(common, 2):
        est = invT(estimated[first].transform) @ estimated[second].transform
        gt = invT(gt_cameras[first]) @ gt_cameras[second]
        est_translation = est[:3, 3]
        gt_translation = gt[:3, 3]
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "pair": f"{first}-{second}",
            "from_camera": first,
            "to_camera": second,
            "translation_error_cm": 100.0
            * float(np.linalg.norm(est_translation - gt_translation)),
            "rotation_error_deg": float(rot_error_deg(est, gt)),
            "gt_baseline_m": float(np.linalg.norm(gt_translation)),
            "estimated_baseline_m": float(np.linalg.norm(est_translation)),
            "baseline_error_cm": 100.0
            * abs(
                float(np.linalg.norm(est_translation))
                - float(np.linalg.norm(gt_translation))
            ),
            "direction_error_deg": _angle_between(
                est_translation, gt_translation
            ),
        }
        row.update(_pose_columns("gt_", gt))
        row.update(_pose_columns("estimated_", est))
        rows.append(row)
    return rows


def _anchor_camera_gt_rows(
    method: str,
    label: str,
    anchor_payload: dict[str, Any],
    *,
    anchor_marker_id: int,
    gt_cameras: dict[str, np.ndarray],
    gt_markers: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    gt_anchor = gt_markers.get(anchor_marker_id)
    if gt_anchor is None:
        return []
    anchor_world = invT(gt_anchor)
    rows: list[dict[str, Any]] = []
    for camera in anchor_payload.get("cameras", []):
        if not isinstance(camera, dict):
            continue
        camera_id = str(camera.get("camera_id", ""))
        if camera_id not in gt_cameras or camera.get("matrix") is None:
            continue
        estimated = _matrix(camera["matrix"])
        ground_truth = anchor_world @ gt_cameras[camera_id]
        delta = estimated[:3, 3] - ground_truth[:3, 3]
        estimated_quaternion = rotation_to_quaternion(estimated[:3, :3])
        gt_quaternion = rotation_to_quaternion(ground_truth[:3, :3])
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "anchor_marker_id": anchor_marker_id,
            "camera": camera_id,
            "translation_error_cm": 100.0
            * float(
                np.linalg.norm(
                    estimated[:3, 3] - ground_truth[:3, 3]
                )
            ),
            "rotation_error_deg": float(
                rot_error_deg(estimated, ground_truth)
            ),
            "delta_x_m": float(delta[0]),
            "delta_y_m": float(delta[1]),
            "delta_z_m": float(delta[2]),
            "estimated_qx": estimated_quaternion[0],
            "estimated_qy": estimated_quaternion[1],
            "estimated_qz": estimated_quaternion[2],
            "estimated_qw": estimated_quaternion[3],
            "gt_qx": gt_quaternion[0],
            "gt_qy": gt_quaternion[1],
            "gt_qz": gt_quaternion[2],
            "gt_qw": gt_quaternion[3],
            "evaluation": (
                "direct_anchor_relative_posthoc_gt_no_fit_no_scale"
            ),
        }
        row.update(_pose_columns("estimated_", estimated))
        row.update(_pose_columns("gt_", ground_truth))
        rows.append(row)
    return rows


def _anchor_pose_records(
    anchor_payload: dict[str, Any],
) -> dict[str, PoseRecord]:
    records: dict[str, PoseRecord] = {}
    for camera in anchor_payload.get("cameras", []):
        if not isinstance(camera, dict) or camera.get("matrix") is None:
            continue
        camera_id = str(camera.get("camera_id", "")).strip()
        if not camera_id:
            continue
        try:
            transform = _matrix(camera["matrix"])
        except (TypeError, ValueError):
            continue
        records[camera_id] = PoseRecord(
            entity_id=camera_id,
            transform=transform,
            source="camera_extrinsics_anchor.json",
            reference_frame=str(anchor_payload.get("parent_frame", "")),
            transform_convention=str(
                anchor_payload.get("transform_convention", "")
            ),
        )
    return records


def _ground_truth_anchor_records(
    *,
    anchor_marker_id: int,
    gt_cameras: dict[str, np.ndarray],
    gt_markers: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    anchor = gt_markers.get(anchor_marker_id)
    if anchor is None:
        return {}
    anchor_world = invT(anchor)
    return {
        camera: anchor_world @ transform
        for camera, transform in gt_cameras.items()
    }


def _pose_alignment(
    estimated: dict[str, PoseRecord],
    ground_truth: dict[str, np.ndarray],
) -> np.ndarray:
    common = sorted(set(estimated) & set(ground_truth))
    if len(common) < 3:
        raise RuntimeError("At least three common camera poses are required")
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for camera in common:
        accumulator += (
            ground_truth[camera][:3, :3]
            @ estimated[camera].transform[:3, :3].T
        )
    left, _, right = np.linalg.svd(accumulator)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translations = [
        ground_truth[camera][:3, 3]
        - rotation @ estimated[camera].transform[:3, 3]
        for camera in common
    ]
    return make_T(rotation, np.mean(np.vstack(translations), axis=0))


def _apply_alignment(alignment: np.ndarray, transform: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = alignment[:3, :3] @ transform[:3, :3]
    output[:3, 3] = (
        alignment[:3, :3] @ transform[:3, 3] + alignment[:3, 3]
    )
    return output


def _camera_map_rows(
    method: str,
    label: str,
    estimated: dict[str, PoseRecord],
    ground_truth: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    alignment = _pose_alignment(estimated, ground_truth)
    rows: list[dict[str, Any]] = []
    for camera in sorted(set(estimated) & set(ground_truth)):
        aligned = _apply_alignment(alignment, estimated[camera].transform)
        gt = ground_truth[camera]
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "camera": camera,
            "alignment": "best_fit_SE3_all_static_cameras_no_scale",
            "translation_error_cm": 100.0
            * float(np.linalg.norm(aligned[:3, 3] - gt[:3, 3])),
            "rotation_error_deg": float(rot_error_deg(aligned, gt)),
        }
        row.update(_pose_columns("aligned_estimated_", aligned))
        row.update(_pose_columns("gt_", gt))
        rows.append(row)
    return rows


def _point_alignment(
    source: list[np.ndarray], destination: list[np.ndarray]
) -> np.ndarray:
    first = np.asarray(source, dtype=np.float64)
    second = np.asarray(destination, dtype=np.float64)
    if first.shape != second.shape or first.shape[0] < 3:
        raise RuntimeError("At least three matching 3D entities are required")
    first_mean = first.mean(axis=0)
    second_mean = second.mean(axis=0)
    covariance = (first - first_mean).T @ (second - second_mean)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1, :] *= -1
        rotation = right.T @ left.T
    translation = second_mean - rotation @ first_mean
    return make_T(rotation, translation)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_translation_error_cm": _mean(
            row.get("translation_error_cm") for row in rows
        ),
        "median_translation_error_cm": _median(
            row.get("translation_error_cm") for row in rows
        ),
        "max_translation_error_cm": _maximum(
            row.get("translation_error_cm") for row in rows
        ),
        "mean_rotation_error_deg": _mean(
            row.get("rotation_error_deg") for row in rows
        ),
        "median_rotation_error_deg": _median(
            row.get("rotation_error_deg") for row in rows
        ),
        "max_rotation_error_deg": _maximum(
            row.get("rotation_error_deg") for row in rows
        ),
    }


def _simulation_primary_text(
    experiment: str,
    parameters: dict[str, Any],
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    method_payloads: list[dict[str, Any]],
    evaluation_anchor: dict[str, Any],
) -> str:
    _baseline_contract = current_reporting_bindings().baseline_contract
    width = 138
    baseline_contract = _baseline_contract(
        category="simulation",
        method_payloads=method_payloads,
        evaluation_anchor=evaluation_anchor,
    )
    lines = [
        "SIMULATION CALIBRATION RESULTS — CAMERA-TO-CAMERA VS GROUND TRUTH",
        "=" * width,
        "",
        f"Experiment: {experiment}",
        "Simulation parameters: "
        + ", ".join(f"{key}={value}" for key, value in parameters.items()),
        (
            "Common evaluation anchor: marker "
            f"{evaluation_anchor.get('selected', '-')} "
            f"(configured {evaluation_anchor.get('configured', '-')}; "
            "frozen during preflight)"
        ),
        f"Anchor reason: {evaluation_anchor.get('reason', '-')}",
        "",
        _baseline_contract_text(baseline_contract),
        "METHOD / VARIANT SUMMARY",
        "-" * width,
    ]
    payload_by_key = {
        (str(item.get("method")), str(item.get("label"))): item
        for item in method_payloads
    }
    summary_table = []
    for item in summaries:
        payload = payload_by_key.get((item["method"], item["label"]), {})
        summary_table.append(
            [
                item["method"],
                item["label"],
                payload.get("artifact_status", "available"),
                payload.get("quality_status", "-"),
                item["count"],
                _fmt(item["mean_translation_error_cm"]),
                _fmt(item["mean_rotation_error_deg"]),
                _fmt(item["max_translation_error_cm"]),
                _fmt(item["max_rotation_error_deg"]),
                _config_text(payload.get("config_summary", {})),
            ]
        )
    lines.extend(
        [
            _text_table(
                [
                    "Method",
                    "Variant",
                    "Artifact",
                    "Quality",
                    "Pairs",
                    "mean t [cm]",
                    "mean r [deg]",
                    "max t [cm]",
                    "max r [deg]",
                    "Key configuration",
                ],
                summary_table,
            ),
            "",
            "SCALE COMPARISON",
            "-" * width,
            _scale_comparison_text(
                _scale_comparison_rows(method_payloads)
            ),
            "",
            "DETAILED CAMERA-PAIR RESULTS",
            "-" * width,
        ]
    )
    for method, label in sorted(
        {
            (str(item["method"]), str(item["label"]))
            for item in summaries
        }
    ):
        selected = [
            row
            for row in rows
            if row["method"] == method and row["label"] == label
        ]
        lines.extend(
            [
                "",
                f"{method} / {label}",
                (
                    _text_table(
                    [
                        "Pair",
                        "t err [cm]",
                        "r err [deg]",
                        "GT base [m]",
                        "Est base [m]",
                        "base err [cm]",
                        "dir err [deg]",
                    ],
                    [
                        [
                            row["pair"],
                            _fmt(row["translation_error_cm"]),
                            _fmt(row["rotation_error_deg"]),
                            _fmt(row["gt_baseline_m"]),
                            _fmt(row["estimated_baseline_m"]),
                            _fmt(row["baseline_error_cm"]),
                            _fmt(row["direction_error_deg"]),
                        ]
                        for row in selected
                    ],
                    )
                    if selected
                    else (
                        "Evaluation unavailable: the exact direct "
                        "anchor-relative camera set is incomplete."
                    )
                ),
            ]
        )
    lines.extend(
        [
            "",
            "METHOD REPORTS AND DETAIL ARTIFACTS",
            "-" * width,
        ]
    )
    for payload in method_payloads:
        method = str(payload.get("method", "-"))
        label = str(payload.get("label", "-"))
        prefix = Path("methods") / method / label
        lines.append(f"- {prefix / 'RESULT.txt'}")
        for path in payload.get("detail_artifacts", []):
            lines.append(f"  - {prefix / str(path)}")
    lines.append("")
    return "\n".join(lines)


def _camera_map_text(
    experiment: str,
    rows: list[dict[str, Any]],
) -> str:
    width = 118
    lines = [
        "SECONDARY STATIC-CAMERA MAP VS GROUND TRUTH",
        "=" * width,
        "",
        f"Experiment: {experiment}",
    ]
    for method, label in sorted(
        {(row["method"], row["label"]) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if row["method"] == method and row["label"] == label
        ]
        summary = _summary(selected)
        lines.extend(
            [
                "",
                f"{method} / {label}",
                "-" * width,
                (
                    f"Summary: mean {_fmt(summary['mean_translation_error_cm'])} "
                    f"cm / {_fmt(summary['mean_rotation_error_deg'])} deg; "
                    f"max {_fmt(summary['max_translation_error_cm'])} cm / "
                    f"{_fmt(summary['max_rotation_error_deg'])} deg"
                ),
                _text_table(
                    ["Camera", "t error [cm]", "r error [deg]"],
                    [
                        [
                            row["camera"],
                            _fmt(row["translation_error_cm"]),
                            _fmt(row["rotation_error_deg"]),
                        ]
                        for row in selected
                    ],
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _ap02_marker_map(
    result_root: Path,
    gt_cameras: dict[str, np.ndarray],
    gt_markers: dict[int, np.ndarray],
) -> tuple[dict[str, Any], str]:
    _read_json = current_reporting_bindings().read_json
    payload = _read_json(result_root / "RESULT.json")
    marker_file = (
        result_root
        / "diagnostics"
        / "method"
        / "graph_ba"
        / "with_moving"
        / "optimized_marker_poses_ref_marker.csv"
    )
    estimated_markers_raw = load_pose_records(marker_file)
    estimated_markers = {
        int(marker): record
        for marker, record in estimated_markers_raw.items()
        if str(marker).lstrip("-").isdigit()
    }
    estimated_cameras = load_pose_records(
        result_root / "camera_extrinsics.csv"
    )
    reference = payload.get("reference_marker_id")
    if reference is None:
        match = re.search(
            r"(\d+)", str(payload.get("reference_frame", ""))
        )
        reference = int(match.group(1)) if match else None
    if reference is None or int(reference) not in gt_markers:
        unavailable = {
            "method": "ap02",
            "label": payload.get("label", result_root.name),
            "status": "unavailable",
            "reason": "Reference marker is missing from simulation GT.",
        }
        return unavailable, (
            "SECONDARY AP02 MARKER-MAP RESULTS\n"
            "=================================\n\n"
            f"Unavailable: {unavailable['reason']}\n"
        )
    reference = int(reference)
    world_from_reference = gt_markers[reference]
    reference_from_world = invT(world_from_reference)
    direct_rows: list[dict[str, Any]] = []
    for marker in sorted(set(estimated_markers) & set(gt_markers)):
        estimated = estimated_markers[marker].transform
        gt = reference_from_world @ gt_markers[marker]
        translation_error = 100.0 * float(
            np.linalg.norm(estimated[:3, 3] - gt[:3, 3])
        )
        rotation_error = float(rot_error_deg(estimated, gt))
        if marker == reference:
            # The selected marker defines both coordinate systems. Avoid
            # displaying floating-point roundoff as a physical residual.
            translation_error = 0.0
            rotation_error = 0.0
        direct_rows.append(
            {
                "marker_id": marker,
                "is_reference_marker": marker == reference,
                "translation_error_cm": translation_error,
                "rotation_error_deg": rotation_error,
                **_pose_columns("estimated_", estimated),
                **_pose_columns("gt_", gt),
            }
        )

    source_points: list[np.ndarray] = []
    destination_points: list[np.ndarray] = []
    for camera in sorted(set(estimated_cameras) & set(gt_cameras)):
        source_points.append(estimated_cameras[camera].transform[:3, 3])
        destination_points.append(gt_cameras[camera][:3, 3])
    for marker in sorted(set(estimated_markers) & set(gt_markers)):
        if marker == reference:
            continue
        source_points.append(estimated_markers[marker].transform[:3, 3])
        destination_points.append(gt_markers[marker][:3, 3])
    alignment = _point_alignment(source_points, destination_points)
    fitted_rows: list[dict[str, Any]] = []
    for marker in sorted(set(estimated_markers) & set(gt_markers)):
        aligned = _apply_alignment(
            alignment, estimated_markers[marker].transform
        )
        gt = gt_markers[marker]
        fitted_rows.append(
            {
                "marker_id": marker,
                "used_for_alignment": marker != reference,
                "translation_error_cm": 100.0
                * float(np.linalg.norm(aligned[:3, 3] - gt[:3, 3])),
                "rotation_error_deg": float(rot_error_deg(aligned, gt)),
                **_pose_columns("aligned_estimated_", aligned),
                **_pose_columns("gt_", gt),
            }
        )
    result = {
        "method": "ap02",
        "label": payload.get("label", result_root.name),
        "status": "available",
        "reference_marker_id": reference,
        "direct_reference_frame": {
            "alignment": "none",
            "summary": _summary(direct_rows),
            "rows": direct_rows,
        },
        "best_fit_diagnostic": {
            "alignment": (
                "best_fit_SE3_static_cameras_and_nonreference_markers_no_scale"
            ),
            "reference_marker_held_out": True,
            "summary": _summary(fitted_rows),
            "rows": fitted_rows,
        },
    }
    width = 126
    lines = [
        "SECONDARY AP02 MARKER-MAP VS GROUND TRUTH",
        "=" * width,
        "",
        f"Method / variant: ap02 / {result['label']}",
        f"Reference marker: {reference}",
        "",
        "A. DIRECT REFERENCE-MARKER FRAME (NO ALIGNMENT)",
        "-" * width,
        (
            f"Summary: mean {_fmt(result['direct_reference_frame']['summary']['mean_translation_error_cm'])} "
            f"cm / {_fmt(result['direct_reference_frame']['summary']['mean_rotation_error_deg'])} deg; "
            f"max {_fmt(result['direct_reference_frame']['summary']['max_translation_error_cm'])} "
            f"cm / {_fmt(result['direct_reference_frame']['summary']['max_rotation_error_deg'])} deg"
        ),
        _text_table(
            ["Marker", "Reference", "t error [cm]", "r error [deg]"],
            [
                [
                    row["marker_id"],
                    "yes" if row["is_reference_marker"] else "no",
                    _fmt(row["translation_error_cm"]),
                    _fmt(row["rotation_error_deg"]),
                ]
                for row in direct_rows
            ],
        ),
        "",
        "B. BEST-FIT SE(3) DIAGNOSTIC (NO SCALE)",
        "-" * width,
        (
            f"Summary: mean {_fmt(result['best_fit_diagnostic']['summary']['mean_translation_error_cm'])} "
            f"cm / {_fmt(result['best_fit_diagnostic']['summary']['mean_rotation_error_deg'])} deg; "
            f"max {_fmt(result['best_fit_diagnostic']['summary']['max_translation_error_cm'])} "
            f"cm / {_fmt(result['best_fit_diagnostic']['summary']['max_rotation_error_deg'])} deg"
        ),
        _text_table(
            ["Marker", "Used in fit", "t error [cm]", "r error [deg]"],
            [
                [
                    row["marker_id"],
                    "yes" if row["used_for_alignment"] else "held out",
                    _fmt(row["translation_error_cm"]),
                    _fmt(row["rotation_error_deg"]),
                ]
                for row in fitted_rows
            ],
        ),
        "",
    ]
    return result, "\n".join(lines)


def _latest_marker_report(experiment_root: Path) -> tuple[str, Path | None]:
    reconciled = (
        experiment_root
        / "evaluations"
        / "method_anchors_reconciled"
        / "REAL_DATA_MARKER_CONSISTENCY.txt"
    )
    if reconciled.is_file():
        return reconciled.read_text(encoding="utf-8"), reconciled
    candidates = sorted(
        (experiment_root / "evaluations").rglob(
            "REAL_DATA_MARKER_CONSISTENCY.txt"
        ),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return (
            "COMMON MARKER CONSISTENCY\n"
            "=========================\n\n"
            "Unavailable: no completed common-anchor evaluation was published. "
            "The camera poses above remain authoritative method outputs.\n",
            None,
        )
    return candidates[0].read_text(encoding="utf-8"), candidates[0]


def _real_variant_disagreement(
    pair_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare estimates with each other without implying real-world accuracy."""
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in pair_rows:
        key = (str(row.get("method")), str(row.get("label")))
        grouped.setdefault(key, {})[str(row.get("pair"))] = row
    detailed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for first, second in combinations(sorted(grouped), 2):
        common = sorted(set(grouped[first]) & set(grouped[second]))
        comparison_rows: list[dict[str, Any]] = []
        for pair in common:
            first_transform = _pose_from_row(grouped[first][pair])
            second_transform = _pose_from_row(grouped[second][pair])
            row = {
                "first_method": first[0],
                "first_label": first[1],
                "second_method": second[0],
                "second_label": second[1],
                "pair": pair,
                "translation_delta_cm": 100.0
                * float(
                    np.linalg.norm(
                        first_transform[:3, 3]
                        - second_transform[:3, 3]
                    )
                ),
                "rotation_delta_deg": float(
                    rot_error_deg(first_transform, second_transform)
                ),
                "baseline_delta_cm": 100.0
                * abs(
                    float(np.linalg.norm(first_transform[:3, 3]))
                    - float(np.linalg.norm(second_transform[:3, 3]))
                ),
            }
            comparison_rows.append(row)
            detailed.append(row)
        summaries.append(
            {
                "first_method": first[0],
                "first_label": first[1],
                "second_method": second[0],
                "second_label": second[1],
                "pair_count": len(comparison_rows),
                "mean_translation_delta_cm": _mean(
                    row["translation_delta_cm"]
                    for row in comparison_rows
                ),
                "max_translation_delta_cm": _maximum(
                    row["translation_delta_cm"]
                    for row in comparison_rows
                ),
                "mean_rotation_delta_deg": _mean(
                    row["rotation_delta_deg"]
                    for row in comparison_rows
                ),
                "max_rotation_delta_deg": _maximum(
                    row["rotation_delta_deg"]
                    for row in comparison_rows
                ),
                "mean_baseline_delta_cm": _mean(
                    row["baseline_delta_cm"]
                    for row in comparison_rows
                ),
            }
        )
    return summaries, detailed



__all__ = [
    '_repository_root',
    '_matrix',
    '_simulation_gt_maps',
    '_simulation_pairwise',
    '_anchor_camera_gt_rows',
    '_anchor_pose_records',
    '_ground_truth_anchor_records',
    '_pose_alignment',
    '_apply_alignment',
    '_camera_map_rows',
    '_point_alignment',
    '_summary',
    '_simulation_primary_text',
    '_camera_map_text',
    '_ap02_marker_map',
    '_latest_marker_report',
    '_real_variant_disagreement',
]
