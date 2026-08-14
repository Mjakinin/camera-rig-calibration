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
    _config_text,
    _configuration_summary,
)
from .reporting_core import (
    PoseRecord,
    _fmt,
    _now,
    _read_json,
    _text_table,
    _write_csv,
    _write_json,
    _write_text,
    load_pose_records,
    pairwise_rows,
)
from .reporting_diagnostics import (
    _method_diagnostics,
)
from .reporting_quality import (
    _quality_details,
)
from .reporting_simulation_geometry import (
    _repository_root,
)
from .reporting_bindings import current_reporting_bindings


def _method_report_text(
    payload: dict[str, Any],
    poses: dict[str, PoseRecord],
    pairs: list[dict[str, Any]],
    anchor_cameras: list[dict[str, Any]] | None = None,
) -> str:
    width = 118
    lines = [
        "RIGCAL CALIBRATION RESULT",
        "=" * width,
        "",
        f"Method / variant: {payload['method']} / {payload['label']}",
        f"Execution status: {payload.get('execution_status', '-')}",
        f"Solver status: {payload.get('solver_status', '-')}",
        f"Artifact status: {payload['artifact_status']}",
        f"Calibration status: {payload.get('calibration_status', '-')}",
        f"Quality status: {payload['quality_status']}",
        f"Evaluation status: {payload.get('evaluation_status', '-')}",
        f"Anchor export status: {payload.get('anchor_export_status', 'unavailable')}",
        f"RViz visualization status: {payload.get('visualization_status', 'unavailable')}",
        f"Primary result: {payload.get('primary_result', '-')}",
        (
            f"Runtime: {float(payload['runtime_seconds']):.1f} s"
            if payload.get("runtime_seconds") is not None
            else "Runtime: n/a"
        ),
        f"Reference frame: {payload.get('reference_frame', '-')}",
        f"Transform convention: {payload.get('transform_convention', '-')}",
        f"Configuration: {_config_text(payload.get('config_summary', {}))}",
        "",
    ]
    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend(["WARNINGS", "-" * width])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    anchor_cameras = anchor_cameras or []
    lines.extend(
        [
            "COMMON EVALUATION / EXPORT ANCHOR",
            "-" * width,
            (
                f"Marker: {payload.get('anchor_marker_id', '-')}; "
                f"export status: {payload.get('anchor_export_status', 'unavailable')}"
            ),
        ]
    )
    anchor_rows = [
        [
            str(camera.get("camera_id", "-")),
            _fmt(camera.get("x_m")),
            _fmt(camera.get("y_m")),
            _fmt(camera.get("z_m")),
            _fmt(camera.get("roll_rad")),
            _fmt(camera.get("pitch_rad")),
            _fmt(camera.get("yaw_rad")),
            _fmt(camera.get("qx"), 6),
            _fmt(camera.get("qy"), 6),
            _fmt(camera.get("qz"), 6),
            _fmt(camera.get("qw"), 6),
        ]
        for camera in anchor_cameras
    ]
    lines.extend(
        [
            (
                _text_table(
                    [
                        "Camera",
                        "x [m]",
                        "y [m]",
                        "z [m]",
                        "roll [rad]",
                        "pitch [rad]",
                        "yaw [rad]",
                        "qx",
                        "qy",
                        "qz",
                        "qw",
                    ],
                    anchor_rows,
                )
                if anchor_rows
                else (
                    "No common-anchor camera pose is available. "
                    "See diagnostics/anchor_alignment.json."
                )
            ),
            (
                "Full exports: camera_extrinsics_anchor.csv, "
                "camera_extrinsics_anchor.json, camera_extrinsics_anchor.yaml"
            ),
            "",
        ]
    )
    graph = payload.get("metrics", {}).get("ap02_combined_graph", {})
    if payload.get("method") == "ap02" and graph:
        component_rows = [
            [
                component.get("component_id", "-"),
                ",".join(component.get("static_cameras", [])) or "-",
                ",".join(str(value) for value in component.get("marker_ids", []))
                or "-",
                component.get("moving_frame_count", 0),
                component.get("connecting_moving_frame_count", 0),
                ("yes" if component.get("calibratable") else "no"),
            ]
            for component in graph.get("components", [])
        ]
        lines.extend(
            [
                "AP02 COMBINED GRAPH",
                "-" * width,
                (
                    f"Primary coverage: "
                    f"{graph.get('reached_static_camera_count', 0)}/"
                    f"{graph.get('expected_static_camera_count', 0)} cameras"
                ),
                "Cause: " + ", ".join(graph.get("cause_codes", [])),
                str(graph.get("explanation", "")),
                _text_table(
                    [
                        "Component",
                        "Cameras",
                        "Markers",
                        "Moving frames",
                        "Bridging frames",
                        "Runnable",
                    ],
                    component_rows,
                ),
                ("Relationships between different components: " "not observable"),
                "",
            ]
        )
    diagnostics = payload.get("metrics", {})
    if payload.get("method") == "ap01":
        scale = diagnostics.get("ap01_scale", {})
        relay = diagnostics.get("ap01_relay_selection", [])
        if scale or relay:
            relay_markers = {
                int(row["marker_id"])
                for row in relay
                if isinstance(row, dict) and row.get("marker_id") is not None
            }
            relay_selected = sum(
                str(row.get("selected", "")).strip().lower() in {"true", "1", "yes"}
                for row in relay
                if isinstance(row, dict)
            )
            lines.extend(
                [
                    "AP01 SELECTION AND SCALE DIAGNOSTICS",
                    "-" * width,
                    (
                        "Relay markers: "
                        f"{len(relay_markers)}; relay observations "
                        f"selected/registered: {relay_selected}/"
                        f"{len(relay)}; scale pairs used/total: "
                        f"{scale.get('used_pairs', '-')}/"
                        f"{scale.get('raw_pairs', '-')}; scale: "
                        f"{_fmt(scale.get('scale_m_per_colmap_unit'), 6)} "
                        "m/COLMAP-unit"
                    ),
                    (
                        "Scale observations selected per marker: "
                        + json.dumps(
                            scale.get("selected_observations_per_marker", {}),
                            sort_keys=True,
                        )
                    ),
                    "",
                ]
            )
    elif payload.get("method") == "ap02":
        frames = diagnostics.get("ap02_frame_selection", {})
        optimization = diagnostics.get("ap02_combined_optimization", {})
        if frames or optimization:
            lines.extend(
                [
                    "AP02 FRAME SELECTION AND COMBINED BA",
                    "-" * width,
                    (
                        "Moving frames selected/input/minimum graph set: "
                        f"{frames.get('selected_moving_frames', '-')}/"
                        f"{frames.get('input_moving_frames', '-')}/"
                        f"{frames.get('minimum_graph_preserving_frames', '-')}"
                    ),
                    (
                        "Combined BA: status="
                        f"{optimization.get('solver_status', '-')}, "
                        f"nfev={optimization.get('nfev', '-')}/"
                        f"{optimization.get('maximum_function_evaluations', '-')}, "
                        "RMSE initial/final="
                        f"{_fmt(optimization.get('initial_reprojection_rmse_px'))}/"
                        f"{_fmt(optimization.get('final_reprojection_rmse_px'))} px, "
                        f"cost={_fmt(optimization.get('initial_cost'))} -> "
                        f"{_fmt(optimization.get('final_cost'))}"
                    ),
                    "",
                ]
            )
    elif payload.get("method") in {
        "ap03",
        "ap03_single",
        "ap03_multi",
    }:
        scale = diagnostics.get("ap03_scale", {})
        reconstruction = diagnostics.get("ap03_reconstruction", {})
        if scale or reconstruction:
            reconstruction_rows = [
                [
                    str(camera.get("camera_id", "-")),
                    str(camera.get("registered", False)).lower(),
                    str(camera.get("colmap_camera_id", "-")),
                    str(camera.get("track_support", "-")),
                    str(camera.get("shared_tracks_with_moving", "-")),
                    _fmt(camera.get("reprojection_rmse_px")),
                    ",".join(camera.get("warnings", [])) or "none",
                ]
                for camera in reconstruction.get("static_cameras", [])
                if isinstance(camera, dict)
            ]
            lines.extend(
                [
                    "AP03 COLMAP / RANSAC / SCALE DIAGNOSTICS",
                    "-" * width,
                    (
                        "Registered images/static cameras: "
                        f"{scale.get('registered_images', reconstruction.get('registered_image_count', '-'))}/"
                        f"{scale.get('registered_static_cameras', reconstruction.get('registered_static_camera_count', '-'))}; "
                        f"moving frames={reconstruction.get('registered_moving_frame_count', '-')}; "
                        f"best model={reconstruction.get('best_model', '-')}; "
                        f"sparse points={reconstruction.get('sparse_point_count', '-')}; "
                        "triangulated corners="
                        f"{scale.get('triangulated_marker_corners', '-')}"
                    ),
                    (
                        "Scale observations used/total: "
                        f"{scale.get('num_scale_observations_used', '-')}/"
                        f"{scale.get('num_scale_observations_total', '-')}; "
                        f"scale={_fmt(scale.get('scale_m_per_colmap_unit'), 6)} "
                        "m/COLMAP-unit; relative std="
                        f"{_fmt(scale.get('used_rel_std_scale'), 6)}"
                    ),
                    (
                        _text_table(
                            [
                                "Camera",
                                "Registered",
                                "Group",
                                "Tracks",
                                "Moving tracks",
                                "Reproj RMSE [px]",
                                "Warnings",
                            ],
                            reconstruction_rows,
                        )
                        if reconstruction_rows
                        else "Per-camera reconstruction support unavailable."
                    ),
                    (
                        "Camera groups: "
                        + json.dumps(
                            reconstruction.get("camera_groups", {}),
                            sort_keys=True,
                        )
                    ),
                    "Ground truth used by these diagnostics: no",
                    "",
                ]
            )
    pose_rows: list[list[str]] = []
    for camera, record in sorted(poses.items()):
        transform = record.transform
        rpy = R_to_rpy_deg(transform[:3, :3])
        pose_rows.append(
            [
                camera,
                record.source or "-",
                _fmt(transform[0, 3]),
                _fmt(transform[1, 3]),
                _fmt(transform[2, 3]),
                _fmt(rpy[0]),
                _fmt(rpy[1]),
                _fmt(rpy[2]),
            ]
        )
    lines.extend(
        [
            "FINAL STATIC-CAMERA POSES",
            "-" * width,
            (
                _text_table(
                    [
                        "Camera",
                        "Source",
                        "x [m]",
                        "y [m]",
                        "z [m]",
                        "roll [deg]",
                        "pitch [deg]",
                        "yaw [deg]",
                    ],
                    pose_rows,
                )
                if pose_rows
                else "No final static-camera pose is available."
            ),
            "",
            "PAIRWISE CAMERA-TO-CAMERA EXTRINSICS",
            "-" * width,
        ]
    )
    pair_rows = [
        [
            row["pair"],
            _fmt(row["x_m"]),
            _fmt(row["y_m"]),
            _fmt(row["z_m"]),
            _fmt(row["roll_deg"]),
            _fmt(row["pitch_deg"]),
            _fmt(row["yaw_deg"]),
            _fmt(row["baseline_m"]),
        ]
        for row in pairs
    ]
    lines.append(
        _text_table(
            [
                "Pair",
                "tx [m]",
                "ty [m]",
                "tz [m]",
                "roll [deg]",
                "pitch [deg]",
                "yaw [deg]",
                "baseline [m]",
            ],
            pair_rows,
        )
        if pair_rows
        else "Fewer than two camera poses are available."
    )
    lines.append("")
    detail_paths = payload.get("detail_artifacts", [])
    if detail_paths:
        lines.extend(
            [
                "DETAIL ARTIFACTS",
                "-" * width,
                *(f"- {path}" for path in detail_paths),
                "",
            ]
        )
    return "\n".join(lines)


def refresh_method_reports(experiment_root: Path) -> list[dict[str, Any]]:
    """Upgrade every published method front door without changing estimates."""
    hooks = current_reporting_bindings()
    _read_json = hooks.read_json
    _write_json = hooks.write_json
    _method_report_text = hooks.method_report_text
    _repository_root = hooks.repository_root
    results: list[dict[str, Any]] = []
    for result_path in sorted((experiment_root / "methods").glob("*/*/RESULT.json")):
        root = result_path.parent
        payload = _read_json(result_path)
        method = str(payload.get("method") or root.parent.name)
        if (
            method == "ap03"
            and payload.get("comparison_visibility")
            == "hidden_when_scale_variants_available"
            and (
                experiment_root / "methods" / "ap03_single" / root.name / "RESULT.json"
            ).is_file()
            and (
                experiment_root / "methods" / "ap03_multi" / root.name / "RESULT.json"
            ).is_file()
        ):
            continue
        previous_label = str(payload.get("label") or root.name)
        label = root.name
        if previous_label != label:
            payload.setdefault("original_public_label", previous_label)
            payload["label_reconciled_reason"] = (
                "The public directory label is authoritative; original "
                "requested naming remains in provenance."
            )
        poses = load_pose_records(root / "camera_extrinsics.csv")
        pairs = pairwise_rows(poses, method=method, label=label)
        _write_csv(root / "pairwise_camera_extrinsics.csv", pairs)
        previous_evaluation_metrics = (
            payload.get("metrics", {}).get("evaluation")
            if isinstance(payload.get("metrics"), dict)
            else None
        )
        quality, warnings, metrics = _quality_details(root, method)
        method_diagnostics, detail_paths = _method_diagnostics(root, method)
        metrics.update(method_diagnostics)
        if previous_evaluation_metrics is not None:
            metrics["evaluation"] = previous_evaluation_metrics
        config_summary = _configuration_summary(root, method)
        anchor_payload = _read_json(root / "camera_extrinsics_anchor.json")
        anchor_cameras = [
            item for item in anchor_payload.get("cameras", []) if isinstance(item, dict)
        ]
        solver_details = metrics.get("solver", {})
        if method == "ap02":
            solver_status = (
                "success"
                if solver_details.get("success")
                else (
                    "limit_reached"
                    if solver_details.get("limit_reached")
                    else "failed" if solver_details else "unknown"
                )
            )
        else:
            solver_status = "not_applicable"
        payload.update(
            {
                "schema_version": 5,
                "layout_version": 2,
                "method": method,
                "label": label,
                "status": "available",
                "artifact_status": "available",
                "execution_status": payload.get("execution_status", "completed"),
                "solver_status": solver_status,
                "quality_status": quality,
                "warnings": warnings,
                "metrics": metrics,
                "config_summary": config_summary,
                "static_camera_count": len(poses),
                "available_static_cameras": sorted(poses),
                "pairwise_camera_extrinsics": ("pairwise_camera_extrinsics.csv"),
                "pairwise_camera_count": len(pairs),
                "detail_artifacts": detail_paths,
                "calibration_status": payload.get("calibration_status", "available"),
                "evaluation_status": payload.get("evaluation_status", "not_run"),
                "anchor_export_status": payload.get(
                    "anchor_export_status", "ANCHOR_NOT_AVAILABLE"
                ),
                "anchor_export_available": bool(anchor_cameras),
                "anchor_camera_count": len(anchor_cameras),
                "visualization_status": payload.get(
                    "visualization_status", "not_generated"
                ),
            }
        )
        _write_json(result_path, payload)
        _write_text(
            root / "RESULT.txt",
            _method_report_text(payload, poses, pairs, anchor_cameras),
        )
        results.append(payload)
    return results


def complete_existing_dataset(
    dataset_root: Path,
    experiment_root: Path,
) -> bool:
    """Backfill late preflight evidence from an already published method."""
    observations = dataset_root / "observations"
    required = (
        "SELECTION_CANDIDATES.json",
        "REFERENCE_SELECTIONS.json",
        "REFERENCE_MARKER_ID.txt",
    )
    candidates = sorted(
        (experiment_root / "methods").glob("*/*/diagnostics/preflight/observations")
    )
    source = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / name).is_file() for name in required)
        ),
        None,
    )
    if source is None:
        return all((observations / name).is_file() for name in required)
    observations.mkdir(parents=True, exist_ok=True)
    for name in required:
        destination = observations / name
        if not destination.is_file():
            shutil.copy2(source / name, destination)
    quality = observations / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    preflight = source.parent
    for name in (
        "accepted_observations.csv",
        "rejected_observations.csv",
        "observation_filter_summary.json",
        "preflight_summary.json",
    ):
        source_file = preflight / name
        destination = quality / name
        if source_file.is_file() and not destination.is_file():
            shutil.copy2(source_file, destination)
    publication_path = observations / "PUBLICATION_COMPLETE.json"
    existing_publication = _read_json(publication_path)
    finalized_at = existing_publication.get("finalized_at") or _now()
    _write_json(
        publication_path,
        {
            "schema_version": 5,
            "layout_version": 2,
            "status": "complete",
            "selection_files": list(required),
            "quality_directory": "quality",
            "debug_images": (
                "debug_images" if (observations / "debug_images").is_dir() else None
            ),
            "finalized_at": finalized_at,
            "reconciled_from": str(source.relative_to(experiment_root)),
        },
    )
    return True


def run_real_marker_consistency(
    experiment_root: Path,
    dataset_root: Path,
    *,
    force: bool = False,
) -> Path | None:
    """Evaluate every published real-data method without rerunning it."""
    selection_path = dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    selection = _read_json(selection_path)
    anchor_value = selection.get("evaluation_anchor", {}).get("selected")
    if anchor_value is None:
        return None
    anchor = int(anchor_value)
    output = experiment_root / "evaluations" / "method_anchors_reconciled"
    report = output / "REAL_DATA_MARKER_CONSISTENCY.txt"
    if report.is_file() and not force:
        return report
    methods: list[tuple[str, Path]] = []
    for result_path in sorted((experiment_root / "methods").glob("*/*/RESULT.json")):
        method = result_path.parents[1].name
        label = result_path.parent.name
        method_root = result_path.parent / "diagnostics" / "method"
        if method_root.is_dir():
            payload = _read_json(result_path)
            config_summary = payload.get("config_summary", {})
            if method == "ap01":
                display_name = (
                    "AP01 "
                    f"(root {config_summary.get('root_camera', 'auto')}, "
                    f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
                )
            elif method == "ap02":
                display_name = (
                    "AP02 "
                    f"(ref {config_summary.get('reference_marker_id', 'auto')}, "
                    f"nfev {config_summary.get('combined_max_nfev', '-')}, "
                    f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
                )
            elif method == "ap03":
                display_name = (
                    "AP03 "
                    f"(multi {config_summary.get('multi_marker_count', '-')} markers, "
                    f"aruco {config_summary.get('aruco_detection_mode', 'baseline')})"
                )
            else:
                display_name = f"{method.upper()} ({label})"
            methods.append(
                (
                    display_name,
                    method_root,
                )
            )
    if not methods:
        return None
    dataset = _read_json(dataset_root / "dataset.json")
    cameras = [
        str(item["id"])
        for item in dataset.get("static_cameras", [])
        if isinstance(item, dict) and item.get("id")
    ]
    first_config = next(
        (
            result_path.parent / "provenance" / "resolved_config.yaml"
            for result_path in sorted(
                (experiment_root / "methods").glob("*/*/RESULT.json")
            )
            if (result_path.parent / "provenance" / "resolved_config.yaml").is_file()
        ),
        None,
    )
    marker_length = 0.17
    if first_config is not None:
        try:
            resolved = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            marker_length = float(
                resolved.get("markers", {}).get("length_m", marker_length)
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass
    command = [
        sys.executable,
        "-m",
        "camera_rig_calibration.evaluation.marker_consistency",
        "--dataset",
        str(dataset_root),
        "--results-root",
        str(experiment_root),
        "--observations-root",
        str(dataset_root / "observations"),
        "--output-root",
        str(output),
        "--anchor-marker-id",
        str(anchor),
        "--marker-length-m",
        str(marker_length),
        "--cameras",
        ",".join(cameras),
    ]
    for name, method_root in methods:
        command.extend(["--method", f"{name}={method_root.resolve()}"])
    completed = subprocess.run(
        command,
        cwd=_repository_root(experiment_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not report.is_file():
        _write_json(
            output / "COMMON_ANCHOR_STATUS.json",
            {
                "schema_version": 5,
                "status": "unavailable",
                "anchor_marker_id": anchor,
                "return_code": completed.returncode,
                "reason": ("Common marker evaluation failed; see evaluation.log."),
            },
        )
        return None
    _write_json(
        output / "COMMON_ANCHOR_STATUS.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "status": "available",
            "evaluation_scope": "single_preflight_frozen_anchor",
            "anchor_marker_id": anchor,
            "methods": [name for name, _ in methods],
            "reconciled_without_method_rerun": True,
        },
    )
    return report


__all__ = [
    "_method_report_text",
    "refresh_method_reports",
    "complete_existing_dataset",
    "run_real_marker_consistency",
]
