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

from .reporting_core import (
    _finite,
    _read_json,
)
from .reporting_bindings import current_reporting_bindings

def _quality_details(
    result_root: Path, method: str
) -> tuple[str, list[str], dict[str, Any]]:
    _read_json = current_reporting_bindings().read_json
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    quality = "good"
    if method == "ap01":
        diagnostics = _read_json(
            result_root
            / "diagnostics"
            / "method"
            / "static_extrinsics"
            / "AP01_DIAGNOSTICS.json"
        )
        per_target = diagnostics.get("per_target_diagnostics", {})
        unstable: list[dict[str, Any]] = []
        weak: list[str] = []
        disagreements: list[str] = []
        for camera, item in per_target.items():
            if not isinstance(item, dict):
                continue
            selected = str(item.get("selected_method", ""))
            details = (
                item.get("direct")
                if selected == "direct_multimarker"
                else item.get("relay")
            )
            if not isinstance(details, dict):
                continue
            candidates = int(details.get("candidates") or 0)
            translation = _finite(
                details.get(
                    "maximum_inlier_translation_dispersion_m",
                    details.get("translation_deviation_median_m"),
                )
            )
            rotation = _finite(
                details.get(
                    "maximum_inlier_rotation_dispersion_deg",
                    details.get("rotation_deviation_median_deg"),
                )
            )
            translation_floor, rotation_floor = (
                (0.12, 4.0)
                if selected == "direct_multimarker"
                else (0.30, 7.0)
            )
            row = {
                "camera": camera,
                "selected_path": selected,
                "candidates": candidates,
                "inliers": details.get("inliers"),
                "inlier_ratio": details.get("inlier_ratio"),
                "inlier_marker_ids": details.get("inlier_marker_ids", []),
                "pose_fallback_used": details.get("pose_fallback_used"),
                "translation_dispersion_m": translation,
                "rotation_dispersion_deg": rotation,
                "translation_warning_floor_m": translation_floor,
                "rotation_warning_floor_deg": rotation_floor,
            }
            if (
                selected == "direct_multimarker"
                and int(details.get("independent_inlier_marker_count") or 0)
                < 3
            ):
                weak.append(str(camera))
                row["support"] = "fewer_than_three_independent_inlier_markers"
            if (
                selected == "quality_rejected"
                or details.get("stable") is False
                or (
                    "stable" not in details
                    and (
                        (
                            translation is not None
                            and translation > translation_floor
                        )
                        or (
                            rotation is not None
                            and rotation > rotation_floor
                        )
                    )
                )
            ):
                unstable.append(row)
            if item.get("quality_warning") == (
                "warning_direct_relay_disagreement"
            ):
                disagreements.append(str(camera))
        metrics["ap01_consensus"] = {
            "path_thresholds": {
                "direct": {
                    "translation_m": 0.12,
                    "rotation_deg": 4.0,
                },
                "relay": {
                    "translation_m": 0.30,
                    "rotation_deg": 7.0,
                },
            },
            "unstable_targets": unstable,
            "weak_direct_single_support": weak,
            "direct_relay_disagreements": disagreements,
            "per_target": per_target,
        }
        if unstable:
            quality = "warning_unstable_consensus"
            warnings.append(
                "AP01 selected-path consensus exceeds the documented "
                "dispersion floor for: "
                + ", ".join(str(item["camera"]) for item in unstable)
                + "."
            )
        if weak:
            warnings.append(
                "AP01 direct support is weak (one candidate) for: "
                + ", ".join(weak)
                + "."
            )
        if disagreements:
            if quality == "good":
                quality = "warning_direct_relay_disagreement"
            warnings.append(
                "AP01 Direct and Relay are individually stable but disagree "
                "beyond 0.12 m / 4 deg for: "
                + ", ".join(disagreements)
                + "; Direct remains the published path."
            )
    if method == "ap02":
        optimizer = _read_json(
            result_root
            / "diagnostics"
            / "method"
            / "graph_ba"
            / "with_moving"
            / "ap02_optimization_summary.json"
        )
        if not optimizer:
            report = _read_json(
                result_root
                / "diagnostics"
                / "method"
                / "final_results"
                / "AP02_REPORT.json"
            )
            optimizer = report.get("combined_optimizer", {})
        if isinstance(optimizer, dict) and optimizer:
            metrics["optimizer"] = optimizer
            success = bool(
                optimizer.get("solver_success", optimizer.get("success"))
            )
            message = str(
                optimizer.get("solver_message", optimizer.get("message", ""))
            ).strip()
            rmse = _finite(
                optimizer.get(
                    "final_reprojection_rmse_px",
                    optimizer.get("final_rmse_px"),
                )
            )
            metrics["combined_reprojection_rmse_px"] = rmse
            if rmse is None:
                quality = "warning_reprojection_unavailable"
                warnings.append(
                    "AP02 Combined final reprojection RMSE is unavailable; "
                    "the solver status alone is not a quality measure."
                )
            elif rmse > 25.0:
                quality = "poor_high_reprojection"
                warnings.append(
                    f"AP02 Combined reprojection RMSE is {rmse:.3f} px "
                    "(poor: >25 px)."
                )
            elif rmse > 5.0:
                quality = "warning_high_reprojection"
                warnings.append(
                    f"AP02 Combined reprojection RMSE is {rmse:.3f} px "
                    "(warning: >5 px)."
                )
            if not success and (
                "maximum number" in message.lower()
                or optimizer.get("nfev")
                == optimizer.get("maximum_function_evaluations")
            ):
                warnings.append(
                    "The AP02 combined optimizer reached its configured "
                    f"limit ({optimizer.get('nfev', '?')} evaluations). "
                    "Solver completion and geometric quality are reported "
                    "independently."
                )
            elif not success:
                warnings.append(
                    "The AP02 combined optimizer did not report convergence"
                    + (f": {message}" if message else ".")
                )
            metrics["solver"] = {
                "success": success,
                "message": message,
                "status": optimizer.get(
                    "solver_status", optimizer.get("status")
                ),
                "nfev": optimizer.get("nfev"),
                "njev": optimizer.get("njev"),
                "optimality": optimizer.get("optimality"),
                "maximum_function_evaluations": optimizer.get(
                    "maximum_function_evaluations"
                ),
                "limit_reached": bool(
                    not success
                    and (
                        "maximum number" in message.lower()
                        or optimizer.get("nfev")
                        == optimizer.get("maximum_function_evaluations")
                    )
                ),
            }
    if method in {"ap03_single", "ap03_multi"}:
        result = _read_json(result_root / "RESULT.json")
        scale = result.get("metrics", {}).get("ap03_scale", {})
        relative_std = _finite(scale.get("used_rel_std_scale"))
        metrics["ap03_scale"] = scale
        metrics["ap03_scale_relative_std"] = relative_std
        metrics["ap03_registration"] = {
            "registered_static_cameras": scale.get(
                "registered_static_cameras"
            ),
            "registered_moving_frames": scale.get(
                "registered_moving_frames"
            ),
            "registered_images": scale.get("registered_images"),
            "sparse_points": scale.get("num_sparse_points3d"),
            "missing_static_cameras": scale.get(
                "missing_static_cameras", []
            ),
        }
        provenance = _read_json(
            result_root / "provenance" / "derived_result.json"
        )
        shared_container = provenance.get("shared_colmap_container")
        reconstruction = {}
        if shared_container:
            experiment_root = result_root.parents[2]
            reconstruction = _read_json(
                experiment_root
                / str(shared_container)
                / "diagnostics"
                / "method"
                / "colmap"
                / "inspection"
                / "AP03_RECONSTRUCTION_DIAGNOSTICS.json"
            )
        if reconstruction:
            metrics["ap03_reconstruction"] = reconstruction
            reconstruction_warnings = reconstruction.get("warnings", [])
            if reconstruction_warnings:
                if quality == "good":
                    quality = "warning_weak_reconstruction_support"
                warnings.append(
                    "AP03 reconstruction support warnings: "
                    + ", ".join(
                        f"{item.get('camera_id') or 'groups'}:"
                        f"{item.get('code')}"
                        for item in reconstruction_warnings
                        if isinstance(item, dict)
                    )
                    + "."
                )
        if relative_std is None:
            quality = "warning_scale_dispersion_unavailable"
            warnings.append("AP03 scale dispersion is unavailable.")
        elif relative_std > 0.10:
            quality = "poor_scale_dispersion"
            warnings.append(
                f"AP03 {method.removeprefix('ap03_').title()} scale "
                f"relative standard deviation is {100.0 * relative_std:.2f}% "
                "(poor: >10%)."
            )
        elif relative_std > 0.05:
            quality = "warning_scale_dispersion"
            warnings.append(
                f"AP03 {method.removeprefix('ap03_').title()} scale "
                f"relative standard deviation is {100.0 * relative_std:.2f}% "
                "(warning: >5%)."
            )
    status = _read_json(
        result_root / "diagnostics" / "method" / "METHOD_STATUS.json"
    )
    for key in ("warning", "error"):
        value = status.get(key)
        if value and str(value) not in warnings:
            warnings.append(str(value))
            if quality == "good":
                quality = "warning"
    override_paths = sorted(
        (
            *(
                result_root / "diagnostics" / "preflight"
            ).rglob("OBSERVATION_REVIEW_OVERRIDE.json"),
            *(
                result_root / "diagnostics" / "preflight"
            ).rglob("REQUIRED_CAMERA_OVERRIDE.json"),
        )
    )
    if override_paths:
        override = _read_json(override_paths[0])
        missing = list(override.get("missing_required_cameras", []))
        quality = "partial_coverage"
        metrics["missing_required_cameras"] = missing
        warning = str(
            override.get("warning")
            or "The operator explicitly continued with incomplete required-camera coverage."
        )
        if warning not in warnings:
            warnings.insert(0, warning)
    if method == "ap02":
        graph_paths = sorted(
            (result_root / "diagnostics" / "preflight").rglob(
                "AP02_COMBINED_GRAPH.json"
            )
        )
        method_graph = _read_json(
            result_root
            / "diagnostics"
            / "method"
            / "aruco_observations"
            / "component_manifest.json"
        )
        graph = _read_json(graph_paths[0]) if graph_paths else {}
        if method_graph:
            components = list(method_graph.get("components", []))
            primary_id = method_graph.get("primary_component_id")
            primary = next(
                (
                    component
                    for component in components
                    if component.get("component_id") == primary_id
                ),
                {},
            )
            expected = list(
                method_graph.get("expected_static_cameras", [])
            )
            reached = list(primary.get("static_cameras", []))
            graph.update(
                {
                    "components": components,
                    "component_count": len(components),
                    "reference_marker_id": method_graph.get(
                        "reference_marker_id"
                    ),
                    "reference_component_id": primary_id,
                    "expected_static_cameras": expected,
                    "expected_static_camera_count": len(expected),
                    "reached_static_cameras": reached,
                    "reached_static_camera_count": len(reached),
                    "missing_static_cameras": sorted(
                        set(expected) - set(reached)
                    ),
                    "complete": len(reached) == len(expected),
                }
            )
        if graph:
            metrics["ap02_combined_graph"] = graph
            if not graph.get("complete", True):
                quality = "partial_coverage"
                warning = (
                    f"AP02 Combined reaches "
                    f"{graph.get('reached_static_camera_count', 0)}/"
                    f"{graph.get('expected_static_camera_count', 0)} "
                    f"static cameras in {graph.get('component_count', 0)} "
                    "disconnected components. Cross-component extrinsics "
                    "are not observable."
                )
                if warning not in warnings:
                    warnings.insert(0, warning)
        component_results = _read_json(
            result_root
            / "diagnostics"
            / "method"
            / "component_diagnostics"
            / "AP02_COMPONENT_RESULTS.json"
        )
        if component_results:
            metrics["ap02_component_results"] = component_results
    return quality, warnings, metrics



__all__ = [
    '_quality_details',
]
