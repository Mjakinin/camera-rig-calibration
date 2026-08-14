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
    _read_json,
    _text_table,
)
from .reporting_bindings import current_reporting_bindings

def _configuration_summary(result_root: Path, method: str) -> dict[str, Any]:
    _read_json = current_reporting_bindings().read_json
    resolved_path = result_root / "provenance" / "resolved_config.yaml"
    try:
        config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    methods = config.get("methods", {}) if isinstance(config, dict) else {}
    colmap = config.get("colmap", {}) if isinstance(config, dict) else {}
    quality = (
        config.get("observation_quality", {})
        if isinstance(config, dict)
        else {}
    )
    markers = config.get("markers", {}) if isinstance(config, dict) else {}
    evaluation = (
        config.get("evaluation", {}) if isinstance(config, dict) else {}
    )
    config_method = "ap03" if method in {"ap03_single", "ap03_multi"} else method
    method_config = methods.get(config_method, {})
    overrides = (
        method_config.get("observation_quality", {})
        if isinstance(method_config, dict)
        else {}
    )
    effective_quality = dict(quality)
    quality_sources = {
        key: "global" for key in effective_quality
    }
    for key, value in overrides.items():
        if value is not None:
            effective_quality[key] = value
            quality_sources[key] = "method_override"
    manifest = _read_json(
        result_root / "provenance" / "run_manifest.json"
    )
    colmap_resolution = manifest.get("colmap_resolution", {})
    selection_paths = sorted(
        (result_root / "diagnostics" / "preflight").rglob(
            "SELECTION_CANDIDATES.json"
        )
    )
    selection = _read_json(selection_paths[0]) if selection_paths else {}
    ap02_selection = selection.get("ap02_reference_marker", {})
    common = {
        "baseline_version": (
            "baseline_v1"
            if method_config.get("method_contract", "baseline_v1")
            == "baseline_v1"
            else "saved compatibility"
        ),
        "evaluation_anchor_marker_id": evaluation.get(
            "anchor_marker_id"
        ),
        "quality_area_ratio": effective_quality.get(
            "minimum_marker_area_ratio"
        ),
        "quality_pnp_rmse_px": effective_quality.get(
            "maximum_pnp_reprojection_error_px"
        ),
        "quality_positive_depth": effective_quality.get(
            "require_positive_depth"
        ),
        "quality_max_distance_m": effective_quality.get(
            "maximum_marker_distance_m"
        ),
        "quality_sources": ",".join(
            f"{key}:{value}"
            for key, value in sorted(quality_sources.items())
        ),
        "aruco_detection_mode": markers.get(
            "detection_mode", "baseline"
        ),
    }
    if method == "ap01":
        direct_gate = method_config.get("direct_quality_gate", {})
        relay_gate = method_config.get("relay_quality_gate", {})
        consistency = method_config.get("direct_relay_consistency", {})
        return {
            **common,
            "root_camera": method_config.get("root_camera"),
            "top_moving_per_marker": method_config.get(
                "top_moving_per_marker"
            ),
            "scale_top_per_marker": method_config.get(
                "scale_top_per_marker"
            ),
            "matcher": colmap.get("matcher"),
            "compute_configured": colmap_resolution.get(
                "configured_compute_mode", colmap.get("compute_mode")
            ),
            "compute_resolved": colmap_resolution.get(
                "resolved_compute_mode", colmap.get("compute_mode")
            ),
            "colmap_version": colmap_resolution.get("version"),
            "gpu_requested": colmap_resolution.get(
                "requested_gpu_mode", colmap.get("gpu_mode")
            ),
            "gpu_resolved": colmap_resolution.get(
                "resolved_gpu_mode", colmap.get("gpu_mode")
            ),
            "maximum_image_size": colmap.get("maximum_image_size"),
            "maximum_features": colmap.get("maximum_features"),
            "mapper_minimum_matches": colmap.get(
                "mapper_minimum_matches"
            ),
            "intrinsics_refinement": colmap_resolution.get(
                "intrinsics_refinement"
            ),
            "direct_minimum_independent_markers": direct_gate.get(
                "minimum_independent_markers"
            ),
            "direct_minimum_inlier_ratio": direct_gate.get(
                "minimum_inlier_ratio"
            ),
            "direct_maximum_translation_dispersion_m": direct_gate.get(
                "maximum_translation_dispersion_m"
            ),
            "direct_maximum_rotation_dispersion_deg": direct_gate.get(
                "maximum_rotation_dispersion_deg"
            ),
            "relay_minimum_inlier_ratio": relay_gate.get(
                "minimum_inlier_ratio"
            ),
            "relay_maximum_translation_dispersion_m": relay_gate.get(
                "maximum_translation_dispersion_m"
            ),
            "relay_maximum_rotation_dispersion_deg": relay_gate.get(
                "maximum_rotation_dispersion_deg"
            ),
            "path_maximum_translation_disagreement_m": consistency.get(
                "maximum_translation_disagreement_m"
            ),
            "path_maximum_rotation_disagreement_deg": consistency.get(
                "maximum_rotation_disagreement_deg"
            ),
        }
    if method == "ap02":
        return {
            **common,
            "reference_marker_selection_mode": method_config.get(
                "reference_marker_selection_mode",
                ap02_selection.get("selection_mode"),
            ),
            "reference_marker_id": method_config.get("reference_marker_id"),
            "resolved_reference_marker_id": ap02_selection.get(
                "selected",
                manifest.get("resolved_selections", {}).get(
                    "ap02_reference_marker_id"
                ),
            ),
            "reference_marker_reason": ap02_selection.get("reason"),
            "reference_marker_evidence": ap02_selection.get("evidence"),
            "initialization_algorithm": "maximum_bottleneck",
            "initialization_diagnostic": "unweighted_first_hit_bfs",
            "reference_marker_maximum_frames": method_config.get(
                "reference_marker_maximum_frames"
            ),
            "top_per_marker": method_config.get("top_per_marker"),
            "top_per_marker_pair": method_config.get(
                "top_per_marker_pair"
            ),
            "maximum_total_frames": method_config.get(
                "maximum_total_frames"
            ),
            "static_max_nfev": method_config.get(
                "static_only_ba_max_function_evaluations"
            ),
            "combined_max_nfev": method_config.get(
                "combined_ba_max_function_evaluations"
            ),
            "loss": method_config.get("ba_robust_loss"),
            "loss_scale_px": method_config.get("ba_robust_loss_scale_px"),
        }
    single = method_config.get("single", {})
    multi = method_config.get("multi", {})
    marker_ids = multi.get("marker_ids")
    feature_limit_policy = method_config.get(
        "feature_limit_policy", "colmap_defaults_v1"
    )
    explicit_limits = feature_limit_policy == "wizard_explicit_limits_v1"
    configured_ap03_image_size = (
        colmap.get("ap03_maximum_image_size")
        or colmap.get("maximum_image_size")
    )
    configured_ap03_features = (
        colmap.get("ap03_maximum_features")
        or colmap.get("maximum_features")
    )
    return {
        **common,
        "feature_limits": (
            "explicit limits" if explicit_limits else "COLMAP defaults"
        ),
        "scale_input": {
            "registered_image_redetection_v1": (
                "registered-image detection"
            ),
            "wizard_filtered_observations_v1": (
                "filtered registered-image detection"
            ),
        }.get(
            method_config.get("scale_input_policy"),
            "registered-image detection",
        ),
        "minimum_marker_area_px2": method_config.get(
            "minimum_marker_area_px2"
        ),
        "single_scale_marker_id": single.get("scale_marker_id"),
        "multi_marker_count": (
            len(marker_ids) if isinstance(marker_ids, list) else marker_ids
        ),
        "matcher": colmap.get("matcher"),
        "compute_configured": colmap_resolution.get(
            "configured_compute_mode", colmap.get("compute_mode")
        ),
        "compute_resolved": colmap_resolution.get(
            "resolved_compute_mode", colmap.get("compute_mode")
        ),
        "colmap_version": colmap_resolution.get("version"),
        "gpu_requested": colmap_resolution.get(
            "requested_gpu_mode", colmap.get("gpu_mode")
        ),
        "gpu_resolved": colmap_resolution.get(
            "resolved_gpu_mode", colmap.get("gpu_mode")
        ),
        "maximum_image_size": (
            configured_ap03_image_size if explicit_limits else None
        ),
        "maximum_features": (
            configured_ap03_features if explicit_limits else None
        ),
        "mapper_minimum_matches": colmap.get("mapper_minimum_matches"),
        "intrinsics_refinement": colmap_resolution.get(
            "intrinsics_refinement"
        ),
        "scale_reprojection_threshold_px": (
            method_config.get("scale", {}).get(
                "reprojection_threshold_px"
            )
        ),
        "scale_ransac_iterations": method_config.get("scale", {}).get(
            "ransac_iterations"
        ),
        "scale_minimum_inliers": method_config.get("scale", {}).get(
            "minimum_inliers"
        ),
        "scale_maximum_observations_per_marker": (
            method_config.get("scale", {}).get(
                "maximum_observations_per_marker"
            )
        ),
    }



def _config_text(summary: dict[str, Any]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float) and value != 0.0 and abs(value) < 1e-4:
            return format(value, ".15f").rstrip("0").rstrip(".")
        return str(value)

    return ", ".join(
        f"{key}={render(value)}"
        for key, value in summary.items()
        if value is not None
        and key not in {
            "reference_marker_evidence",
            "reference_marker_reason",
            "intrinsics_refinement",
            "quality_sources",
            "colmap_version",
        }
    ) or "baseline/default configuration"


def _baseline_contract(
    *,
    category: str,
    method_payloads: list[dict[str, Any]],
    evaluation_anchor: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the auditable Route-2 CPU baseline contract."""

    anchor = evaluation_anchor.get("selected")
    def integer(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    common_checks = {
        "simulation_category": category == "simulation",
        "evaluation_anchor_marker_14": str(anchor) == "14",
    }
    variants: list[dict[str, Any]] = []
    for payload in method_payloads:
        method = str(payload.get("method", ""))
        config = payload.get("config_summary", {})
        checks: dict[str, bool] = dict(common_checks)
        if method == "ap01":
            checks.update(
                {
                    "root_cam_edge_3": config.get("root_camera")
                    == "cam_edge_3",
                    "configured_cpu_baseline": config.get(
                        "compute_configured"
                    )
                    == "cpu_baseline",
                    "resolved_cpu_baseline": config.get(
                        "compute_resolved"
                    )
                    == "cpu_baseline",
                    "exhaustive_matcher": config.get("matcher")
                    == "exhaustive",
                    "maximum_image_size_1600": int(
                        config.get("maximum_image_size") or 0
                    )
                    == 1600,
                    "maximum_features_4096": int(
                        config.get("maximum_features") or 0
                    )
                    == 4096,
                    "mapper_minimum_matches_8": int(
                        config.get("mapper_minimum_matches") or 0
                    )
                    == 8,
                }
            )
        elif method == "ap02":
            checks.update(
                {
                    "reference_mode_baseline": config.get(
                        "reference_marker_selection_mode"
                    )
                    == "baseline",
                    "reference_marker_14": integer(
                        config.get("resolved_reference_marker_id")
                        or config.get("reference_marker_id")
                    )
                    == 14,
                    "static_nfev_50": integer(
                        config.get("static_max_nfev") or 0
                    )
                    == 50,
                    "combined_nfev_50": integer(
                        config.get("combined_max_nfev") or 0
                    )
                    == 50,
                    "maximum_bottleneck_initialization": config.get(
                        "initialization_algorithm"
                    )
                    == "maximum_bottleneck",
                }
            )
        elif method in {"ap03", "ap03_single", "ap03_multi"}:
            checks.update(
                {
                    "configured_cpu_baseline": config.get(
                        "compute_configured"
                    )
                    == "cpu_baseline",
                    "resolved_cpu_baseline": config.get(
                        "compute_resolved"
                    )
                    == "cpu_baseline",
                    "exhaustive_matcher": config.get("matcher")
                    == "exhaustive",
                    "colmap_default_feature_limits": (
                        config.get("feature_limits") == "COLMAP defaults"
                        and config.get("maximum_image_size") is None
                        and config.get("maximum_features") is None
                    ),
                    "mapper_minimum_matches_8": int(
                        config.get("mapper_minimum_matches") or 0
                    )
                    == 8,
                }
            )
        else:
            continue
        variants.append(
            {
                "method": method,
                "label": payload.get("label"),
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    return {
        "contract": "route2_cpu_ref14_50x50_v1",
        "category": category,
        "evaluation_anchor_marker_id": anchor,
        "variants": variants,
        "passes": bool(variants) and all(
            item["passes"] for item in variants
        ),
    }


def _baseline_contract_text(contract: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for variant in contract.get("variants", []):
        failed = [
            key
            for key, value in variant.get("checks", {}).items()
            if not value
        ]
        rows.append(
            [
                str(variant.get("method", "-")),
                str(variant.get("label", "-")),
                "PASS" if variant.get("passes") else "NOT BASELINE",
                ", ".join(failed) if failed else "all checks satisfied",
            ]
        )
    return "\n".join(
        [
            "BASELINE CONTRACT",
            "-" * 138,
            f"Contract: {contract.get('contract')}",
            (
                "Overall: PASS"
                if contract.get("passes")
                else "Overall: NOT A COMPLETE BASELINE CONTRACT"
            ),
            _text_table(
                ["Method", "Variant", "Status", "Failed checks / evidence"],
                rows,
            ),
            "",
        ]
    )



__all__ = [
    '_configuration_summary',
    '_config_text',
    '_baseline_contract',
    '_baseline_contract_text',
]
