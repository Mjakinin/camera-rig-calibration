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
@dataclass(frozen=True)
class PoseRecord:
    entity_id: str
    transform: np.ndarray
    source: str
    reference_frame: str
    transform_convention: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO

    buffer = StringIO(newline="")
    with buffer:
        writer = csv.DictWriter(
            buffer,
            fieldnames=fields or ["status"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        text = buffer.getvalue()
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 3) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _mean(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.mean(clean)) if clean else None


def _median(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return float(np.median(clean)) if clean else None


def _maximum(values: Iterable[Any]) -> float | None:
    clean = [value for item in values if (value := _finite(item)) is not None]
    return max(clean) if clean else None


def _text_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(
            [
                len(str(header)),
                *(
                    len(row[index])
                    for row in rendered
                    if index < len(row)
                ),
            ]
        )
        for index, header in enumerate(headers)
    ]
    output = [
        " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        ).rstrip(),
        "-+-".join("-" * width for width in widths),
    ]
    output.extend(
        " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        ).rstrip()
        for row in rendered
    )
    return "\n".join(output)


def _pose_from_row(row: dict[str, Any]) -> np.ndarray:
    if all(str(row.get(key, "")).strip() for key in ("rvec_x", "rvec_y", "rvec_z")):
        rotation = rvec_to_R(
            np.asarray(
                [
                    _float(row, "rvec_x"),
                    _float(row, "rvec_y"),
                    _float(row, "rvec_z"),
                ],
                dtype=np.float64,
            )
        )
    else:
        rotation = rpy_to_R(
            math.radians(_float(row, "roll_deg")),
            math.radians(_float(row, "pitch_deg")),
            math.radians(_float(row, "yaw_deg")),
        )
    return make_T(
        rotation,
        [_float(row, "x_m"), _float(row, "y_m"), _float(row, "z_m")],
    )


def load_pose_records(path: Path) -> dict[str, PoseRecord]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: dict[str, PoseRecord] = {}
    for row in rows:
        entity_id = str(
            row.get("entity_id")
            or row.get("static_camera")
            or row.get("camera")
            or ""
        ).strip()
        if not entity_id:
            continue
        records[entity_id] = PoseRecord(
            entity_id=entity_id,
            transform=_pose_from_row(row),
            source=str(row.get("source", "")),
            reference_frame=str(row.get("reference_frame", "")),
            transform_convention=str(row.get("transform_convention", "")),
        )
    return records


def _pose_columns(prefix: str, transform: np.ndarray) -> dict[str, float]:
    rpy = R_to_rpy_deg(transform[:3, :3])
    rvec = R_to_rvec(transform[:3, :3])
    return {
        f"{prefix}x_m": float(transform[0, 3]),
        f"{prefix}y_m": float(transform[1, 3]),
        f"{prefix}z_m": float(transform[2, 3]),
        f"{prefix}roll_deg": float(rpy[0]),
        f"{prefix}pitch_deg": float(rpy[1]),
        f"{prefix}yaw_deg": float(rpy[2]),
        f"{prefix}rvec_x": float(rvec[0]),
        f"{prefix}rvec_y": float(rvec[1]),
        f"{prefix}rvec_z": float(rvec[2]),
    }


def _direction(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.full(3, np.nan)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def pairwise_rows(
    records: dict[str, PoseRecord],
    *,
    method: str,
    label: str,
) -> list[dict[str, Any]]:
    """Return gauge-invariant T_from_camera_to_camera relations."""
    rows: list[dict[str, Any]] = []
    for first, second in combinations(sorted(records), 2):
        transform = invT(records[first].transform) @ records[second].transform
        translation = transform[:3, 3]
        direction = _direction(translation)
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "from_camera": first,
            "to_camera": second,
            "pair": f"{first}-{second}",
            "transform_convention": (
                "T_from_camera_to_camera = "
                "inv(T_reference_from_camera) @ T_reference_to_camera"
            ),
            "baseline_m": float(np.linalg.norm(translation)),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
        }
        row.update(_pose_columns("", transform))
        rows.append(row)
    return rows


def _configuration_summary(result_root: Path, method: str) -> dict[str, Any]:
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
    return {
        **common,
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
            colmap.get("ap03_maximum_image_size")
            or colmap.get("maximum_image_size")
        ),
        "maximum_features": (
            colmap.get("ap03_maximum_features")
            or colmap.get("maximum_features")
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


def _quality_details(
    result_root: Path, method: str
) -> tuple[str, list[str], dict[str, Any]]:
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
                    "maximum_image_size_2400": int(
                        config.get("maximum_image_size") or 0
                    )
                    == 2400,
                    "maximum_features_8192": int(
                        config.get("maximum_features") or 0
                    )
                    == 8192,
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


def _method_diagnostics(
    result_root: Path, method: str
) -> tuple[dict[str, Any], list[str]]:
    method_root = result_root / "diagnostics" / "method"
    diagnostics: dict[str, Any] = {}
    paths: list[str] = []
    candidates: tuple[tuple[str, Path], ...]
    if method == "ap01":
        candidates = (
            (
                "ap01_scale",
                method_root / "metric_scale" / "SCALE_DIAGNOSTICS.json",
            ),
            (
                "ap01_relay_selection",
                method_root / "candidates" / "AP01_RELAY_SELECTION.json",
            ),
        )
    elif method == "ap02":
        candidates = (
            (
                "ap02_frame_selection",
                method_root
                / "aruco_observations"
                / "ap02_frame_selection.json",
            ),
            (
                "ap02_combined_optimization",
                method_root
                / "graph_ba"
                / "with_moving"
                / "ap02_optimization_summary.json",
            ),
        )
    elif method == "ap03":
        candidates = (
            (
                "ap03_scale",
                method_root
                / "scale_multi"
                / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json",
            ),
        )
    elif method in {"ap03_single", "ap03_multi"}:
        provenance = _read_json(
            result_root / "provenance" / "derived_result.json"
        )
        experiment_root = result_root.parents[2]
        metadata = provenance.get("scale_metadata")
        candidates = (
            (
                "ap03_scale",
                experiment_root / str(metadata)
                if metadata
                else Path("__missing__"),
            ),
        )
        if provenance:
            diagnostics["shared_colmap"] = provenance
            paths.append("provenance/derived_result.json")
            shared = provenance.get("shared_colmap_container")
            if shared:
                reconstruction = (
                    experiment_root
                    / str(shared)
                    / "diagnostics"
                    / "method"
                    / "colmap"
                    / "inspection"
                    / "AP03_RECONSTRUCTION_DIAGNOSTICS.json"
                )
                if reconstruction.is_file():
                    diagnostics["ap03_reconstruction"] = _read_json(
                        reconstruction
                    )
                    paths.append(
                        str(
                            reconstruction.relative_to(experiment_root)
                        )
                    )
    else:
        candidates = ()
    for key, path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        diagnostics[key] = value
        try:
            relative = path.relative_to(result_root)
        except ValueError:
            relative = (
                Path("../../..")
                / path.relative_to(result_root.parents[2])
            )
        paths.append(relative.as_posix())
    for path in (
        method_root.rglob("*optimization_history.csv")
        if method == "ap02"
        else ()
    ):
        paths.append(str(path.relative_to(result_root)))
    return diagnostics, sorted(set(paths))


def _scale_comparison_rows(
    method_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize method scale mechanisms without treating unlike values alike."""
    rows: list[dict[str, Any]] = []
    for payload in method_payloads:
        method = str(payload.get("method", ""))
        metrics = payload.get("metrics", {})
        if method == "ap01":
            scale = metrics.get("ap01_scale", {})
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "marker-motion pair scale",
                    "scale_m_per_colmap_unit": scale.get(
                        "scale_m_per_colmap_unit"
                    ),
                    "used": scale.get("used_pairs"),
                    "total": scale.get("raw_pairs"),
                    "relative_std": scale.get("used_relative_std"),
                }
            )
        elif method == "ap02":
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "metric marker-graph BA",
                    "scale_m_per_colmap_unit": None,
                    "used": None,
                    "total": None,
                    "relative_std": None,
                }
            )
        elif method in {"ap03", "ap03_single", "ap03_multi"}:
            scale = metrics.get("ap03_scale", {})
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "marker-corner RANSAC scale",
                    "scale_m_per_colmap_unit": scale.get(
                        "scale_m_per_colmap_unit"
                    ),
                    "used": scale.get("num_scale_observations_used"),
                    "total": scale.get("num_scale_observations_total"),
                    "relative_std": scale.get("used_rel_std_scale"),
                }
            )
    return rows


def _scale_comparison_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No successful method scale result is available."
    return _text_table(
        [
            "Method",
            "Variant",
            "Scale mechanism",
            "Scale [m/COLMAP unit]",
            "Used/total",
            "Relative std",
        ],
        [
            [
                row["method"],
                row["label"],
                row["mechanism"],
                (
                    _fmt(row["scale_m_per_colmap_unit"], 6)
                    if row["scale_m_per_colmap_unit"] is not None
                    else "n/a (already metric)"
                ),
                (
                    f"{row['used']}/{row['total']}"
                    if row["used"] is not None
                    and row["total"] is not None
                    else "-"
                ),
                (
                    _fmt(row["relative_std"], 6)
                    if row["relative_std"] is not None
                    else "-"
                ),
            ]
            for row in rows
        ],
    )


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
                ",".join(
                    str(value)
                    for value in component.get("marker_ids", [])
                )
                or "-",
                component.get("moving_frame_count", 0),
                component.get("connecting_moving_frame_count", 0),
                (
                    "yes"
                    if component.get("calibratable")
                    else "no"
                ),
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
                (
                    "Relationships between different components: "
                    "not observable"
                ),
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
                str(row.get("selected", "")).strip().lower()
                in {"true", "1", "yes"}
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
                            scale.get(
                                "selected_observations_per_marker", {}
                            ),
                            sort_keys=True,
                        )
                    ),
                    "",
                ]
            )
    elif payload.get("method") == "ap02":
        frames = diagnostics.get("ap02_frame_selection", {})
        optimization = diagnostics.get(
            "ap02_combined_optimization", {}
        )
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
            else "No final static-camera pose is available.",
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
    results: list[dict[str, Any]] = []
    for result_path in sorted(
        (experiment_root / "methods").glob("*/*/RESULT.json")
    ):
        root = result_path.parent
        payload = _read_json(result_path)
        method = str(payload.get("method") or root.parent.name)
        if (
            method == "ap03"
            and payload.get("comparison_visibility")
            == "hidden_when_scale_variants_available"
            and (
                experiment_root
                / "methods"
                / "ap03_single"
                / root.name
                / "RESULT.json"
            ).is_file()
            and (
                experiment_root
                / "methods"
                / "ap03_multi"
                / root.name
                / "RESULT.json"
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
        method_diagnostics, detail_paths = _method_diagnostics(
            root, method
        )
        metrics.update(method_diagnostics)
        if previous_evaluation_metrics is not None:
            metrics["evaluation"] = previous_evaluation_metrics
        config_summary = _configuration_summary(root, method)
        anchor_payload = _read_json(root / "camera_extrinsics_anchor.json")
        anchor_cameras = [
            item
            for item in anchor_payload.get("cameras", [])
            if isinstance(item, dict)
        ]
        solver_details = metrics.get("solver", {})
        if method == "ap02":
            solver_status = (
                "success"
                if solver_details.get("success")
                else "limit_reached"
                if solver_details.get("limit_reached")
                else "failed"
                if solver_details
                else "unknown"
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
                "execution_status": payload.get(
                    "execution_status", "completed"
                ),
                "solver_status": solver_status,
                "quality_status": quality,
                "warnings": warnings,
                "metrics": metrics,
                "config_summary": config_summary,
                "static_camera_count": len(poses),
                "available_static_cameras": sorted(poses),
                "pairwise_camera_extrinsics": (
                    "pairwise_camera_extrinsics.csv"
                ),
                "pairwise_camera_count": len(pairs),
                "detail_artifacts": detail_paths,
                "calibration_status": payload.get(
                    "calibration_status", "available"
                ),
                "evaluation_status": payload.get(
                    "evaluation_status", "not_run"
                ),
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
        (experiment_root / "methods").glob(
            "*/*/diagnostics/preflight/observations"
        )
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
                "debug_images"
                if (observations / "debug_images").is_dir()
                else None
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
    selection_path = (
        dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    selection = _read_json(selection_path)
    anchor_value = selection.get("evaluation_anchor", {}).get("selected")
    if anchor_value is None:
        return None
    anchor = int(anchor_value)
    output = (
        experiment_root
        / "evaluations"
        / "method_anchors_reconciled"
    )
    report = output / "REAL_DATA_MARKER_CONSISTENCY.txt"
    if report.is_file() and not force:
        return report
    methods: list[tuple[str, Path]] = []
    for result_path in sorted(
        (experiment_root / "methods").glob("*/*/RESULT.json")
    ):
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
            result_path.parent
            / "provenance"
            / "resolved_config.yaml"
            for result_path in sorted(
                (experiment_root / "methods").glob("*/*/RESULT.json")
            )
            if (
                result_path.parent
                / "provenance"
                / "resolved_config.yaml"
            ).is_file()
        ),
        None,
    )
    marker_length = 0.17
    if first_config is not None:
        try:
            resolved = yaml.safe_load(
                first_config.read_text(encoding="utf-8")
            )
            marker_length = float(
                resolved.get("markers", {}).get("length_m", marker_length)
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass
    command = [
        sys.executable,
        str(Path(__file__).with_name("marker_consistency.py")),
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
    (output / "evaluation.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0 or not report.is_file():
        _write_json(
            output / "COMMON_ANCHOR_STATUS.json",
            {
                "schema_version": 5,
                "status": "unavailable",
                "anchor_marker_id": anchor,
                "return_code": completed.returncode,
                "reason": (
                    "Common marker evaluation failed; see evaluation.log."
                ),
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


def _real_results_text(
    experiment_root: Path,
    method_payloads: list[dict[str, Any]],
    dataset_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    width = 138
    dataset_root = dataset_root or experiment_root
    dataset = _read_json(dataset_root / "dataset.json")
    selection = _read_json(
        dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    evaluation_anchor = selection.get("evaluation_anchor", {})
    lines = [
        "REAL-VEHICLE CAMERA-RIG CALIBRATION RESULTS",
        "=" * width,
        "",
        f"Experiment: {experiment_root.name}",
        f"Dataset: {dataset.get('id', dataset_root.name)}",
        (
            "Common evaluation anchor: marker "
            f"{evaluation_anchor.get('selected', '-')} "
            f"(configured {evaluation_anchor.get('configured', '-')}; "
            "frozen during preflight)"
        ),
        f"Anchor reason: {evaluation_anchor.get('reason', '-')}",
        "",
    ]
    marker_text, marker_path = _latest_marker_report(experiment_root)
    lines.extend(
        [
            marker_text,
            "",
            "METHOD / VARIANT OVERVIEW",
            "-" * width,
        ]
    )
    overview: list[list[str]] = []
    all_pairs: list[dict[str, Any]] = []
    for payload in method_payloads:
        root = (
            experiment_root
            / "methods"
            / str(payload["method"])
            / str(payload["label"])
        )
        pairs_path = root / "pairwise_camera_extrinsics.csv"
        if pairs_path.is_file():
            with pairs_path.open(newline="", encoding="utf-8") as handle:
                all_pairs.extend(list(csv.DictReader(handle)))
        quality_text = str(payload.get("quality_status", "-"))
        graph = payload.get("metrics", {}).get(
            "ap02_combined_graph", {}
        )
        if payload.get("method") == "ap02" and graph and not graph.get(
            "complete", True
        ):
            quality_text = (
                "partial — primary "
                f"{graph.get('reached_static_camera_count', 0)}/"
                f"{graph.get('expected_static_camera_count', 0)}, "
                f"{graph.get('component_count', 0)} graph components"
            )
        overview.append(
            [
                payload["method"],
                payload["label"],
                payload.get("artifact_status", "available"),
                quality_text,
                payload.get("static_camera_count", 0),
                (
                    _fmt(payload.get("runtime_seconds"), 1) + " s"
                    if payload.get("runtime_seconds") is not None
                    else "-"
                ),
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
                    "Cameras",
                    "Runtime",
                    "Key configuration",
                ],
                overview,
            ),
            "",
            "SCALE COMPARISON",
            "-" * width,
            _scale_comparison_text(
                _scale_comparison_rows(method_payloads)
            ),
            "",
        ]
    )
    for payload in method_payloads:
        root = (
            experiment_root
            / "methods"
            / str(payload["method"])
            / str(payload["label"])
        )
        lines.extend(
            [
                f"{payload['method']} / {payload['label']}",
                "-" * width,
                (root / "RESULT.txt").read_text(encoding="utf-8"),
                "",
            ]
        )
    disagreement_summaries, disagreement_rows = (
        _real_variant_disagreement(all_pairs)
    )
    lines.extend(
        [
            "DIRECT VARIANT-TO-VARIANT DISAGREEMENT",
            "-" * width,
            _text_table(
                [
                    "First",
                    "Second",
                    "Pairs",
                    "mean t delta [cm]",
                    "max t delta [cm]",
                    "mean r delta [deg]",
                    "max r delta [deg]",
                    "mean baseline delta [cm]",
                ],
                [
                    [
                        f"{row['first_method']}/{row['first_label']}",
                        f"{row['second_method']}/{row['second_label']}",
                        row["pair_count"],
                        _fmt(row["mean_translation_delta_cm"]),
                        _fmt(row["max_translation_delta_cm"]),
                        _fmt(row["mean_rotation_delta_deg"]),
                        _fmt(row["max_rotation_delta_deg"]),
                        _fmt(row["mean_baseline_delta_cm"]),
                    ]
                    for row in disagreement_summaries
                ],
            ),
            "",
        ]
    )
    payload = {
        "category": "real_vehicle",
        "experiment": experiment_root.name,
        "methods": method_payloads,
        "pairwise_camera_extrinsics": all_pairs,
        "variant_disagreement_summary": disagreement_summaries,
        "variant_disagreement_rows": disagreement_rows,
        "marker_consistency_path": (
            str(marker_path.relative_to(experiment_root))
            if marker_path is not None
            else None
        ),
        "evaluation_anchor": evaluation_anchor,
        "scale_comparison": _scale_comparison_rows(method_payloads),
    }
    return "\n".join(lines), payload


def _simulation_results(
    experiment_root: Path,
    dataset_root: Path,
    method_payloads: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    gt_payload = ensure_simulation_ground_truth(
        dataset_root, backfilled=True
    )
    if gt_payload.get("status") != "available":
        text = (
            "SIMULATION CALIBRATION RESULTS\n"
            "==============================\n\n"
            f"Ground truth unavailable: {gt_payload.get('reason', 'unknown')}\n"
        )
        return text, {
            "category": "simulation",
            "experiment": experiment_root.name,
            "status": "evaluation_unavailable",
            "ground_truth": gt_payload,
            "methods": method_payloads,
        }
    gt_cameras, gt_markers = _simulation_gt_maps(gt_payload)
    pair_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    marker_results: list[dict[str, Any]] = []
    marker_texts: list[str] = []
    selection = _read_json(
        dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    anchor_value = selection.get("evaluation_anchor", {}).get("selected")
    anchor_marker_id = int(anchor_value) if anchor_value is not None else None
    gt_anchor_cameras = (
        _ground_truth_anchor_records(
            anchor_marker_id=anchor_marker_id,
            gt_cameras=gt_cameras,
            gt_markers=gt_markers,
        )
        if anchor_marker_id is not None
        else {}
    )
    expected_cameras = sorted(gt_cameras)
    anchor_gt_rows: list[dict[str, Any]] = []
    anchor_gt_summaries: list[dict[str, Any]] = []
    for payload in method_payloads:
        method = str(payload["method"])
        label = str(payload["label"])
        root = experiment_root / "methods" / method / label
        anchor_payload = _read_json(
            root / "camera_extrinsics_anchor.json"
        )
        estimated = _anchor_pose_records(anchor_payload)
        complete_camera_set = set(estimated) == set(expected_cameras)
        rows = (
            _simulation_pairwise(
                method, label, estimated, gt_anchor_cameras
            )
            if complete_camera_set
            and set(gt_anchor_cameras) == set(expected_cameras)
            else []
        )
        pair_rows.extend(rows)
        evaluation_status = (
            "available"
            if len(rows)
            == len(expected_cameras) * (len(expected_cameras) - 1) // 2
            and bool(rows)
            else "evaluation_unavailable"
        )
        summaries.append(
            {
                "method": method,
                "label": label,
                "evaluation_status": evaluation_status,
                "expected_camera_count": len(expected_cameras),
                "evaluated_camera_count": len(estimated),
                "missing_cameras": sorted(
                    set(expected_cameras) - set(estimated)
                ),
                "reason": (
                    None
                    if evaluation_status == "available"
                    else (
                        "The direct anchor-relative estimate does not contain "
                        "the exact Ground-Truth camera set; no pair subset is "
                        "published as a complete evaluation."
                    )
                ),
                **_summary(rows),
            }
        )
        if anchor_marker_id is not None:
            direct_rows = _anchor_camera_gt_rows(
                method,
                label,
                anchor_payload,
                anchor_marker_id=anchor_marker_id,
                gt_cameras=gt_cameras,
                gt_markers=gt_markers,
            )
            anchor_gt_rows.extend(direct_rows)
            anchor_gt_summaries.append(
                {
                    "method": method,
                    "label": label,
                    "evaluation_status": (
                        "available"
                        if len(direct_rows) == len(expected_cameras)
                        and complete_camera_set
                        else "evaluation_unavailable"
                    ),
                    **_summary(direct_rows),
                }
            )
        try:
            map_rows.extend(
                _camera_map_rows(
                    method, label, estimated, gt_anchor_cameras
                )
            )
        except RuntimeError:
            pass
        if method == "ap02":
            marker_result, marker_text = _ap02_marker_map(
                root, gt_cameras, gt_markers
            )
            marker_results.append(marker_result)
            marker_texts.append(marker_text)
    evaluation_root = experiment_root / "evaluations"
    _write_csv(evaluation_root / "camera_pairwise_gt.csv", pair_rows)
    _write_json(
        evaluation_root / "camera_pairwise_gt.json",
        {"summaries": summaries, "rows": pair_rows},
    )
    _write_csv(evaluation_root / "camera_map_gt.csv", map_rows)
    _write_json(
        evaluation_root / "camera_map_gt.json",
        {"rows": map_rows},
    )
    _write_json(
        evaluation_root / "ap02_marker_map_gt.json",
        {"variants": marker_results},
    )
    _write_csv(
        evaluation_root / "anchor_camera_gt.csv", anchor_gt_rows
    )
    _write_json(
        evaluation_root / "anchor_camera_gt.json",
        {
            "anchor_marker_id": anchor_marker_id,
            "evaluation": (
                "direct_anchor_relative_posthoc_gt_no_fit_no_scale"
            ),
            "summaries": anchor_gt_summaries,
            "rows": anchor_gt_rows,
        },
    )
    map_text = _camera_map_text(experiment_root.name, map_rows)
    (experiment_root / "SECONDARY_CAMERA_MAP_RESULTS.txt").write_text(
        map_text, encoding="utf-8"
    )
    (experiment_root / "SECONDARY_AP02_MARKER_MAP_RESULTS.txt").write_text(
        "\n".join(marker_texts)
        if marker_texts
        else (
            "SECONDARY AP02 MARKER-MAP RESULTS\n"
            "=================================\n\n"
            "Unavailable: no successful AP02 result exists.\n"
        ),
        encoding="utf-8",
    )
    dataset = _read_json(dataset_root / "dataset.json")
    parameters = dataset.get("simulation_parameters", {}) or {}
    text = _simulation_primary_text(
        experiment_root.name,
        parameters,
        summaries,
        pair_rows,
        method_payloads,
        _read_json(
            dataset_root
            / "observations"
            / "SELECTION_CANDIDATES.json"
        ).get("evaluation_anchor", {}),
    )
    text += (
        "\n\nDIRECT COMMON-ANCHOR CAMERA POSES VS GROUND TRUTH\n"
        + "-" * 138
        + "\n"
        + (
            _text_table(
                [
                    "Method",
                    "Variant",
                    "Cameras",
                    "mean translation [cm]",
                    "max translation [cm]",
                    "mean rotation [deg]",
                    "max rotation [deg]",
                ],
                [
                    [
                        row["method"],
                        row["label"],
                        row["count"],
                        _fmt(row["mean_translation_error_cm"]),
                        _fmt(row["max_translation_error_cm"]),
                        _fmt(row["mean_rotation_error_deg"]),
                        _fmt(row["max_rotation_error_deg"]),
                    ]
                    for row in anchor_gt_summaries
                ],
            )
            if any(row["count"] for row in anchor_gt_summaries)
            else (
                "Unavailable: the frozen anchor or a method-specific "
                "anchor export is missing."
            )
        )
        + "\nDetailed values: evaluations/anchor_camera_gt.csv\n"
    )
    return text, {
        "category": "simulation",
        "experiment": experiment_root.name,
        "status": "available",
        "simulation_parameters": parameters,
        "storage": dataset.get("storage", {}),
        "ground_truth": {
            key: gt_payload.get(key)
            for key in (
                "snapshot_origin",
                "world_snapshot",
                "world_sha256",
                "camera_transform_convention",
                "marker_transform_convention",
            )
        },
        "methods": method_payloads,
        "baseline_contract": _baseline_contract(
            category="simulation",
            method_payloads=method_payloads,
            evaluation_anchor=selection.get("evaluation_anchor", {}),
        ),
        "scale_comparison": _scale_comparison_rows(method_payloads),
        "primary_camera_pairwise": {
            "summaries": summaries,
            "rows": pair_rows,
        },
        "anchor_camera_ground_truth": {
            "anchor_marker_id": anchor_marker_id,
            "summaries": anchor_gt_summaries,
            "rows": anchor_gt_rows,
            "path": "evaluations/anchor_camera_gt.csv",
        },
        "secondary_camera_map": {
            "rows": map_rows,
            "path": "SECONDARY_CAMERA_MAP_RESULTS.txt",
        },
        "secondary_ap02_marker_map": {
            "variants": marker_results,
            "path": "SECONDARY_AP02_MARKER_MAP_RESULTS.txt",
        },
    }


def _factor_report(factor_root: Path, factor: str) -> None:
    simulation_root = factor_root.parent
    experiment_payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(factor_root.glob("*/RESULTS.json")):
        payload = _read_json(path)
        if payload.get("category") == "simulation":
            experiment_payloads.append((path.parent.name, payload))
    for path in sorted(
        (simulation_root / "baseline").glob("*/RESULTS.json")
    ):
        payload = _read_json(path)
        if payload.get("category") == "simulation":
            experiment_payloads.append(
                (f"baseline/{path.parent.name}", payload)
            )
    if not any(not name.startswith("baseline/") for name, _ in experiment_payloads):
        return
    rows: list[dict[str, Any]] = []
    for name, payload in experiment_payloads:
        storage = payload.get("storage", {})
        if not name.startswith("baseline/") and storage.get("factor") != factor:
            continue
        value = (
            f"baseline ({storage.get('value', path_name(name))})"
            if name.startswith("baseline/")
            else str(storage.get("value") or name)
        )
        for row in payload.get("primary_camera_pairwise", {}).get("rows", []):
            rows.append({"factor_value": value, "experiment": name, **row})
    if not rows:
        return
    lines = [
        f"SIMULATION {factor.upper()} COMPARISON — CAMERA-TO-CAMERA VS GT",
        "=" * 142,
    ]
    for value, method, label in sorted(
        {
            (row["factor_value"], row["method"], row["label"])
            for row in rows
        }
    ):
        selected = [
            row
            for row in rows
            if (
                row["factor_value"],
                row["method"],
                row["label"],
            )
            == (value, method, label)
        ]
        summary = _summary(selected)
        lines.extend(
            [
                "",
                f"{value} — {method}/{label}",
                "-" * 142,
                (
                    f"Summary: mean {_fmt(summary['mean_translation_error_cm'])} "
                    f"cm / {_fmt(summary['mean_rotation_error_deg'])} deg; "
                    f"max {_fmt(summary['max_translation_error_cm'])} cm / "
                    f"{_fmt(summary['max_rotation_error_deg'])} deg"
                ),
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
                ),
            ]
        )
    lines.append("")
    (factor_root / "RESULTS.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_csv(factor_root / "RESULTS.csv", rows)
    _write_json(
        factor_root / "RESULTS.json",
        {
            "schema_version": 5,
            "layout_version": 2,
            "kind": "simulation_factor_comparison",
            "factor": factor,
            "rows": rows,
        },
    )


def path_name(value: str) -> str:
    return Path(value).name


def _refresh_factor_reports(experiment_root: Path, payload: dict[str, Any]) -> None:
    simulation_root = next(
        (
            parent
            for parent in experiment_root.parents
            if parent.name == "simulation"
        ),
        None,
    )
    if simulation_root is None:
        return
    factor = str(payload.get("storage", {}).get("factor", ""))
    if factor in {
        "route",
        "density",
        "resolution",
        "fov",
        "lighting",
        "motion_blur",
    }:
        _factor_report(simulation_root / factor, factor)
    elif factor == "baseline":
        for candidate in (
            "route",
            "density",
            "resolution",
            "fov",
            "lighting",
            "motion_blur",
        ):
            if (simulation_root / candidate).is_dir():
                _factor_report(simulation_root / candidate, candidate)


def _write_route2_baseline_comparison(
    experiment_root: Path,
    current: dict[str, Any],
) -> dict[str, Any] | None:
    """Compare the controlled CPU repair with the immutable Route-2 run."""

    if experiment_root.name != "route2_cpu_ref14_50x50":
        return None
    previous_root = experiment_root.parent / "route2"
    previous = _read_json(previous_root / "RESULTS.json")
    if previous.get("category") != "simulation":
        return None

    rows: list[dict[str, Any]] = []
    for experiment_name, payload in (
        ("route2", previous),
        (experiment_root.name, current),
    ):
        methods = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in payload.get("methods", [])
            if isinstance(item, dict)
        }
        summaries = payload.get("primary_camera_pairwise", {}).get(
            "summaries", []
        )
        anchor_summaries = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in payload.get(
                "anchor_camera_ground_truth", {}
            ).get("summaries", [])
            if isinstance(item, dict)
        }
        pair_rows = payload.get("primary_camera_pairwise", {}).get(
            "rows", []
        )
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            key = (
                str(summary.get("method")),
                str(summary.get("label")),
            )
            method = methods.get(key, {})
            anchor = anchor_summaries.get(key, {})
            edge5 = [
                item
                for item in pair_rows
                if isinstance(item, dict)
                and str(item.get("method")) == key[0]
                and str(item.get("label")) == key[1]
                and "cam_edge_5" in str(item.get("pair", ""))
            ]
            registration = method.get("metrics", {}).get(
                "ap03_registration", {}
            )
            rows.append(
                {
                    "experiment": experiment_name,
                    "method": key[0],
                    "label": key[1],
                    "runtime_seconds": method.get("runtime_seconds"),
                    "execution_status": method.get("execution_status"),
                    "solver_status": method.get("solver_status"),
                    "quality_status": method.get("quality_status"),
                    "pair_count": summary.get("count"),
                    "anchor_camera_count": anchor.get("count"),
                    "mean_pair_translation_error_cm": summary.get(
                        "mean_translation_error_cm"
                    ),
                    "mean_pair_rotation_error_deg": summary.get(
                        "mean_rotation_error_deg"
                    ),
                    "maximum_pair_translation_error_cm": summary.get(
                        "max_translation_error_cm"
                    ),
                    "maximum_pair_rotation_error_deg": summary.get(
                        "max_rotation_error_deg"
                    ),
                    "cam_edge_5_pair_count": len(edge5),
                    "cam_edge_5_maximum_translation_error_cm": _maximum(
                        item.get("translation_error_cm") for item in edge5
                    ),
                    "cam_edge_5_maximum_rotation_error_deg": _maximum(
                        item.get("rotation_error_deg") for item in edge5
                    ),
                    "registered_static_cameras": registration.get(
                        "registered_static_cameras"
                    ),
                    "registered_moving_frames": registration.get(
                        "registered_moving_frames"
                    ),
                    "sparse_points": registration.get("sparse_points"),
                    "configuration": method.get("config_summary", {}),
                }
            )
    if not rows:
        return None
    comparison_payload = {
        "schema_version": 5,
        "comparison": (
            "same_published_route2_input_no_alignment_no_best_fit"
        ),
        "old_experiment": "route2",
        "new_experiment": experiment_root.name,
        "method_rerun_of_old_experiment": False,
        "rows": rows,
    }
    _write_json(
        experiment_root / "BASELINE_COMPARISON.json",
        comparison_payload,
    )
    _write_csv(experiment_root / "BASELINE_COMPARISON.csv", rows)
    text = "\n".join(
        [
            "ROUTE-2 BASELINE REPAIR COMPARISON",
            "=" * 138,
            "",
            "Input: the same published Route-2 images and observations.",
            "Evaluation: direct common-anchor and camera-pair Ground Truth; "
            "no global alignment and no best-fit.",
            "",
            _text_table(
                [
                    "Experiment",
                    "Method",
                    "Variant",
                    "Runtime [s]",
                    "Quality",
                    "Pairs",
                    "Anchors",
                    "mean t [cm]",
                    "mean r [deg]",
                    "cam_edge_5 max t [cm]",
                    "cam_edge_5 max r [deg]",
                    "Static reg.",
                    "Moving reg.",
                    "Sparse points",
                ],
                [
                    [
                        row["experiment"],
                        row["method"],
                        row["label"],
                        _fmt(row["runtime_seconds"], 1),
                        row["quality_status"],
                        row["pair_count"],
                        row["anchor_camera_count"],
                        _fmt(row["mean_pair_translation_error_cm"]),
                        _fmt(row["mean_pair_rotation_error_deg"]),
                        _fmt(
                            row[
                                "cam_edge_5_maximum_translation_error_cm"
                            ]
                        ),
                        _fmt(
                            row[
                                "cam_edge_5_maximum_rotation_error_deg"
                            ]
                        ),
                        row["registered_static_cameras"],
                        row["registered_moving_frames"],
                        row["sparse_points"],
                    ]
                    for row in rows
                ],
            ),
            "",
        ]
    )
    _write_text(experiment_root / "BASELINE_COMPARISON.txt", text)
    return {
        "status": "available",
        "text": "BASELINE_COMPARISON.txt",
        "json": "BASELINE_COMPARISON.json",
        "csv": "BASELINE_COMPARISON.csv",
        "rows": len(rows),
    }


def write_scientific_experiment_reports(
    experiment_root: Path,
    *,
    dataset_root: Path,
    category: str,
) -> dict[str, Any]:
    """Write the canonical human and machine result front doors."""
    ensure_ap03_derived_results(experiment_root)
    if category == "simulation":
        resolve_simulation_ground_truth(dataset_root, backfilled=True)
    ensure_experiment_anchor_exports(experiment_root)
    method_payloads = refresh_method_reports(experiment_root)
    if category == "simulation":
        text, payload = _simulation_results(
            experiment_root, dataset_root, method_payloads
        )
    else:
        text, payload = _real_results_text(
            experiment_root, method_payloads, dataset_root
        )
    evaluation_by_method: dict[tuple[str, str], str] = {}
    evaluation_metrics_by_method: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    if category == "simulation":
        summaries = payload.get(
            "anchor_camera_ground_truth", {}
        ).get("summaries", [])
        pair_summaries = payload.get(
            "primary_camera_pairwise", {}
        ).get("summaries", [])
        evaluation_by_method = {
            (str(item.get("method")), str(item.get("label"))): str(
                item.get("evaluation_status", "evaluation_unavailable")
            )
            for item in summaries
            if isinstance(item, dict)
        }
        anchor_by_key = {
            (str(item.get("method")), str(item.get("label"))): item
            for item in summaries
            if isinstance(item, dict)
        }
        for item in pair_summaries:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("method")),
                str(item.get("label")),
            )
            evaluation_metrics_by_method[key] = {
                "pairwise_gt": item,
                "anchor_camera_gt": anchor_by_key.get(key, {}),
            }
    else:
        marker_available = bool(payload.get("marker_consistency_path"))
        evaluation_by_method = {
            (str(item.get("method")), str(item.get("label"))): (
                "available"
                if marker_available
                and bool(item.get("anchor_export_available"))
                else "unavailable"
            )
            for item in method_payloads
        }
    statuses_changed = False
    for result_path in sorted(
        (experiment_root / "methods").glob("*/*/RESULT.json")
    ):
        method_result = _read_json(result_path)
        key = (
            str(method_result.get("method") or result_path.parents[1].name),
            str(method_result.get("label") or result_path.parent.name),
        )
        evaluation_status = evaluation_by_method.get(key, "unavailable")
        current_metrics = (
            dict(method_result.get("metrics", {}))
            if isinstance(method_result.get("metrics"), dict)
            else {}
        )
        evaluation_metrics = evaluation_metrics_by_method.get(key)
        metrics_changed = (
            evaluation_metrics is not None
            and current_metrics.get("evaluation") != evaluation_metrics
        )
        if metrics_changed:
            current_metrics["evaluation"] = evaluation_metrics
            method_result["metrics"] = current_metrics
        if (
            method_result.get("evaluation_status") != evaluation_status
            or metrics_changed
        ):
            method_result["evaluation_status"] = evaluation_status
            _write_json(result_path, method_result)
            statuses_changed = True
    if statuses_changed:
        method_payloads = refresh_method_reports(experiment_root)
        if category == "simulation":
            text, payload = _simulation_results(
                experiment_root, dataset_root, method_payloads
            )
        else:
            text, payload = _real_results_text(
                experiment_root, method_payloads, dataset_root
            )
    visualization = ensure_visualization_artifacts(experiment_root)
    method_payloads = refresh_method_reports(experiment_root)
    if category == "simulation":
        text, payload = _simulation_results(
            experiment_root, dataset_root, method_payloads
        )
    else:
        text, payload = _real_results_text(
            experiment_root, method_payloads, dataset_root
        )
    baseline_comparison = (
        _write_route2_baseline_comparison(experiment_root, payload)
        if category == "simulation"
        else None
    )
    if baseline_comparison is not None:
        payload["baseline_comparison"] = baseline_comparison
        text = (
            text.rstrip()
            + "\n\n"
            + (
                experiment_root / "BASELINE_COMPARISON.txt"
            ).read_text(encoding="utf-8")
        )
    existing_results = _read_json(experiment_root / "RESULTS.json")
    generated_at = existing_results.get("generated_at") or _now()
    payload.update(
        {
            "schema_version": 5,
            "layout_version": 2,
            "generated_at": generated_at,
            "human_report": "RESULTS.txt",
            "visualization": visualization,
        }
    )
    text = (
        text.rstrip()
        + "\n\nRVIZ VISUALIZATION\n"
        + "-" * 72
        + "\n"
        + f"Status: {visualization.get('status', 'unavailable')}\n"
        + "Manifest: visualization/visualization_manifest.json\n"
        + (
            "Open from rigcal View results; each window uses an isolated "
            "ROS_DOMAIN_ID.\n"
            if visualization.get("available")
            else f"Reason: {visualization.get('reason', '-')}\n"
        )
    )
    _write_text(experiment_root / "RESULTS.txt", text)
    _write_json(experiment_root / "RESULTS.json", payload)
    for obsolete in ("SUMMARY.txt", "COMPARISON.txt"):
        (experiment_root / obsolete).unlink(missing_ok=True)
    if category == "simulation":
        _refresh_factor_reports(experiment_root, payload)
    return payload
