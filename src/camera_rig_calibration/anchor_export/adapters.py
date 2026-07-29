from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config.models import RigConfig
from .geometry import (
    make_transform,
    rigid_fit,
    robust_pose_average,
    rvec_to_rotation,
)


@dataclass(frozen=True)
class AnchorResolution:
    transform_method_anchor: np.ndarray | None
    code: str
    available: bool
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def row_transform(row: dict[str, str]) -> np.ndarray:
    rotation = rvec_to_rotation(
        (_number(row, "rvec_x"), _number(row, "rvec_y"), _number(row, "rvec_z"))
    )
    translation = np.asarray(
        (_number(row, "x_m"), _number(row, "y_m"), _number(row, "z_m")),
        dtype=np.float64,
    )
    return make_transform(rotation, translation)


def load_camera_poses(method_root: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in _rows(method_root / "camera_extrinsics.csv"):
        camera = str(
            row.get("entity_id")
            or row.get("camera_id")
            or row.get("camera_name")
            or ""
        ).strip()
        if not camera:
            continue
        result[camera] = row_transform(row)
    return result


def _ap01(
    method_root: Path,
    config: RigConfig,
    anchor_marker_id: int,
    camera_poses: dict[str, np.ndarray],
) -> AnchorResolution:
    accepted = _rows(
        method_root / "diagnostics" / "preflight" / "accepted_observations.csv"
    )
    candidates: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    weights: list[float] = []
    for row in sorted(
        accepted,
        key=lambda item: (
            str(item.get("observer_id") or item.get("camera_name") or ""),
            str(item.get("frame_id") or ""),
            str(item.get("image_path") or ""),
        ),
    ):
        try:
            marker_id = int(float(row.get("marker_id", "")))
        except ValueError:
            continue
        camera_id = str(
            row.get("observer_id") or row.get("camera_name") or ""
        ).strip()
        if (
            marker_id != anchor_marker_id
            or row.get("observer_type") != "static"
            or camera_id not in camera_poses
        ):
            continue
        try:
            camera_anchor = make_transform(
                rvec_to_rotation(
                    (
                        _number(row, "rvec_x"),
                        _number(row, "rvec_y"),
                        _number(row, "rvec_z"),
                    )
                ),
                np.asarray(
                    (
                        _number(row, "tvec_x_m"),
                        _number(row, "tvec_y_m"),
                        _number(row, "tvec_z_m"),
                    )
                ),
            )
            candidate = camera_poses[camera_id] @ camera_anchor
            weight = max(float(row.get("selection_score") or 1.0), 1e-9)
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(candidate)
        weights.append(weight)
        evidence.append(
            {
                "camera_id": camera_id,
                "frame_id": row.get("frame_id"),
                "image_path": row.get("image_path"),
                "selection_score": weight,
            }
        )
    if not candidates:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("No accepted static-camera observation reconstructs the selected anchor.",),
            {"method": "ap01", "candidates": []},
        )
    try:
        aggregate = robust_pose_average(candidates, weights)
    except ValueError as exc:
        return AnchorResolution(
            None,
            "ANCHOR_POSE_DEGENERATE",
            False,
            (str(exc),),
            {"method": "ap01", "candidates": evidence},
        )
    diagnostics = {
        "method": "ap01",
        "aggregation": "weighted_markley_rotation_and_translation_with_mad_outlier_rejection",
        "candidate_count": len(candidates),
        "inlier_count": len(aggregate.inlier_indices),
        "translation_threshold_m": aggregate.translation_threshold_m,
        "rotation_threshold_deg": aggregate.rotation_threshold_deg,
        "candidates": [
            {
                **item,
                "translation_residual_m": aggregate.translation_residuals_m[index],
                "rotation_residual_deg": aggregate.rotation_residuals_deg[index],
                "inlier": index in aggregate.inlier_indices,
            }
            for index, item in enumerate(evidence)
        ],
    }
    if len(candidates) == 1:
        return AnchorResolution(
            aggregate.transform,
            "WEAK_SINGLE_ANCHOR_OBSERVATION",
            True,
            ("Only one accepted static observation supports the anchor pose.",),
            diagnostics,
        )
    if len(aggregate.inlier_indices) < 2:
        return AnchorResolution(
            None,
            "ANCHOR_POSE_DEGENERATE",
            False,
            ("Fewer than two robust AP01 anchor observations remain.",),
            diagnostics,
        )
    return AnchorResolution(aggregate.transform, "OK", True, (), diagnostics)


def _ap02(
    method_root: Path,
    config: RigConfig,
    anchor_marker_id: int,
) -> AnchorResolution:
    reference = config.methods.ap02.reference_marker_id
    if isinstance(reference, int) and reference == anchor_marker_id:
        return AnchorResolution(
            np.eye(4, dtype=np.float64),
            "OK",
            True,
            (),
            {
                "method": "ap02",
                "alignment": "identity",
                "reference_marker_id": reference,
            },
        )
    marker_file = (
        method_root
        / "diagnostics"
        / "method"
        / "graph_ba"
        / "with_moving"
        / "optimized_marker_poses_ref_marker.csv"
    )
    for row in _rows(marker_file):
        try:
            marker_id = int(float(row.get("entity_id") or row.get("marker_id") or ""))
        except ValueError:
            continue
        if marker_id != anchor_marker_id:
            continue
        try:
            transform = row_transform(row)
        except (KeyError, ValueError) as exc:
            return AnchorResolution(
                None,
                "ANCHOR_POSE_DEGENERATE",
                False,
                (str(exc),),
                {"method": "ap02", "marker_pose_file": str(marker_file)},
            )
        return AnchorResolution(
            transform,
            "OK",
            True,
            (),
            {
                "method": "ap02",
                "alignment": "combined_ba_marker_pose",
                "reference_marker_id": reference,
                "marker_pose_file": str(marker_file),
            },
        )
    return AnchorResolution(
        None,
        "ANCHOR_NOT_RECONSTRUCTED",
        False,
        ("The AP02 Combined BA result has no pose for the selected anchor.",),
        {"method": "ap02", "marker_pose_file": str(marker_file)},
    )


def _ap03(
    method_root: Path,
    config: RigConfig,
    anchor_marker_id: int,
) -> AnchorResolution:
    scale_root = method_root / "diagnostics" / "method" / "scale_multi"
    metadata_path = scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    corners_path = scale_root / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scale = float(metadata["scale_m_per_colmap_unit"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            ("AP03 Multi scale metadata is unavailable.",),
            {"method": "ap03", "metadata_file": str(metadata_path)},
        )
    selected: dict[int, tuple[np.ndarray, dict[str, str]]] = {}
    for row in _rows(corners_path):
        try:
            marker_id = int(float(row["marker_id"]))
            index = int(float(row["corner_idx"]))
        except (KeyError, ValueError):
            continue
        if marker_id != anchor_marker_id or row.get("status", "OK") != "OK":
            continue
        try:
            selected[index] = (
                scale
                * np.asarray(
                    (
                        _number(row, "x_colmap"),
                        _number(row, "y_colmap"),
                        _number(row, "z_colmap"),
                    )
                ),
                row,
            )
        except (KeyError, ValueError):
            continue
    if set(selected) != {0, 1, 2, 3}:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            ("AP03 did not triangulate all four selected-anchor corners.",),
            {
                "method": "ap03",
                "available_corner_indices": sorted(selected),
                "corner_file": str(corners_path),
            },
        )
    half = config.markers.length_m / 2.0
    ideal = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    observed = np.vstack([selected[index][0] for index in range(4)])
    try:
        transform, rmse = rigid_fit(ideal, observed)
    except ValueError as exc:
        return AnchorResolution(
            None,
            "ANCHOR_POSE_DEGENERATE",
            False,
            (str(exc),),
            {"method": "ap03", "corner_file": str(corners_path)},
        )
    side_lengths = [
        float(np.linalg.norm(observed[(index + 1) % 4] - observed[index]))
        for index in range(4)
    ]
    return AnchorResolution(
        transform,
        "OK",
        True,
        (),
        {
            "method": "ap03",
            "alignment": "rigid_kabsch_no_scale_fit",
            "scale_m_per_colmap_unit": scale,
            "square_fit_rmse_m": rmse,
            "side_lengths_m": side_lengths,
            "corner_support": [
                {
                    "corner_idx": index,
                    "observation_count": selected[index][1].get("obs_count"),
                    "inlier_count": selected[index][1].get("inlier_count"),
                    "median_reprojection_px": selected[index][1].get("median_reproj_px"),
                }
                for index in range(4)
            ],
            "corner_file": str(corners_path),
            "metadata_file": str(metadata_path),
        },
    )


def _ap03_derived(
    method_root: Path,
    config: RigConfig,
    anchor_marker_id: int,
    *,
    mode: str,
) -> AnchorResolution:
    """Resolve an AP03 scale-specific anchor from the shared COLMAP geometry."""
    provenance_path = method_root / "provenance" / "derived_result.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        scale = float(provenance["scale_m_per_colmap_unit"])
        experiment_root = method_root.parents[2]
        corner_path = experiment_root / str(
            provenance["shared_anchor_geometry"]
        )
        corner_payload = json.loads(corner_path.read_text(encoding="utf-8"))
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            (f"AP03 {mode.title()} shared anchor geometry is unavailable.",),
            {
                "method": f"ap03_{mode}",
                "provenance_file": str(provenance_path),
            },
        )
    try:
        payload_anchor = int(corner_payload["anchor_marker_id"])
    except (KeyError, TypeError, ValueError):
        payload_anchor = -1
    if payload_anchor != anchor_marker_id:
        return AnchorResolution(
            None,
            "ANCHOR_SOURCE_MISMATCH",
            False,
            (
                "The shared AP03 anchor geometry belongs to marker "
                f"{payload_anchor}, not marker {anchor_marker_id}.",
            ),
            {
                "method": f"ap03_{mode}",
                "corner_file": str(corner_path),
            },
        )
    selected: dict[int, np.ndarray] = {}
    for row in corner_payload.get("corners", []):
        try:
            index = int(row["corner_idx"])
            point = scale * np.asarray(
                (
                    float(row["x_colmap"]),
                    float(row["y_colmap"]),
                    float(row["z_colmap"]),
                ),
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if index in {0, 1, 2, 3} and np.all(np.isfinite(point)):
            selected[index] = point
    if set(selected) != {0, 1, 2, 3}:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            (
                f"AP03 {mode.title()} has fewer than four valid common-anchor "
                "corners.",
            ),
            {
                "method": f"ap03_{mode}",
                "available_corner_indices": sorted(selected),
                "corner_file": str(corner_path),
            },
        )
    half = config.markers.length_m / 2.0
    ideal = np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    observed = np.vstack([selected[index] for index in range(4)])
    try:
        transform, rmse = rigid_fit(ideal, observed)
    except ValueError as exc:
        return AnchorResolution(
            None,
            "ANCHOR_POSE_DEGENERATE",
            False,
            (str(exc),),
            {"method": f"ap03_{mode}", "corner_file": str(corner_path)},
        )
    return AnchorResolution(
        transform,
        "OK",
        True,
        (),
        {
            "method": f"ap03_{mode}",
            "alignment": "rigid_kabsch_no_scale_fit",
            "scale_mode": mode,
            "scale_m_per_colmap_unit": scale,
            "square_fit_rmse_m": rmse,
            "side_lengths_m": [
                float(
                    np.linalg.norm(
                        observed[(index + 1) % 4] - observed[index]
                    )
                )
                for index in range(4)
            ],
            "corner_file": str(corner_path),
            "provenance_file": str(provenance_path),
            "shared_colmap_container": provenance.get(
                "shared_colmap_container"
            ),
            "shared_colmap_best_model": provenance.get(
                "shared_colmap_best_model"
            ),
        },
    )


def resolve_method_anchor(
    method_root: Path,
    config: RigConfig,
    method_id: str,
    anchor_marker_id: int,
    camera_poses: dict[str, np.ndarray],
) -> AnchorResolution:
    if method_id == "ap01":
        return _ap01(method_root, config, anchor_marker_id, camera_poses)
    if method_id == "ap02":
        return _ap02(method_root, config, anchor_marker_id)
    if method_id == "ap03":
        return _ap03(method_root, config, anchor_marker_id)
    if method_id in {"ap03_single", "ap03_multi"}:
        return _ap03_derived(
            method_root,
            config,
            anchor_marker_id,
            mode=method_id.removeprefix("ap03_"),
        )
    return AnchorResolution(
        None,
        "UNSUPPORTED_METHOD",
        False,
        (f"Method '{method_id}' has no anchor adapter.",),
        {"method": method_id},
    )
