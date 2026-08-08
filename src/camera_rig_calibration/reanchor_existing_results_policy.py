from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_export.geometry import invert_transform, rigid_fit, robust_pose_average


_INSTALLED = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _score(row: dict[str, Any]) -> float:
    for key in ("selection_score", "quality", "marker_area_ratio", "area_px2"):
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return 1.0


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ap01_moving_relay_anchor(
    method_root: Path,
    config: Any,
    anchor_marker_id: int,
    camera_poses: dict[str, np.ndarray],
):
    from .anchor_export.adapters import AnchorResolution
    from .methods.ap01 import core

    accepted = _rows(
        method_root / "diagnostics" / "preflight" / "accepted_observations.csv"
    )
    if not accepted:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("No accepted observations are available for AP01 re-anchoring.",),
            {"method": "ap01", "reanchor": "moving_relay"},
        )

    images_candidates = [
        method_root
        / "diagnostics"
        / "method"
        / "moving_colmap"
        / "sparse_txt_best"
        / "images.txt",
        method_root
        / "diagnostics"
        / "method"
        / "moving_colmap"
        / "sparse_txt"
        / "0"
        / "images.txt",
    ]
    images_path = next((path for path in images_candidates if path.is_file()), None)
    scale_path = (
        method_root
        / "diagnostics"
        / "method"
        / "metric_scale"
        / "metric_scale.txt"
    )
    if images_path is None or not scale_path.is_file():
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("AP01 moving-COLMAP poses or metric scale are unavailable.",),
            {
                "method": "ap01",
                "reanchor": "moving_relay",
                "images_candidates": [str(path) for path in images_candidates],
                "scale_file": str(scale_path),
            },
        )
    try:
        poses = core.parse_colmap_poses(images_path)
        scale = float(scale_path.read_text(encoding="utf-8").strip())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            (str(exc),),
            {"method": "ap01", "reanchor": "moving_relay"},
        )
    if not math.isfinite(scale) or scale <= 0.0:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("AP01 metric scale is invalid.",),
            {"method": "ap01", "reanchor": "moving_relay"},
        )

    static_best: dict[tuple[str, int], dict[str, Any]] = {}
    moving_by_marker: dict[int, list[dict[str, Any]]] = {}
    for row in accepted:
        marker = _int(row.get("marker_id"))
        if marker is None:
            continue
        observer_type = str(row.get("observer_type", "")).strip().lower()
        if observer_type == "static":
            camera = str(row.get("observer_id") or row.get("camera_name") or "").strip()
            if camera not in camera_poses:
                continue
            key = (camera, marker)
            if key not in static_best or _score(row) > _score(static_best[key]):
                static_best[key] = row
        elif observer_type == "moving":
            try:
                frame = core.frame_number(row)
            except RuntimeError:
                continue
            if frame not in poses:
                continue
            item = dict(row)
            item["_frame"] = frame
            moving_by_marker.setdefault(marker, []).append(item)

    anchor_rows = sorted(
        moving_by_marker.get(anchor_marker_id, []),
        key=_score,
        reverse=True,
    )[: max(8, min(40, int(getattr(config.methods.ap01, "top_moving_per_marker", 8) or 8) * 2))]
    if not anchor_rows:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("No registered AP01 moving frame observes the selected anchor.",),
            {
                "method": "ap01",
                "reanchor": "moving_relay",
                "anchor_marker_id": anchor_marker_id,
            },
        )

    per_bridge: dict[int, list[tuple[np.ndarray, float, dict[str, Any]]]] = {}
    per_marker_limit = int(getattr(config.methods.ap01, "top_moving_per_marker", 8) or 8)
    for (camera, bridge_marker), static_row in sorted(static_best.items()):
        if bridge_marker == anchor_marker_id:
            continue
        bridge_rows = sorted(
            moving_by_marker.get(bridge_marker, []),
            key=_score,
            reverse=True,
        )[:per_marker_limit]
        if not bridge_rows:
            continue
        try:
            T_method_bridge = camera_poses[camera] @ core.T_from_observation(static_row)
        except RuntimeError:
            continue
        for bridge_moving in bridge_rows:
            try:
                frame_i = int(bridge_moving["_frame"])
                T_method_moving_i = (
                    T_method_bridge
                    @ invert_transform(core.T_from_observation(bridge_moving))
                )
            except (RuntimeError, ValueError):
                continue
            for anchor_moving in anchor_rows:
                frame_j = int(anchor_moving["_frame"])
                try:
                    T_i_j = poses[frame_i] @ invert_transform(poses[frame_j])
                    T_i_j = np.asarray(T_i_j, dtype=np.float64).copy()
                    T_i_j[:3, 3] *= scale
                    candidate = (
                        T_method_moving_i
                        @ T_i_j
                        @ core.T_from_observation(anchor_moving)
                    )
                except (KeyError, RuntimeError, ValueError):
                    continue
                quality = (
                    max(_score(static_row), 1e-9)
                    * max(_score(bridge_moving), 1e-9)
                    * max(_score(anchor_moving), 1e-9)
                ) ** (1.0 / 3.0)
                per_bridge.setdefault(bridge_marker, []).append(
                    (
                        candidate,
                        quality,
                        {
                            "static_camera": camera,
                            "bridge_marker_id": bridge_marker,
                            "bridge_frame": frame_i,
                            "anchor_frame": frame_j,
                        },
                    )
                )

    bridge_estimates: list[np.ndarray] = []
    bridge_weights: list[float] = []
    bridge_diagnostics: list[dict[str, Any]] = []
    for bridge_marker, candidates in sorted(per_bridge.items()):
        transforms = [item[0] for item in candidates]
        weights = [item[1] for item in candidates]
        try:
            aggregate = robust_pose_average(transforms, weights)
        except ValueError:
            continue
        bridge_estimates.append(aggregate.transform)
        inlier_weights = [weights[index] for index in aggregate.inlier_indices]
        bridge_weights.append(float(np.mean(inlier_weights)) if inlier_weights else 1.0)
        bridge_diagnostics.append(
            {
                "bridge_marker_id": bridge_marker,
                "candidate_count": len(candidates),
                "inlier_count": len(aggregate.inlier_indices),
                "translation_threshold_m": aggregate.translation_threshold_m,
                "rotation_threshold_deg": aggregate.rotation_threshold_deg,
                "support": [candidates[index][2] for index in aggregate.inlier_indices[:20]],
            }
        )
    if not bridge_estimates:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_AVAILABLE",
            False,
            ("No robust AP01 moving-relay chain reconstructs the selected anchor.",),
            {
                "method": "ap01",
                "reanchor": "moving_relay",
                "anchor_marker_id": anchor_marker_id,
                "registered_moving_frames": len(poses),
            },
        )
    try:
        final = robust_pose_average(bridge_estimates, bridge_weights)
    except ValueError as exc:
        return AnchorResolution(
            None,
            "ANCHOR_POSE_DEGENERATE",
            False,
            (str(exc),),
            {
                "method": "ap01",
                "reanchor": "moving_relay",
                "bridge_estimates": bridge_diagnostics,
            },
        )
    code = "OK_MOVING_RELAY_ANCHOR" if len(final.inlier_indices) >= 2 else "WEAK_SINGLE_RELAY_CHAIN"
    warnings: tuple[str, ...] = ()
    if code != "OK_MOVING_RELAY_ANCHOR":
        warnings = (
            "Only one independent AP01 bridge-marker estimate supports the common anchor.",
        )
    return AnchorResolution(
        final.transform,
        code,
        True,
        warnings,
        {
            "method": "ap01",
            "alignment": "existing_moving_colmap_relay_to_anchor_no_method_rerun",
            "anchor_marker_id": anchor_marker_id,
            "registered_moving_frames": len(poses),
            "metric_scale_m_per_colmap_unit": scale,
            "anchor_moving_observation_count": len(anchor_rows),
            "bridge_marker_estimate_count": len(bridge_estimates),
            "bridge_marker_inlier_count": len(final.inlier_indices),
            "translation_threshold_m": final.translation_threshold_m,
            "rotation_threshold_deg": final.rotation_threshold_deg,
            "bridge_estimates": bridge_diagnostics,
            "ground_truth_used": False,
        },
    )


def _ap03_existing_corner_anchor(
    method_root: Path,
    config: Any,
    anchor_marker_id: int,
    *,
    mode: str,
):
    from .anchor_export.adapters import AnchorResolution

    provenance = _read_json(method_root / "provenance" / "derived_result.json")
    try:
        scale = float(provenance["scale_m_per_colmap_unit"])
        experiment_root = method_root.parents[2]
        container = experiment_root / str(provenance["shared_colmap_container"])
    except (KeyError, TypeError, ValueError):
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            (f"AP03 {mode.title()} provenance cannot resolve the existing COLMAP geometry.",),
            {"method": f"ap03_{mode}"},
        )
    corners_path = (
        container
        / "diagnostics"
        / "method"
        / "scale_multi"
        / "AP03_MARKER_SIZE_SCALE_ONLY_TRIANGULATED_CORNERS.csv"
    )
    selected: dict[int, np.ndarray] = {}
    support: dict[int, dict[str, Any]] = {}
    for row in _rows(corners_path):
        marker = _int(row.get("marker_id"))
        index = _int(row.get("corner_idx"))
        if marker != anchor_marker_id or index not in {0, 1, 2, 3}:
            continue
        if str(row.get("status", "OK")).strip().upper() != "OK":
            continue
        try:
            point = scale * np.asarray(
                [
                    float(row["x_colmap"]),
                    float(row["y_colmap"]),
                    float(row["z_colmap"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(point)):
            continue
        selected[index] = point
        support[index] = {
            "observation_count": row.get("obs_count"),
            "inlier_count": row.get("inlier_count"),
            "median_reprojection_px": row.get("median_reproj_px"),
        }
    if set(selected) != {0, 1, 2, 3}:
        return AnchorResolution(
            None,
            "ANCHOR_NOT_RECONSTRUCTED",
            False,
            (
                f"Existing AP03 {mode.title()} COLMAP geometry has fewer than four "
                f"valid corners for marker {anchor_marker_id}."
            ,),
            {
                "method": f"ap03_{mode}",
                "anchor_marker_id": anchor_marker_id,
                "available_corner_indices": sorted(selected),
                "corner_file": str(corners_path),
            },
        )
    half = float(config.markers.length_m) / 2.0
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
            {"method": f"ap03_{mode}", "corner_file": str(corners_path)},
        )
    return AnchorResolution(
        transform,
        "OK_REANCHORED_EXISTING_COLMAP",
        True,
        (),
        {
            "method": f"ap03_{mode}",
            "alignment": "existing_colmap_marker_corners_rigid_fit_no_method_rerun",
            "anchor_marker_id": anchor_marker_id,
            "scale_mode": mode,
            "scale_m_per_colmap_unit": scale,
            "square_fit_rmse_m": rmse,
            "side_lengths_m": [
                float(np.linalg.norm(observed[(index + 1) % 4] - observed[index]))
                for index in range(4)
            ],
            "corner_support": [
                {"corner_idx": index, **support.get(index, {})}
                for index in range(4)
            ],
            "corner_file": str(corners_path),
            "shared_colmap_container": provenance.get("shared_colmap_container"),
            "shared_colmap_best_model": provenance.get("shared_colmap_best_model"),
            "ground_truth_used": False,
        },
    )


def install_reanchor_existing_results_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from .anchor_export import adapters, exporter

    original = adapters.resolve_method_anchor
    if getattr(original, "_rigcal_existing_result_reanchor", False):
        _INSTALLED = True
        return

    def resolve_method_anchor(method_root, config, method_id, anchor_marker_id, camera_poses):
        result = original(method_root, config, method_id, anchor_marker_id, camera_poses)
        if result.available:
            return result
        if method_id == "ap01":
            relay = _ap01_moving_relay_anchor(
                Path(method_root), config, int(anchor_marker_id), camera_poses
            )
            return relay if relay.available else result
        if method_id in {"ap03_single", "ap03_multi"}:
            existing = _ap03_existing_corner_anchor(
                Path(method_root),
                config,
                int(anchor_marker_id),
                mode=method_id.removeprefix("ap03_"),
            )
            return existing if existing.available else result
        return result

    resolve_method_anchor._rigcal_existing_result_reanchor = True  # type: ignore[attr-defined]
    adapters.resolve_method_anchor = resolve_method_anchor
    # exporter imported the function by name, so bind the enhanced resolver there too.
    exporter.resolve_method_anchor = resolve_method_anchor
    _INSTALLED = True
