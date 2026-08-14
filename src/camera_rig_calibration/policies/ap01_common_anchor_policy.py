from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..anchor_export.geometry import (
    invert_transform,
    make_transform,
    robust_pose_average,
    rotation_error_deg,
    rvec_to_rotation,
)


_INSTALLED = False


def _category(config: Any) -> str:
    value = getattr(getattr(config, "dataset", None), "category", "real_vehicle")
    return str(getattr(value, "value", value))


def _score(row: dict[str, str]) -> float:
    try:
        value = float(row.get("selection_score") or 1.0)
    except (TypeError, ValueError):
        value = 1.0
    return max(value if math.isfinite(value) else 1.0, 1e-9)


def _marker(row: dict[str, str]) -> int | None:
    try:
        return int(float(row.get("marker_id", "")))
    except (TypeError, ValueError):
        return None


def _camera(row: dict[str, str]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


def _frame(row: dict[str, str]) -> int | None:
    for key in ("frame_id", "observer_id", "image_path"):
        matches = re.findall(r"(\d+)", str(row.get(key) or ""))
        if matches:
            return int(matches[-1])
    return None


def _observation_transform(row: dict[str, str]) -> np.ndarray:
    from ..anchor_export import adapters

    return make_transform(
        rvec_to_rotation(
            (
                adapters._number(row, "rvec_x"),
                adapters._number(row, "rvec_y"),
                adapters._number(row, "rvec_z"),
            )
        ),
        np.asarray(
            (
                adapters._number(row, "tvec_x_m"),
                adapters._number(row, "tvec_y_m"),
                adapters._number(row, "tvec_z_m"),
            ),
            dtype=np.float64,
        ),
    )


def _ap01_metric_colmap_poses(method_root: Path) -> tuple[dict[int, np.ndarray], float, str]:
    from ..methods.ap01.core import parse_colmap_poses

    diagnostics = method_root / "diagnostics" / "method"
    diag_path = diagnostics / "static_extrinsics" / "AP01_DIAGNOSTICS.json"
    if not diag_path.is_file():
        candidates = sorted(diagnostics.rglob("AP01_DIAGNOSTICS.json"))
        if not candidates:
            raise RuntimeError("AP01 diagnostics do not contain metric-scale metadata")
        diag_path = candidates[0]
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    scale = float(payload["metric_scale"]["scale_m_per_colmap_unit"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("AP01 metric COLMAP scale is invalid")

    images_path = diagnostics / "moving_colmap" / "sparse_txt_best" / "images.txt"
    if not images_path.is_file():
        candidates = sorted(diagnostics.rglob("sparse_txt_best/images.txt"))
        if not candidates:
            raise RuntimeError("AP01 metric moving COLMAP poses are unavailable")
        images_path = candidates[0]

    world_to_camera = parse_colmap_poses(images_path)
    world_camera: dict[int, np.ndarray] = {}
    for frame, transform in world_to_camera.items():
        scaled = transform.copy()
        scaled[:3, 3] *= scale
        world_camera[int(frame)] = invert_transform(scaled)
    return world_camera, scale, str(images_path)


def _best_static_rows(
    rows: list[dict[str, str]], camera_poses: dict[str, np.ndarray]
) -> dict[tuple[str, int], dict[str, str]]:
    best: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if str(row.get("observer_type")) != "static":
            continue
        camera_id = _camera(row)
        marker_id = _marker(row)
        if marker_id is None or camera_id not in camera_poses:
            continue
        try:
            _observation_transform(row)
        except (KeyError, TypeError, ValueError):
            continue
        key = (camera_id, marker_id)
        if key not in best or _score(row) > _score(best[key]):
            best[key] = row
    return best


def _best_moving_rows(
    rows: list[dict[str, str]], registered_frames: set[int]
) -> dict[tuple[int, int], dict[str, str]]:
    best: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if str(row.get("observer_type")) != "moving":
            continue
        marker_id = _marker(row)
        frame = _frame(row)
        if marker_id is None or frame is None or frame not in registered_frames:
            continue
        try:
            _observation_transform(row)
        except (KeyError, TypeError, ValueError):
            continue
        key = (frame, marker_id)
        if key not in best or _score(row) > _score(best[key]):
            best[key] = row
    return best


def _average_payload(result: Any) -> dict[str, Any]:
    return {
        "candidate_count": len(result.translation_residuals_m),
        "inlier_count": len(result.inlier_indices),
        "translation_threshold_m": result.translation_threshold_m,
        "rotation_threshold_deg": result.rotation_threshold_deg,
        "translation_residuals_m": list(result.translation_residuals_m),
        "rotation_residuals_deg": list(result.rotation_residuals_deg),
    }


def _moving_bridge_anchor(
    method_root: Path,
    anchor_marker_id: int,
    camera_poses: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any], tuple[str, ...]]:
    from ..anchor_export import adapters

    rows = adapters._rows(
        method_root / "diagnostics" / "preflight" / "accepted_observations.csv"
    )
    if not rows:
        return None, {"reason": "accepted observations are unavailable"}, ()

    try:
        world_camera, scale, images_path = _ap01_metric_colmap_poses(method_root)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return None, {"reason": str(exc)}, ()

    static_best = _best_static_rows(rows, camera_poses)
    moving_best = _best_moving_rows(rows, set(world_camera))
    moving_by_marker: dict[int, list[dict[str, str]]] = {}
    for (_, marker_id), row in moving_best.items():
        moving_by_marker.setdefault(marker_id, []).append(row)

    # First determine one AP01-method <- COLMAP-world transform per independent
    # bridge marker.  A marker may be seen by several solved static cameras and
    # many moving frames; those correlated samples are collapsed before the
    # independent marker-level consensus is formed.
    bridge_estimates: list[np.ndarray] = []
    bridge_weights: list[float] = []
    bridge_reports: list[dict[str, Any]] = []
    for marker_id in sorted(moving_by_marker):
        candidates: list[np.ndarray] = []
        weights: list[float] = []
        support_cameras: set[str] = set()
        support_frames: set[int] = set()
        for (camera_id, static_marker), static_row in static_best.items():
            if static_marker != marker_id:
                continue
            method_marker = camera_poses[camera_id] @ _observation_transform(static_row)
            for moving_row in moving_by_marker[marker_id]:
                frame = _frame(moving_row)
                if frame is None or frame not in world_camera:
                    continue
                method_moving = method_marker @ invert_transform(
                    _observation_transform(moving_row)
                )
                method_world = method_moving @ invert_transform(world_camera[frame])
                candidates.append(method_world)
                weights.append(math.sqrt(_score(static_row) * _score(moving_row)))
                support_cameras.add(camera_id)
                support_frames.add(frame)
        if not candidates:
            continue
        try:
            aggregate = robust_pose_average(candidates, weights)
        except ValueError:
            continue
        bridge_estimates.append(aggregate.transform)
        bridge_weights.append(
            float(np.mean([weights[index] for index in aggregate.inlier_indices]))
        )
        bridge_reports.append(
            {
                "marker_id": marker_id,
                "static_cameras": sorted(support_cameras),
                "moving_frames": sorted(support_frames),
                **_average_payload(aggregate),
            }
        )

    if not bridge_estimates:
        return None, {
            "reason": "no static-to-moving bridge marker aligns AP01 with its metric COLMAP trajectory",
            "metric_scale_m_per_colmap_unit": scale,
            "colmap_pose_source": images_path,
        }, ()
    try:
        world_alignment = robust_pose_average(bridge_estimates, bridge_weights)
    except ValueError as exc:
        return None, {"reason": str(exc), "bridge_markers": bridge_reports}, ()

    anchor_rows = [
        row
        for (frame, marker_id), row in sorted(moving_best.items())
        if marker_id == anchor_marker_id and frame in world_camera
    ]
    if len(anchor_rows) < 2:
        return None, {
            "reason": "fewer than two registered moving observations support the requested AP01 anchor",
            "bridge_markers": bridge_reports,
            "bridge_consensus": _average_payload(world_alignment),
        }, ()

    anchor_candidates: list[np.ndarray] = []
    anchor_weights: list[float] = []
    anchor_frames: list[int] = []
    for row in anchor_rows:
        frame = _frame(row)
        if frame is None:
            continue
        anchor_candidates.append(
            world_alignment.transform
            @ world_camera[frame]
            @ _observation_transform(row)
        )
        anchor_weights.append(_score(row))
        anchor_frames.append(frame)
    try:
        anchor_average = robust_pose_average(anchor_candidates, anchor_weights)
    except ValueError as exc:
        return None, {"reason": str(exc), "bridge_markers": bridge_reports}, ()
    if len(anchor_average.inlier_indices) < 2:
        return None, {
            "reason": "AP01 moving-anchor consensus retained fewer than two inliers",
            "bridge_markers": bridge_reports,
            "anchor_consensus": _average_payload(anchor_average),
        }, ()

    diagnostics: dict[str, Any] = {
        "method": "ap01",
        "alignment": "metric_moving_colmap_bridge_marker_consensus",
        "metric_scale_m_per_colmap_unit": scale,
        "colmap_pose_source": images_path,
        "independent_bridge_marker_count": len(bridge_estimates),
        "bridge_markers": bridge_reports,
        "bridge_consensus": _average_payload(world_alignment),
        "anchor_marker_id": anchor_marker_id,
        "anchor_moving_frames": anchor_frames,
        "anchor_consensus": _average_payload(anchor_average),
        "ground_truth_used": False,
    }

    # Direct static observations remain a useful cross-check, but they no longer
    # own the whole output frame when marker 0 has only one static supporter.
    direct_candidates: list[np.ndarray] = []
    direct_weights: list[float] = []
    direct_cameras: list[str] = []
    for (camera_id, marker_id), row in static_best.items():
        if marker_id != anchor_marker_id:
            continue
        direct_candidates.append(camera_poses[camera_id] @ _observation_transform(row))
        direct_weights.append(_score(row))
        direct_cameras.append(camera_id)
    warnings: list[str] = []
    if direct_candidates:
        try:
            direct = robust_pose_average(direct_candidates, direct_weights)
            translation_disagreement = float(
                np.linalg.norm(
                    direct.transform[:3, 3] - anchor_average.transform[:3, 3]
                )
            )
            rotation_disagreement = rotation_error_deg(
                direct.transform, anchor_average.transform
            )
            diagnostics["direct_static_cross_check"] = {
                "static_cameras": direct_cameras,
                **_average_payload(direct),
                "translation_disagreement_m": translation_disagreement,
                "rotation_disagreement_deg": rotation_disagreement,
            }
            if translation_disagreement > 0.25 or rotation_disagreement > 5.0:
                warnings.append(
                    "Direct static marker-0 PnP disagrees with the AP01 moving-trajectory anchor consensus; moving consensus is authoritative for the export frame."
                )
        except ValueError:
            pass

    return anchor_average.transform, diagnostics, tuple(warnings)


def install_ap01_common_anchor_policy() -> None:
    """Use robust moving-trajectory support for AP01's real marker-0 output frame."""

    global _INSTALLED
    if _INSTALLED:
        return
    from ..anchor_export import adapters

    original = adapters._ap01
    if getattr(original, "_rigcal_ap01_marker_zero_bridge", False):
        _INSTALLED = True
        return

    def ap01(method_root, config, anchor_marker_id, camera_poses):
        if _category(config) != "real_vehicle" or int(anchor_marker_id) != 0:
            return original(method_root, config, anchor_marker_id, camera_poses)

        transform, diagnostics, warnings = _moving_bridge_anchor(
            method_root, int(anchor_marker_id), camera_poses
        )
        if transform is not None:
            return adapters.AnchorResolution(
                transform,
                "OK_MOVING_BRIDGE_CONSENSUS",
                True,
                warnings,
                diagnostics,
            )

        # A multi-camera direct anchor is still acceptable.  What caused the
        # observed RViz failure was specifically accepting one single PnP pose as
        # the global frame for the complete AP01 rig.
        direct = original(method_root, config, anchor_marker_id, camera_poses)
        candidate_count = int(direct.diagnostics.get("candidate_count") or 0)
        if direct.available and candidate_count >= 2:
            return direct
        return adapters.AnchorResolution(
            None,
            "ANCHOR_ALIGNMENT_UNDERCONSTRAINED",
            False,
            (
                "Real-vehicle marker 0 cannot be used as AP01's global export frame from a single static PnP observation. Robust moving-trajectory support or at least two independent static-camera observations are required.",
            ),
            {
                "method": "ap01",
                "requested_anchor_marker_id": int(anchor_marker_id),
                "moving_bridge_attempt": diagnostics,
                "direct_static_attempt": direct.diagnostics,
                "ground_truth_used": False,
            },
        )

    ap01._rigcal_ap01_marker_zero_bridge = True  # type: ignore[attr-defined]
    adapters._ap01 = ap01
    _INSTALLED = True
