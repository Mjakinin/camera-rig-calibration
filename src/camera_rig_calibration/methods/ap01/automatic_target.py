from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from ...config.models import RigConfig
from . import core
from .contracts import resolve_ap01_method_contract


def _camera_id(row: dict[str, str]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


def _finite(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        raw = row.get(key)
        if raw in {None, ""}:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _median_pair_rmse(
    rows_by_camera_marker: dict[tuple[str, int], dict[str, str]],
    root: str,
    target: str,
    marker_ids: list[int],
) -> float | None:
    values: list[float] = []
    for marker in marker_ids:
        first = rows_by_camera_marker.get((root, marker))
        second = rows_by_camera_marker.get((target, marker))
        if first is None or second is None:
            continue
        first_rmse = _finite(
            first,
            "pnp_reprojection_rmse_px",
            "reprojection_rmse_px",
            "reprojection_error_px",
        )
        second_rmse = _finite(
            second,
            "pnp_reprojection_rmse_px",
            "reprojection_rmse_px",
            "reprojection_error_px",
        )
        if first_rmse is not None and second_rmse is not None:
            values.append(max(first_rmse, second_rmse))
    return float(statistics.median(values)) if values else None


def automatic_ap01_direct_target(
    config: RigConfig,
    observations_root: Path,
    root_camera: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Resolve AP01's one Direct branch from its own GT-free baseline evidence.

    The baseline AP01 estimator has one Direct branch and uses moving-camera Relay
    for every other non-root camera.  The Direct target is therefore a derived
    pre-method selection, not an operator parameter.  This selector runs on the
    already quality-filtered observation table and deliberately reuses the exact
    baseline Direct candidate construction and quality/medoid-MAD aggregation.

    A target is Direct-eligible only when at least two quality-filtered independent
    shared markers survive without the single-candidate fallback and at least two
    of them are MAD inliers.  If no target satisfies that evidence contract, the
    caller freezes ``relay_only`` and AP01 constructs Relay candidates for every
    non-root camera.  No ground truth, SDF pose or evaluation result is read here.
    """

    path = Path(observations_root) / "shared_all_aruco_observations.csv"
    if not path.is_file():
        raise RuntimeError(
            "AP01 automatic Direct-target selection requires the filtered "
            f"observation table: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    camera_ids = tuple(camera.id for camera in config.static_cameras)
    static_rows: list[dict[str, str]] = []
    raw_by_camera_marker: dict[tuple[str, int], dict[str, str]] = {}
    for raw in rows:
        if str(raw.get("observer_type", "")).strip().lower() != "static":
            continue
        camera = _camera_id(raw)
        if camera not in camera_ids:
            continue
        row = dict(raw)
        row["camera_name"] = camera
        row.setdefault("pnp_success", "true")
        static_rows.append(row)
        try:
            marker = int(float(row["marker_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        previous = raw_by_camera_marker.get((camera, marker))
        current_score = _finite(row, "selection_score") or 0.0
        previous_score = (
            (_finite(previous, "selection_score") or 0.0)
            if previous is not None
            else float("-inf")
        )
        if previous is None or current_score > previous_score:
            raw_by_camera_marker[(camera, marker)] = row

    contract = resolve_ap01_method_contract(
        "baseline_v1",
        direct_target_camera="automatic_selection_probe",
        top_moving_per_marker=config.methods.ap01.top_moving_per_marker,
        scale_top_per_marker=config.methods.ap01.scale_top_per_marker,
        colmap_matcher=config.colmap.matcher,
        colmap_use_gpu=config.colmap.use_gpu,
        colmap_maximum_image_size=config.colmap.maximum_image_size,
        colmap_maximum_features=config.colmap.maximum_features,
        colmap_sequential_overlap=config.colmap.sequential_overlap,
        colmap_loop_detection=config.colmap.loop_detection,
        colmap_mapper_minimum_matches=config.colmap.mapper_minimum_matches,
    )
    prepared_static, _ = core.prepare_observations(
        static_rows,
        [],
        (1280, 720),
        (1280, 720),
        contract=contract,
    )
    static_best = core.best_static_by_camera_marker(prepared_static)

    candidates: list[dict[str, Any]] = []
    for target in camera_ids:
        if target == root_camera:
            continue
        direct = core.direct_candidates(root_camera, target, static_best)
        stats: dict[str, Any] | None = None
        if direct:
            _, stats = core.aggregate_baseline_direct_candidates(direct, contract)
        quality_count = int((stats or {}).get("num_quality_candidates") or 0)
        inlier_count = int((stats or {}).get("num_quality_mad_inliers") or 0)
        fallback_used = bool((stats or {}).get("quality_filter_fallback_used"))
        inlier_markers = sorted(
            {
                int(item["root_marker"])
                for item in direct
                if bool(item.get("inlier"))
            }
        )
        eligible = (
            len(direct) >= 2
            and quality_count >= 2
            and inlier_count >= 2
            and len(inlier_markers) >= 2
            and not fallback_used
        )
        rmse = _median_pair_rmse(
            raw_by_camera_marker,
            root_camera,
            target,
            inlier_markers,
        )
        quality_values = [
            float(item.get("quality") or 0.0)
            for item in direct
            if bool(item.get("inlier"))
        ]
        candidates.append(
            {
                "id": target,
                "compatible": eligible,
                "shared_marker_ids": sorted(
                    int(item["root_marker"]) for item in direct
                ),
                "independent_shared_markers": len(direct),
                "quality_filtered_markers": quality_count,
                "inlier_marker_ids": inlier_markers,
                "independent_inlier_markers": len(inlier_markers),
                "quality_filter_fallback_used": fallback_used,
                "selected_baseline_marker_id": (
                    (stats or {}).get("selected_marker_id")
                ),
                "median_pair_pnp_reprojection_rmse_px": rmse,
                "median_direct_quality": (
                    float(statistics.median(quality_values))
                    if quality_values
                    else 0.0
                ),
                "ground_truth_used": False,
            }
        )

    eligible = [item for item in candidates if item["compatible"]]
    eligible.sort(
        key=lambda item: (
            -int(item["independent_inlier_markers"]),
            -int(item["quality_filtered_markers"]),
            (
                float(item["median_pair_pnp_reprojection_rmse_px"])
                if item["median_pair_pnp_reprojection_rmse_px"] is not None
                else float("inf")
            ),
            -float(item["median_direct_quality"]),
            str(item["id"]),
        )
    )
    selected = str(eligible[0]["id"]) if eligible else None
    for item in candidates:
        item["recommended"] = item["id"] == selected
    return selected, candidates
