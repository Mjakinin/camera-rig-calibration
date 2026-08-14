from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config.models import DatasetCategory, RigConfig



from .observation_core import (
    _ap02_rank,
    _best_candidate,
    _marker_id,
    _median_value,
    _observer_id,
)
def _root_candidates(
    rows: list[dict[str, str]], camera_ids: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    static_markers: dict[str, set[int]] = {camera: set() for camera in camera_ids}
    static_rows: dict[str, list[dict[str, str]]] = {
        camera: [] for camera in camera_ids
    }
    moving_markers = {
        _marker_id(row) for row in rows if row.get("observer_type") == "moving"
    }
    for row in rows:
        camera = _observer_id(row)
        if row.get("observer_type") == "static" and camera in static_markers:
            static_markers[camera].add(_marker_id(row))
            static_rows[camera].append(row)

    direct_graph: dict[str, set[str]] = {
        camera: set() for camera in camera_ids
    }
    moving_graph: dict[str, set[str]] = {
        camera: set() for camera in camera_ids
    }
    for first in camera_ids:
        for second in camera_ids:
            if first >= second:
                continue
            if static_markers[first] & static_markers[second]:
                direct_graph[first].add(second)
                direct_graph[second].add(first)
            if (
                static_markers[first] & moving_markers
                and static_markers[second] & moving_markers
            ):
                moving_graph[first].add(second)
                moving_graph[second].add(first)

    result: dict[str, dict[str, Any]] = {}
    all_cameras = set(camera_ids)
    for root in camera_ids:
        direct = direct_graph[root]
        bridges = moving_graph[root]
        reachable = {root}
        pending = [root]
        while pending:
            current = pending.pop()
            for target in direct_graph[current] | moving_graph[current]:
                if target not in reachable:
                    reachable.add(target)
                    pending.append(target)
        result[root] = {
            "id": root,
            "compatible": bool(static_markers[root]),
            "reachable_cameras": sorted(reachable),
            "unreachable_cameras": sorted(all_cameras - reachable),
            "direct_connections": sorted(direct),
            "moving_bridges": sorted(bridges),
            "distinct_markers": len(static_markers[root]),
            "observations": len(static_rows[root]),
            "median_pnp_reprojection_rmse_px": _median_value(
                static_rows[root],
                "pnp_reprojection_rmse_px",
                "reprojection_rmse_px",
                "reprojection_error_px",
            ),
            "median_marker_area_px2": _median_value(
                static_rows[root], "area_px2"
            ),
            "median_marker_area_ratio": _median_value(
                static_rows[root], "marker_area_ratio"
            ),
            "median_selection_score": _median_value(
                static_rows[root], "selection_score"
            ),
            "median_score_area": _median_value(
                static_rows[root], "score_area"
            ),
            "median_score_reprojection": _median_value(
                static_rows[root], "score_reprojection"
            ),
            "median_score_border": _median_value(
                static_rows[root], "score_border"
            ),
            "median_score_distance": _median_value(
                static_rows[root], "score_distance"
            ),
        }
    return result


def _marker_candidates(
    rows: list[dict[str, str]], camera_ids: tuple[str, ...]
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    static_marker_to_cameras: dict[int, set[str]] = defaultdict(set)
    static_camera_to_markers: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        marker = _marker_id(row)
        grouped[marker].append(row)
        if row.get("observer_type") == "static":
            camera = _observer_id(row)
            if camera in camera_ids:
                static_marker_to_cameras[marker].add(camera)
                static_camera_to_markers[camera].add(marker)

    marker_to_observers: dict[int, set[str]] = defaultdict(set)
    observer_to_markers: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        marker = _marker_id(row)
        observer = _observer_id(row)
        marker_to_observers[marker].add(observer)
        observer_to_markers[observer].add(marker)

    def combined_reachability(start_marker: int) -> tuple[set[str], set[int]]:
        reached_observers: set[str] = set()
        reached_markers = {start_marker}
        pending: list[tuple[str, str | int]] = [("marker", start_marker)]
        while pending:
            kind, value = pending.pop()
            if kind == "marker":
                for observer in marker_to_observers.get(int(value), set()):
                    if observer not in reached_observers:
                        reached_observers.add(observer)
                        pending.append(("observer", observer))
            else:
                for candidate in observer_to_markers.get(str(value), set()):
                    if candidate not in reached_markers:
                        reached_markers.add(candidate)
                        pending.append(("marker", candidate))
        return reached_observers, reached_markers

    def static_reachability(start_marker: int) -> set[str]:
        reached_cameras: set[str] = set()
        reached_markers = {start_marker}
        pending: list[tuple[str, str | int]] = [("marker", start_marker)]
        while pending:
            kind, value = pending.pop()
            if kind == "marker":
                for camera in static_marker_to_cameras.get(int(value), set()):
                    if camera not in reached_cameras:
                        reached_cameras.add(camera)
                        pending.append(("camera", camera))
            else:
                for candidate in static_camera_to_markers.get(str(value), set()):
                    if candidate not in reached_markers:
                        reached_markers.add(candidate)
                        pending.append(("marker", candidate))
        return reached_cameras

    result: dict[int, dict[str, Any]] = {}
    for marker, marker_rows in grouped.items():
        static_cameras = {
            _observer_id(row)
            for row in marker_rows
            if row.get("observer_type") == "static"
            and _observer_id(row) in camera_ids
        }
        moving = [
            row for row in marker_rows if row.get("observer_type") == "moving"
        ]
        moving_frames = {
            str(row.get("frame_id") or _observer_id(row)) for row in moving
        }
        reachable = static_reachability(marker)
        combined_observers, combined_markers = combined_reachability(marker)
        combined_static_cameras = set(camera_ids).intersection(
            combined_observers
        )
        combined_moving_observers = {
            observer
            for observer in combined_observers
            if observer not in camera_ids
        }
        partial_compatible = (
            len(combined_static_cameras) >= min(2, len(camera_ids))
            and bool(combined_moving_observers)
        )
        result[marker] = {
            "id": marker,
            "static_cameras": sorted(static_cameras),
            "static_camera_count": len(static_cameras),
            "static_graph_reachable_cameras": sorted(reachable),
            "static_graph_reachable_count": len(reachable),
            "combined_graph_reachable_static_cameras": sorted(
                combined_static_cameras
            ),
            "combined_graph_reachable_static_count": len(
                combined_static_cameras
            ),
            "combined_graph_reachable_marker_count": len(combined_markers),
            "combined_graph_reachable_marker_ids": sorted(combined_markers),
            "combined_graph_reachable_moving_frames": len(
                combined_moving_observers
            ),
            "moving_frames": len(moving_frames),
            "accepted_observations": len(marker_rows),
            "median_pnp_reprojection_rmse_px": _median_value(
                marker_rows,
                "pnp_reprojection_rmse_px",
                "reprojection_rmse_px",
                "reprojection_error_px",
            ),
            "median_marker_area_px2": _median_value(
                marker_rows, "area_px2"
            ),
            "median_marker_area_ratio": _median_value(
                marker_rows, "marker_area_ratio"
            ),
            "median_selection_score": _median_value(
                marker_rows, "selection_score"
            ),
            "moving_median_pnp_reprojection_rmse_px": _median_value(
                moving,
                "pnp_reprojection_rmse_px",
                "reprojection_rmse_px",
                "reprojection_error_px",
            ),
            "moving_median_marker_area_px2": _median_value(
                moving, "area_px2"
            ),
            "moving_median_marker_area_ratio": _median_value(
                moving, "marker_area_ratio"
            ),
            "moving_median_selection_score": _median_value(
                moving, "selection_score"
            ),
            "median_score_area": _median_value(marker_rows, "score_area"),
            "median_score_reprojection": _median_value(
                marker_rows, "score_reprojection"
            ),
            "median_score_border": _median_value(
                marker_rows, "score_border"
            ),
            "median_score_distance": _median_value(
                marker_rows, "score_distance"
            ),
            "ap02_compatible": len(combined_static_cameras) == len(camera_ids),
            "ap02_partial_compatible": partial_compatible,
            "ap02_static_only_partial": len(reachable) < len(camera_ids),
            "ap03_compatible": len(moving_frames) >= 2,
            "automatic_candidate": (
                len(marker_rows) >= 2
                and (
                    len(marker_to_observers[marker]) >= 2
                    or len(moving_frames) >= 2
                )
            ),
        }
    return result


def _marker_choice(
    configured: int | str,
    candidates: dict[int, dict[str, Any]],
    *,
    compatibility_key: str,
    purpose: str,
    rank,
    require_compatibility: bool = True,
    expected_camera_ids: tuple[str, ...] = (),
) -> int:
    compatible = [
        details
        for details in candidates.values()
        if details.get(compatibility_key, False)
    ]
    if configured == "auto" and require_compatibility and not compatible:
        if purpose == "AP02 reference" and candidates and expected_camera_ids:
            partial = {
                marker: details
                for marker, details in candidates.items()
                if details.get("ap02_partial_compatible", False)
                and details.get("automatic_candidate", False)
            }
            if partial:
                return int(
                    _best_candidate(
                        partial,
                        lambda details: (
                            int(
                                details.get(
                                    "combined_graph_reachable_static_count", 0
                                )
                            ),
                            _ap02_rank(details),
                        ),
                        compatibility_key="_all",
                    )
                )
            best = max(
                candidates.values(),
                key=lambda details: (
                    int(details.get("combined_graph_reachable_static_count", 0)),
                    _ap02_rank(details),
                ),
            )
            reached = set(
                str(value)
                for value in best.get(
                    "combined_graph_reachable_static_cameras", []
                )
            )
            missing = sorted(set(expected_camera_ids) - reached)
            raise RuntimeError(
                "AP02 combined observation graph has no usable component: its "
                f"largest reference-marker component reaches "
                f"{len(reached)}/{len(expected_camera_ids)} static cameras; "
                f"missing {', '.join(missing) or 'unknown'}."
            )
        raise RuntimeError(
            f"No compatible selection candidates are available for {purpose}"
        )
    if configured == "auto":
        auto_candidates = {
            marker: details
            for marker, details in candidates.items()
            if (
                details.get(compatibility_key, False)
                or (
                    purpose == "AP02 reference"
                    and details.get("ap02_partial_compatible", False)
                )
                or not require_compatibility
            )
            and details.get("automatic_candidate", False)
        }
        if not auto_candidates:
            raise RuntimeError(
                f"No repeat-supported automatic candidate is available for "
                f"{purpose}; singleton observations remain visible for "
                "diagnostics or explicit review."
            )
        candidates_for_choice = auto_candidates
    else:
        candidates_for_choice = candidates
    selected = (
        int(
            _best_candidate(
                candidates_for_choice,
                rank,
                compatibility_key=(
                    compatibility_key if require_compatibility else "_all"
                ),
            )
        )
        if configured == "auto"
        else int(configured)
    )
    if selected not in candidates:
        raise RuntimeError(f"Configured {purpose} marker {selected} was not detected")
    selected_is_compatible = candidates[selected][compatibility_key]
    if (
        purpose == "AP02 reference"
        and candidates[selected].get("ap02_partial_compatible", False)
    ):
        selected_is_compatible = True
    if require_compatibility and not selected_is_compatible:
        raise RuntimeError(
            f"Configured {purpose} marker {selected} is not compatible with that stage"
        )
    return selected


