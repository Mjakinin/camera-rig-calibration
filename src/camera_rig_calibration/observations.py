from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config.models import RigConfig


def _success(row: dict[str, str]) -> bool:
    return str(row.get("pnp_success", "")).strip().lower() in {"true", "1", "yes"}


def _number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        try:
            value = row.get(key)
            if value not in {None, ""}:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _marker_id(row: dict[str, str]) -> int:
    return int(float(row["marker_id"]))


def _observer_id(row: dict[str, str]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


def _median_value(
    rows: list[dict[str, str]], *keys: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in keys:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
                break
    return float(statistics.median(values)) if values else None


@dataclass(frozen=True)
class ResolvedSelections:
    root_camera: str
    ap02_reference_marker_id: int
    ap03_single_scale_marker_id: int
    ap03_multi_marker_ids: tuple[int, ...]
    evaluation_anchor_marker_id: int | None
    marker_ids: tuple[int, ...]
    payload: dict[str, Any]

    # Schema-v1 compatibility for callers that have not migrated yet.
    @property
    def reference_marker_id(self) -> int:
        return self.ap02_reference_marker_id

    @property
    def scale_anchor_marker_id(self) -> int:
        return (
            self.evaluation_anchor_marker_id
            if self.evaluation_anchor_marker_id is not None
            else self.ap03_single_scale_marker_id
        )


ResolvedReferences = ResolvedSelections


def _read_observations(root: Path) -> list[dict[str, str]]:
    path = root / "shared_all_aruco_observations.csv"
    if not path.is_file():
        raise RuntimeError(f"Observation table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if _success(row)]
    if not successful:
        raise RuntimeError(f"No successful marker observations in {path}")
    return successful


def _best_candidate(
    candidates: dict[str | int, dict[str, Any]],
    rank,
    *,
    compatibility_key: str = "compatible",
) -> str | int:
    pool = {
        key: value
        for key, value in candidates.items()
        if bool(value.get(compatibility_key, True))
    }
    if not pool:
        raise RuntimeError("No compatible selection candidates are available")
    best_rank = max(rank(value) for value in pool.values())
    # The stable ascending ID is deliberately the final tie-breaker.
    return sorted(
        (key for key, value in pool.items() if rank(value) == best_rank),
        key=lambda value: (str(type(value)), value),
    )[0]


def _lower_is_better(value: float | None) -> float:
    return -value if value is not None else float("-inf")


def _higher_is_better(value: float | None) -> float:
    return value if value is not None else float("-inf")


def _root_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        len(candidate["reachable_cameras"]),
        len(candidate["direct_connections"]),
        len(candidate["moving_bridges"]),
        candidate["distinct_markers"],
        candidate["observations"],
        _lower_is_better(candidate["median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["median_marker_area_px2"]),
    )


def _ap02_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["static_graph_reachable_count"],
        candidate["static_camera_count"],
        candidate["moving_frames"],
        candidate["accepted_observations"],
        _lower_is_better(candidate["median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["median_marker_area_px2"]),
    )


def ap03_candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["moving_frames"],
        candidate["static_camera_count"],
        _lower_is_better(candidate["moving_median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["moving_median_marker_area_px2"]),
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
            "moving_median_pnp_reprojection_rmse_px": _median_value(
                moving,
                "pnp_reprojection_rmse_px",
                "reprojection_rmse_px",
                "reprojection_error_px",
            ),
            "moving_median_marker_area_px2": _median_value(
                moving, "area_px2"
            ),
            "ap02_compatible": len(combined_static_cameras) == len(camera_ids),
            "ap02_partial_compatible": partial_compatible,
            "ap02_static_only_partial": len(reachable) < len(camera_ids),
            "ap03_compatible": len(moving_frames) >= 2,
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
    selected = (
        int(
            _best_candidate(
                candidates,
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


def resolve_selections(
    config: RigConfig, observations_root: Path
) -> ResolvedSelections:
    rows = _read_observations(observations_root)
    camera_ids = tuple(camera.id for camera in config.static_cameras)
    declared = set(camera_ids)
    observed = {
        _observer_id(row)
        for row in rows
        if row.get("observer_type") == "static"
    }
    unknown = observed - declared
    if unknown:
        raise RuntimeError(
            f"Observations contain undeclared static camera IDs: {sorted(unknown)}"
        )

    roots = _root_candidates(rows, camera_ids)
    markers = _marker_candidates(rows, camera_ids)
    enabled = set(config.methods.enabled)
    configured_root = config.methods.ap01.root_camera
    root = (
        str(_best_candidate(roots, _root_rank))
        if configured_root == "auto"
        else configured_root
    )
    if root not in roots:
        raise RuntimeError(f"Configured AP01 root camera is not in the rig: {root}")
    if "ap01" in enabled and not roots[root]["compatible"]:
        raise RuntimeError(
            f"Configured AP01 root camera '{root}' has no successful observations"
        )

    ap02_reference = _marker_choice(
        config.methods.ap02.reference_marker_id,
        markers,
        compatibility_key="ap02_compatible",
        purpose="AP02 reference",
        rank=_ap02_rank,
        require_compatibility="ap02" in enabled,
        expected_camera_ids=camera_ids,
    )
    single_marker = _marker_choice(
        config.methods.ap03_single.scale_marker_id,
        markers,
        compatibility_key="ap03_compatible",
        purpose="AP03 Single scale",
        rank=ap03_candidate_rank,
        require_compatibility="ap03" in enabled,
    )

    compatible_multi = tuple(
        sorted(
            marker
            for marker, details in markers.items()
            if details["ap03_compatible"]
        )
    )
    configured_multi = config.methods.ap03_multi.marker_ids
    multi_markers = (
        compatible_multi
        if configured_multi == "auto"
        else tuple(sorted(dict.fromkeys(int(value) for value in configured_multi)))
    )
    if "ap03" not in enabled and not multi_markers:
        multi_markers = tuple(sorted(markers))
    if "ap03" in enabled and not multi_markers:
        raise RuntimeError("AP03 Multi has no compatible moving-camera markers")
    incompatible_multi = [
        marker
        for marker in multi_markers
        if marker not in markers or not markers[marker]["ap03_compatible"]
    ]
    if "ap03" in enabled and incompatible_multi:
        raise RuntimeError(
            f"AP03 Multi markers are not compatible: {incompatible_multi}"
        )

    moving_markers = {
        _marker_id(row) for row in rows if row.get("observer_type") == "moving"
    }
    root_markers = {
        _marker_id(row)
        for row in rows
        if row.get("observer_type") == "static" and _observer_id(row) == root
    }
    evaluation_candidates = {
        marker: details
        for marker, details in markers.items()
        if marker in moving_markers
        and (
            "ap01" not in enabled
            or marker in root_markers
        )
    }
    configured_evaluation = config.evaluation.anchor_marker_id
    evaluation_anchor: int | None
    if configured_evaluation == "auto_common":
        evaluation_anchor = None
    else:
        evaluation_anchor = int(configured_evaluation)
        if evaluation_anchor not in evaluation_candidates:
            requirement = (
                f"AP01 root '{root}' and moving-camera observations"
                if "ap01" in enabled
                else "moving-camera and static observations"
            )
            raise RuntimeError(
                f"Evaluation anchor {evaluation_anchor} must be visible in "
                f"{requirement}"
            )

    if config.selection.mode == "explicit":
        unresolved: list[str] = []
        if "ap01" in enabled and configured_root == "auto":
            unresolved.append("methods.ap01.root_camera")
        if (
            "ap02" in enabled
            and config.methods.ap02.reference_marker_id == "auto"
        ):
            unresolved.append("methods.ap02.reference_marker_id")
        if (
            "ap03" in enabled
            and config.methods.ap03_single.scale_marker_id == "auto"
        ):
            unresolved.append("methods.ap03.single.scale_marker_id")
        if (
            "ap03" in enabled
            and config.methods.ap03_multi.marker_ids == "auto"
        ):
            unresolved.append("methods.ap03.multi.marker_ids")
        if unresolved:
            raise RuntimeError(
                "selection.mode=explicit requires values for: "
                + ", ".join(unresolved)
            )

    marker_ids = tuple(sorted(markers))
    root_payload = [
        {**details, "recommended": camera == root}
        for camera, details in sorted(roots.items())
    ]
    ap02_payload = [
        {
            **details,
            "compatible": (
                details["ap02_compatible"]
                or details["ap02_partial_compatible"]
            ),
            "diagnostic_partial": (
                not details["ap02_compatible"]
                and details["ap02_partial_compatible"]
            ),
            "recommended": marker == ap02_reference,
        }
        for marker, details in sorted(markers.items())
    ]
    ap03_payload = [
        {
            **details,
            "compatible": details["ap03_compatible"],
            "recommended": marker == single_marker,
        }
        for marker, details in sorted(markers.items())
    ]
    payload: dict[str, Any] = {
        "schema_version": 5,
        "selection_mode": config.selection.mode,
        "ap01_root_camera": {
            "configured": configured_root,
            "selected": root,
            "candidates": root_payload,
            "reason": (
                "explicit user configuration"
                if configured_root != "auto"
                else "lexicographic AP01 reachability, direct links, moving bridges, observation quality and stable camera ID"
            ),
        },
        "ap02_reference_marker": {
            "configured": config.methods.ap02.reference_marker_id,
            "selected": ap02_reference,
            "candidates": ap02_payload,
            "reason": (
                "explicit user configuration"
                if config.methods.ap02.reference_marker_id != "auto"
                else (
                    "deterministic recommendation from static-only reachability, "
                    "direct static coverage, moving-frame coverage, observation "
                    "count, median PnP RMSE and median marker area"
                )
            ),
        },
        "ap03_single_scale_marker": {
            "configured": config.methods.ap03_single.scale_marker_id,
            "selected": single_marker,
            "candidates": ap03_payload,
            "reason": (
                "explicit user configuration"
                if config.methods.ap03_single.scale_marker_id != "auto"
                else (
                    "deterministic recommendation from moving-frame coverage, "
                    "direct static coverage, median moving PnP RMSE and median "
                    "moving marker area"
                )
            ),
        },
        "ap03_multi_marker_set": {
            "configured": configured_multi,
            "selected": list(multi_markers),
            "candidates": ap03_payload,
            "reason": (
                "all compatible detected moving-camera markers"
                if configured_multi == "auto"
                else "explicit user configuration"
            ),
        },
        "evaluation_anchor": {
            "configured": configured_evaluation,
            "selected": evaluation_anchor,
            "resolution_stage": (
                "after_methods"
                if configured_evaluation == "auto_common"
                else "before_methods_explicit"
            ),
            "observation_candidates": sorted(evaluation_candidates),
            "reason": (
                "resolved after method outputs from their common available markers"
                if configured_evaluation == "auto_common"
                else "explicit user configuration"
            ),
        },
        "detected_marker_ids": list(marker_ids),
    }
    observations_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    (observations_root / "SELECTION_CANDIDATES.json").write_text(
        text, encoding="utf-8"
    )
    # Compatibility alias for existing result readers.
    (observations_root / "REFERENCE_SELECTIONS.json").write_text(
        text, encoding="utf-8"
    )
    (observations_root / "REFERENCE_MARKER_ID.txt").write_text(
        f"{ap02_reference}\n", encoding="utf-8"
    )
    return ResolvedSelections(
        root,
        ap02_reference,
        single_marker,
        multi_markers,
        evaluation_anchor,
        marker_ids,
        payload,
    )


def resolve_references(
    config: RigConfig, observations_root: Path
) -> ResolvedSelections:
    """Compatibility name for schema-v1 callers."""
    return resolve_selections(config, observations_root)


def freeze_selections(
    config: RigConfig,
    resolved: ResolvedSelections,
    overrides: dict[str, Any] | None = None,
) -> RigConfig:
    """Return a prompt-free config with every pre-method decision explicit."""
    values = dict(overrides or {})
    root = str(values.get("root_camera", resolved.root_camera))
    ap02_marker = int(
        values.get(
            "ap02_reference_marker_id", resolved.ap02_reference_marker_id
        )
    )
    single_marker = int(
        values.get(
            "ap03_single_scale_marker_id",
            resolved.ap03_single_scale_marker_id,
        )
    )
    multi_value = values.get(
        "ap03_multi_marker_ids", resolved.ap03_multi_marker_ids
    )
    multi_markers = tuple(sorted(dict.fromkeys(int(item) for item in multi_value)))

    available_roots = {
        str(item["id"])
        for item in resolved.payload["ap01_root_camera"]["candidates"]
        if item.get("compatible", True)
    }
    if root not in available_roots:
        raise ValueError(f"AP01 root camera is not compatible: {root}")
    marker_details = {
        int(item["id"]): item
        for item in resolved.payload["ap03_single_scale_marker"]["candidates"]
    }
    if (
        "ap02" in config.methods.enabled
        and (
            ap02_marker not in marker_details
            or not (
                marker_details[ap02_marker].get("ap02_compatible", False)
                or marker_details[ap02_marker].get(
                    "ap02_partial_compatible", False
                )
            )
        )
    ):
        raise ValueError(f"AP02 reference marker is not compatible: {ap02_marker}")
    if (
        "ap03" in config.methods.enabled
        and single_marker not in marker_details
    ):
        raise ValueError(
            f"AP03 Single scale marker was not detected: {single_marker}"
        )
    invalid_multi = [
        marker
        for marker in multi_markers
        if marker not in marker_details
        or not marker_details[marker].get("ap03_compatible", False)
    ]
    if "ap03" in config.methods.enabled and invalid_multi:
        raise ValueError(f"AP03 Multi markers are not compatible: {invalid_multi}")

    ap03 = config.methods.ap03.model_copy(
        update={
            "single": config.methods.ap03.single.model_copy(
                update={"scale_marker_id": single_marker}
            ),
            "multi": config.methods.ap03.multi.model_copy(
                update={"marker_ids": list(multi_markers)}
            ),
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={
            "ap01": config.methods.ap01.model_copy(
                update={"root_camera": root}
            ),
            "ap02": config.methods.ap02.model_copy(
                update={"reference_marker_id": ap02_marker}
            ),
            "ap03": ap03,
        },
        deep=True,
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "selection": config.selection.model_copy(
                    update={"mode": "explicit"}
                ),
                "methods": methods,
            },
            deep=True,
        ).model_dump(mode="python")
    )
