from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable


def _marker_id(row: dict[str, Any]) -> int:
    return int(float(row["marker_id"]))


def _observer_id(row: dict[str, Any]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


@dataclass(frozen=True)
class AP02GraphComponent:
    component_id: str
    static_cameras: tuple[str, ...]
    marker_ids: tuple[int, ...]
    moving_frames: tuple[str, ...]
    connecting_moving_frames: tuple[str, ...]
    observation_count: int
    anchor_marker_id: int

    @property
    def calibratable(self) -> bool:
        return len(self.static_cameras) >= 2 and bool(self.moving_frames)

    def model_dump(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "static_cameras": list(self.static_cameras),
            "static_camera_count": len(self.static_cameras),
            "marker_ids": list(self.marker_ids),
            "marker_count": len(self.marker_ids),
            "moving_frames": list(self.moving_frames),
            "moving_frame_count": len(self.moving_frames),
            "connecting_moving_frames": list(
                self.connecting_moving_frames
            ),
            "connecting_moving_frame_count": len(
                self.connecting_moving_frames
            ),
            "observation_count": self.observation_count,
            "anchor_marker_id": self.anchor_marker_id,
            "calibratable": self.calibratable,
        }


@dataclass(frozen=True)
class AP02GraphDiagnosis:
    expected_static_cameras: tuple[str, ...]
    components: tuple[AP02GraphComponent, ...]
    reference_marker_id: int
    reference_component_id: str | None
    reached_static_cameras: tuple[str, ...]
    missing_static_cameras: tuple[str, ...]
    raw_component_count: int
    raw_reached_static_cameras: tuple[str, ...]
    cause_codes: tuple[str, ...]
    rejected_bridge_reasons: tuple[tuple[str, int], ...]
    explanation: str

    @property
    def complete(self) -> bool:
        return not self.missing_static_cameras

    @property
    def calibratable_components(self) -> tuple[AP02GraphComponent, ...]:
        return tuple(item for item in self.components if item.calibratable)

    def model_dump(self) -> dict[str, Any]:
        missing_links = [
            {
                "component_a": first.component_id,
                "component_b": second.component_id,
                "component_a_cameras": list(first.static_cameras),
                "component_b_cameras": list(second.static_cameras),
                "component_a_markers": list(first.marker_ids),
                "component_b_markers": list(second.marker_ids),
                "detected_cross_group_co_observations": 0,
            }
            for first, second in combinations(self.components, 2)
        ]
        return {
            "schema_version": 5,
            "expected_static_cameras": list(self.expected_static_cameras),
            "expected_static_camera_count": len(self.expected_static_cameras),
            "components": [item.model_dump() for item in self.components],
            "component_count": len(self.components),
            "calibratable_component_count": len(
                self.calibratable_components
            ),
            "reference_marker_id": self.reference_marker_id,
            "reference_component_id": self.reference_component_id,
            "reached_static_cameras": list(self.reached_static_cameras),
            "reached_static_camera_count": len(self.reached_static_cameras),
            "missing_static_cameras": list(self.missing_static_cameras),
            "raw_component_count": self.raw_component_count,
            "raw_reached_static_cameras": list(
                self.raw_reached_static_cameras
            ),
            "cause_codes": list(self.cause_codes),
            "rejected_bridge_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.rejected_bridge_reasons
            ],
            "missing_cross_group_co_observations": missing_links,
            "explanation": self.explanation,
            "complete": self.complete,
        }


def graph_components(
    rows: Iterable[dict[str, Any]],
    static_camera_ids: Iterable[str],
) -> tuple[AP02GraphComponent, ...]:
    """Return deterministic observer-marker components for AP02 combined BA."""
    camera_ids = set(static_camera_ids)
    row_list: list[dict[str, Any]] = []
    adjacency: dict[tuple[str, str | int], set[tuple[str, str | int]]] = (
        defaultdict(set)
    )
    markers_by_observer: dict[str, set[int]] = defaultdict(set)
    marker_static_support: dict[int, set[str]] = defaultdict(set)
    marker_moving_support: dict[int, set[str]] = defaultdict(set)
    marker_observation_count: Counter[int] = Counter()
    for row in rows:
        observer = _observer_id(row)
        if not observer:
            continue
        try:
            marker = _marker_id(row)
        except (KeyError, TypeError, ValueError):
            continue
        observer_node = ("observer", observer)
        marker_node = ("marker", marker)
        adjacency[observer_node].add(marker_node)
        adjacency[marker_node].add(observer_node)
        markers_by_observer[observer].add(marker)
        marker_observation_count[marker] += 1
        if observer in camera_ids:
            marker_static_support[marker].add(observer)
        else:
            marker_moving_support[marker].add(observer)
        row_list.append(row)

    raw_components: list[dict[str, Any]] = []
    seen: set[tuple[str, str | int]] = set()
    for start in sorted(adjacency, key=lambda item: (item[0], str(item[1]))):
        if start in seen:
            continue
        seen.add(start)
        pending: deque[tuple[str, str | int]] = deque([start])
        nodes: set[tuple[str, str | int]] = set()
        while pending:
            node = pending.popleft()
            nodes.add(node)
            for neighbor in sorted(
                adjacency[node], key=lambda item: (item[0], str(item[1]))
            ):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        markers = tuple(
            sorted(int(value) for kind, value in nodes if kind == "marker")
        )
        observers = {
            str(value) for kind, value in nodes if kind == "observer"
        }
        cameras = tuple(sorted(camera_ids.intersection(observers)))
        moving = tuple(sorted(observers - camera_ids))
        connecting_moving = tuple(
            observer
            for observer in moving
            if len(markers_by_observer[observer]) >= 2
        )
        marker_set = set(markers)
        component_rows = sum(
            1
            for row in row_list
            if _observer_id(row) in observers
            and _marker_id(row) in marker_set
        )
        anchor = min(
            markers,
            key=lambda marker: (
                -len(marker_static_support[marker]),
                -len(marker_moving_support[marker]),
                -marker_observation_count[marker],
                marker,
            ),
        )
        raw_components.append(
            {
                "static_cameras": cameras,
                "marker_ids": markers,
                "moving_frames": moving,
                "connecting_moving_frames": connecting_moving,
                "observation_count": component_rows,
                "anchor_marker_id": anchor,
            }
        )

    raw_components.sort(
        key=lambda item: (
            -len(item["static_cameras"]),
            -len(item["moving_frames"]),
            -len(item["marker_ids"]),
            item["marker_ids"],
            item["static_cameras"],
        )
    )
    return tuple(
        AP02GraphComponent(
            component_id=f"component_{index:02d}",
            static_cameras=item["static_cameras"],
            marker_ids=item["marker_ids"],
            moving_frames=item["moving_frames"],
            connecting_moving_frames=item["connecting_moving_frames"],
            observation_count=item["observation_count"],
            anchor_marker_id=item["anchor_marker_id"],
        )
        for index, item in enumerate(raw_components, 1)
    )


def _reference_component(
    components: Iterable[AP02GraphComponent], reference_marker_id: int
) -> AP02GraphComponent | None:
    return next(
        (
            component
            for component in components
            if reference_marker_id in component.marker_ids
        ),
        None,
    )


def _bridge_reasons(
    rejected_rows: Iterable[dict[str, Any]],
    *,
    accepted_reference: AP02GraphComponent | None,
    raw_reference: AP02GraphComponent | None,
) -> tuple[tuple[str, int], ...]:
    """Count only rejected observations lying on the raw reference component.

    This avoids blaming unrelated rejected rows.  A row is relevant when both
    endpoints belong to the raw component that connects the complete rig and
    at least one endpoint is absent from the accepted reference component.
    """
    if accepted_reference is None or raw_reference is None:
        return ()
    accepted_observers = set(accepted_reference.static_cameras) | set(
        accepted_reference.moving_frames
    )
    accepted_markers = set(accepted_reference.marker_ids)
    raw_observers = set(raw_reference.static_cameras) | set(
        raw_reference.moving_frames
    )
    raw_markers = set(raw_reference.marker_ids)
    reasons: Counter[str] = Counter()
    for row in rejected_rows:
        observer = _observer_id(row)
        try:
            marker = _marker_id(row)
        except (KeyError, TypeError, ValueError):
            continue
        if observer not in raw_observers or marker not in raw_markers:
            continue
        if observer in accepted_observers and marker in accepted_markers:
            continue
        reasons[
            str(row.get("reason") or "unknown_quality_rejection")
        ] += 1
    return tuple(sorted(reasons.items()))


def diagnose_ap02_graph(
    *,
    raw_rows: Iterable[dict[str, Any]],
    accepted_rows: Iterable[dict[str, Any]],
    rejected_rows: Iterable[dict[str, Any]],
    static_camera_ids: Iterable[str],
    reference_marker_id: int,
) -> AP02GraphDiagnosis:
    expected = tuple(sorted(set(static_camera_ids)))
    raw_row_list = [dict(row) for row in raw_rows]
    accepted_row_list = [dict(row) for row in accepted_rows]
    rejected_row_list = [dict(row) for row in rejected_rows]
    accepted_components = graph_components(accepted_row_list, expected)
    raw_components = graph_components(raw_row_list, expected)
    accepted_reference = _reference_component(
        accepted_components, reference_marker_id
    )
    raw_reference = _reference_component(raw_components, reference_marker_id)
    reached = (
        accepted_reference.static_cameras
        if accepted_reference is not None
        else ()
    )
    raw_reached = (
        raw_reference.static_cameras if raw_reference is not None else ()
    )
    missing = tuple(sorted(set(expected) - set(reached)))
    raw_observers = {
        _observer_id(row)
        for row in raw_row_list
        if _observer_id(row)
    }
    missing_raw = tuple(sorted(set(expected) - raw_observers))
    causes: list[str] = []
    if missing_raw:
        causes.append("required_camera_without_detection")
    if missing and len(raw_reached) == len(expected):
        causes.append("quality_filters_removed_bridges")
    elif missing:
        causes.append("no_detected_cross_group_observations")
    if not causes:
        causes.append("complete")

    reason_counts = _bridge_reasons(
        rejected_row_list,
        accepted_reference=accepted_reference,
        raw_reference=raw_reference,
    )
    explanation_parts: list[str] = []
    if "required_camera_without_detection" in causes:
        explanation_parts.append(
            "Required cameras without any marker detection: "
            + ", ".join(missing_raw)
            + "."
        )
    if "quality_filters_removed_bridges" in causes:
        reason_text = ", ".join(
            f"{reason} ({count})" for reason, count in reason_counts
        ) or "unknown rejection reasons"
        explanation_parts.append(
            "The unfiltered detections connect the full rig, but the configured "
            f"quality filters remove connecting observations: {reason_text}."
        )
    if "no_detected_cross_group_observations" in causes:
        explanation_parts.append(
            "No detected observation connects the listed marker/camera groups. "
            "Possible causes are missing transition frames or markers that were "
            "visible but not detected; the graph alone cannot distinguish them."
        )
    if not explanation_parts:
        explanation_parts.append(
            "All expected static cameras share one AP02 combined component."
        )
    explanation = " ".join(explanation_parts)

    return AP02GraphDiagnosis(
        expected_static_cameras=expected,
        components=accepted_components,
        reference_marker_id=reference_marker_id,
        reference_component_id=(
            accepted_reference.component_id
            if accepted_reference is not None
            else None
        ),
        reached_static_cameras=reached,
        missing_static_cameras=missing,
        raw_component_count=len(raw_components),
        raw_reached_static_cameras=raw_reached,
        cause_codes=tuple(causes),
        rejected_bridge_reasons=reason_counts,
        explanation=explanation,
    )


def rows_for_component(
    rows: Iterable[dict[str, Any]], component: AP02GraphComponent
) -> list[dict[str, Any]]:
    observers = set(component.static_cameras) | set(component.moving_frames)
    markers = set(component.marker_ids)
    return [
        dict(row)
        for row in rows
        if _observer_id(row) in observers
        and _marker_id(row) in markers
    ]
