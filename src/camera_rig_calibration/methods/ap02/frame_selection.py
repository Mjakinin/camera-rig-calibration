"""Deterministic AP02 moving-frame selection with graph preservation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from camera_rig_calibration.ap02_graph import graph_components


class AP02FrameSelectionError(RuntimeError):
    """Raised when a configured AP02 frame cap cannot preserve the graph."""


@dataclass(frozen=True)
class AP02FrameSelection:
    selected_rows: tuple[dict[str, str], ...]
    selected_frame_ids: tuple[str, ...]
    diagnostics: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _frame_id(row: dict[str, str]) -> str:
    return str(
        row.get("observer_id")
        or row.get("frame_id")
        or row.get("image_path")
        or ""
    )


def _stable_frame_key(frame_id: str) -> tuple[int, int | str]:
    digits = "".join(character for character in frame_id if character.isdigit())
    return (0, int(digits)) if digits else (1, frame_id)


def _score(row: dict[str, str]) -> float:
    try:
        return float(row.get("selection_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _reference_component(
    rows: list[dict[str, str]],
    camera_ids: tuple[str, ...],
    reference_marker_id: int,
) -> tuple[set[str], set[int]]:
    component = next(
        (
            item
            for item in graph_components(rows, camera_ids)
            if reference_marker_id in item.marker_ids
        ),
        None,
    )
    if component is None:
        return set(), {reference_marker_id}
    return set(component.static_cameras), set(component.marker_ids)


def _ranked_frames(
    frame_rows: dict[str, list[dict[str, str]]],
    frame_ids: Iterable[str],
) -> list[str]:
    return sorted(
        set(frame_ids),
        key=lambda frame_id: (
            -max((_score(row) for row in frame_rows[frame_id]), default=0.0),
            _stable_frame_key(frame_id),
        ),
    )


def select_ap02_frames(
    rows: list[dict[str, str]],
    *,
    camera_ids: tuple[str, ...],
    reference_marker_id: int,
    reference_marker_maximum_frames: int | None,
    top_per_marker: int | None,
    top_per_marker_pair: int | None,
    maximum_total_frames: int | None,
) -> AP02FrameSelection:
    static_rows = [
        row for row in rows if row.get("observer_type") == "static"
    ]
    moving_rows = [
        row for row in rows if row.get("observer_type") == "moving"
    ]
    frame_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in moving_rows:
        frame_rows[_frame_id(row)].append(row)
    frame_markers = {
        frame_id: {
            int(float(row["marker_id"])) for row in observations
        }
        for frame_id, observations in frame_rows.items()
    }
    reasons: dict[str, set[str]] = defaultdict(set)

    reference_frames = _ranked_frames(
        frame_rows,
        (
            frame_id
            for frame_id, markers in frame_markers.items()
            if reference_marker_id in markers
        ),
    )
    if reference_marker_maximum_frames is not None:
        reference_frames = reference_frames[
            :reference_marker_maximum_frames
        ]
    for frame_id in reference_frames:
        reasons[frame_id].add("reference_marker")

    marker_frames: dict[int, list[str]] = defaultdict(list)
    pair_frames: dict[tuple[int, int], list[str]] = defaultdict(list)
    for frame_id, markers in frame_markers.items():
        for marker_id in markers:
            marker_frames[marker_id].append(frame_id)
        for pair in combinations(sorted(markers), 2):
            pair_frames[pair].append(frame_id)
    for marker_id, frame_ids in sorted(marker_frames.items()):
        ranked = _ranked_frames(frame_rows, frame_ids)
        if top_per_marker is not None:
            ranked = ranked[:top_per_marker]
        for frame_id in ranked:
            reasons[frame_id].add(f"top_marker_{marker_id}")
    for pair, frame_ids in sorted(pair_frames.items()):
        ranked = _ranked_frames(frame_rows, frame_ids)
        if top_per_marker_pair is not None:
            ranked = ranked[:top_per_marker_pair]
        for frame_id in ranked:
            reasons[frame_id].add(
                f"top_marker_pair_{pair[0]}_{pair[1]}"
            )

    target_cameras, target_markers = _reference_component(
        rows, camera_ids, reference_marker_id
    )
    mandatory: list[str] = []
    current_rows = list(static_rows)
    reached_cameras, reached_markers = _reference_component(
        current_rows, camera_ids, reference_marker_id
    )
    remaining = set(frame_rows)
    while (
        reached_cameras != target_cameras
        or reached_markers != target_markers
    ):
        ranked_gains: list[
            tuple[
                int,
                int,
                float,
                tuple[int, int | str],
                str,
                set[str],
                set[int],
            ]
        ] = []
        for frame_id in sorted(remaining, key=_stable_frame_key):
            candidate_rows = current_rows + frame_rows[frame_id]
            candidate_cameras, candidate_markers = _reference_component(
                candidate_rows, camera_ids, reference_marker_id
            )
            camera_gain = len(candidate_cameras - reached_cameras)
            marker_gain = len(candidate_markers - reached_markers)
            ranked_gains.append(
                (
                    -camera_gain,
                    -marker_gain,
                    -max(
                        (_score(row) for row in frame_rows[frame_id]),
                        default=0.0,
                    ),
                    _stable_frame_key(frame_id),
                    frame_id,
                    candidate_cameras,
                    candidate_markers,
                )
            )
        ranked_gains.sort(key=lambda value: value[:4])
        best = ranked_gains[0] if ranked_gains else None
        if (
            best is None
            or (best[0] == 0 and best[1] == 0)
        ):
            raise AP02FrameSelectionError(
                "AP02 frame selection could not reproduce the accepted "
                "reference component; observation graph evidence is "
                "internally inconsistent."
            )
        best_frame = best[4]
        mandatory.append(best_frame)
        reasons[best_frame].add("graph_preservation")
        current_rows.extend(frame_rows[best_frame])
        remaining.remove(best_frame)
        reached_cameras, reached_markers = best[5], best[6]

    minimum_graph_frames = len(mandatory)
    if (
        maximum_total_frames is not None
        and maximum_total_frames < minimum_graph_frames
    ):
        raise AP02FrameSelectionError(
            "AP02 maximum_total_frames="
            f"{maximum_total_frames} is smaller than the minimum "
            f"graph-preserving set ({minimum_graph_frames} moving frames)."
        )

    selected = list(mandatory)
    selected_set = set(selected)
    candidate_union = set(reasons)
    remaining_ranked = sorted(
        candidate_union - selected_set,
        key=lambda frame_id: (
            -int(
                any(
                    reason.startswith("top_marker_pair_")
                    for reason in reasons[frame_id]
                )
            ),
            -int("reference_marker" in reasons[frame_id]),
            -len(reasons[frame_id]),
            -max((_score(row) for row in frame_rows[frame_id]), default=0.0),
            _stable_frame_key(frame_id),
        ),
    )
    for frame_id in remaining_ranked:
        if (
            maximum_total_frames is not None
            and len(selected) >= maximum_total_frames
        ):
            break
        selected.append(frame_id)
        selected_set.add(frame_id)

    diagnostics: list[dict[str, Any]] = []
    for frame_id in sorted(frame_rows, key=_stable_frame_key):
        observations = frame_rows[frame_id]
        best_row = max(observations, key=_score)
        diagnostics.append(
            {
                "frame_id": frame_id,
                "selected": frame_id in selected_set,
                "selection_reasons": sorted(reasons.get(frame_id, set())),
                "marker_ids": sorted(frame_markers[frame_id]),
                "marker_count": len(frame_markers[frame_id]),
                "selection_score": _score(best_row),
                "score_area": best_row.get("score_area", ""),
                "score_reprojection": best_row.get(
                    "score_reprojection", ""
                ),
                "score_border": best_row.get("score_border", ""),
                "score_distance": best_row.get("score_distance", ""),
                "tie_breaker": "stable ascending frame ID",
            }
        )
    selected_rows = [
        *static_rows,
        *(
            row
            for frame_id in selected
            for row in frame_rows[frame_id]
        ),
    ]
    summary = {
        "schema_version": 5,
        "reference_marker_id": reference_marker_id,
        "input_moving_frames": len(frame_rows),
        "reference_marker_candidate_frames": len(reference_frames),
        "deduplicated_candidate_frames": len(candidate_union),
        "minimum_graph_preserving_frames": minimum_graph_frames,
        "selected_moving_frames": len(selected),
        "selected_frame_ids": selected,
        "target_static_cameras": sorted(target_cameras),
        "target_markers": sorted(target_markers),
        "limits": {
            "reference_marker_maximum_frames": (
                reference_marker_maximum_frames
            ),
            "top_per_marker": top_per_marker,
            "top_per_marker_pair": top_per_marker_pair,
            "maximum_total_frames": maximum_total_frames,
        },
    }
    return AP02FrameSelection(
        selected_rows=tuple(selected_rows),
        selected_frame_ids=tuple(selected),
        diagnostics=tuple(diagnostics),
        summary=summary,
    )


def write_ap02_frame_selection(
    selection: AP02FrameSelection,
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "ap02_frame_selection.csv"
    rows = list(selection.diagnostics)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["frame_id", "selected"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "selection_reasons": ",".join(
                        row["selection_reasons"]
                    ),
                    "marker_ids": ",".join(
                        str(value) for value in row["marker_ids"]
                    ),
                }
            )
    json_path = output_directory / "ap02_frame_selection.json"
    json_path.write_text(
        json.dumps(
            {
                **selection.summary,
                "frames": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
