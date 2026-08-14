"""Deterministic AP02 pose-graph initialization."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .common import (
    AP02_ROOT,
    DEFAULT_REF_MARKER_ID,
    ensure_dir,
    read_csv,
    write_csv,
    make_T,
    T_from_detection_row,
    make_observer_known_from_marker,
    make_marker_known_from_observer,
    pose_row,
    pose_fields,
)

from camera_rig_calibration.observation_quality import (
    observation_succeeded as is_success,
    pnp_reprojection_rmse,
)


OBS_CSV = (
    AP02_ROOT
    / "02_aruco_observations"
    / "ap02_all_aruco_observations.csv"
)


Node = tuple[str, str | int]


def marker_node(marker_id: int) -> Node:
    return ("marker", int(marker_id))


def observer_node(observer_id: str) -> Node:
    return ("observer", str(observer_id))


def filter_mode(
    rows: list[dict[str, str]],
    mode: str,
) -> list[dict[str, str]]:
    if mode == "static_only":
        return [
            row
            for row in rows
            if row.get("observer_type") == "static"
        ]

    if mode == "with_moving":
        return rows

    raise RuntimeError(f"Unknown mode: {mode}")


def best_observations(
    rows: list[dict[str, str]],
    *,
    edge_weight_policy: str = "legacy_observation_quality_v1",
) -> list[dict[str, str]]:
    """Choose one deterministic initialization edge per observer/marker.

    Bundle adjustment still receives every accepted observation.  This
    de-duplication only prevents an observer/marker pair from creating parallel
    initialization edges using deterministic visible observation fields.
    """
    best: dict[
        tuple[str, int],
        dict[str, str],
    ] = {}

    if edge_weight_policy == "legacy_observation_quality_v1":
        for row in rows:
            if not is_success(row):
                continue
            try:
                marker_id = int(float(row["marker_id"]))
            except (KeyError, ValueError):
                continue
            score = edge_quality(row, edge_weight_policy)
            if score <= 0.0:
                continue
            key = (row["observer_id"], marker_id)
            if key not in best or score > edge_quality(
                best[key], edge_weight_policy
            ):
                best[key] = row
        return list(best.values())

    def rank(row: dict[str, str]) -> tuple[float, float, float, str]:
        score = edge_quality(row, edge_weight_policy)
        rmse = observation_pnp_rmse(row)
        try:
            area = float(
                row.get("marker_area_ratio")
                or row.get("area_px2", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            area = 0.0
        stable = "|".join(
            (
                str(row.get("observer_id", "")),
                str(row.get("frame_id", "")),
                str(row.get("image_path", "")),
            )
        )
        return (-score, rmse, -area, stable)

    for row in sorted(rows, key=rank):
        if not is_success(row):
            continue

        try:
            marker_id = int(float(row["marker_id"]))
        except (KeyError, ValueError):
            continue

        if not math.isfinite(observation_pnp_rmse(row)):
            continue

        key = (
            row["observer_id"],
            marker_id,
        )

        if key not in best or rank(row) < rank(best[key]):
            best[key] = row

    return [best[key] for key in sorted(best)]


def _finite_float(
    row: dict[str, str], key: str, default: float
) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def observation_pnp_rmse(row: dict[str, str]) -> float:
    """Read the preflight-audited PnP RMSE, with legacy recomputation."""

    stored = _finite_float(
        row, "pnp_reprojection_rmse_px", float("nan")
    )
    if math.isfinite(stored):
        return stored
    return pnp_reprojection_rmse(row)


def edge_quality(
    row: dict[str, str],
    policy: str = "legacy_observation_quality_v1",
) -> float:
    """Return the configured, GT-free AP02 graph-edge score."""
    if policy == "legacy_observation_quality_v1":
        return main_observation_score(row)
    if policy != "wizard_selection_score_v2":
        raise ValueError(f"Unknown AP02 graph-edge weight policy: {policy}")
    score = _finite_float(row, "selection_score", float("nan"))
    if math.isfinite(score) and score > 0.0:
        return score
    # Compatibility for prepared schema-v5 observations produced before the
    # shared selection_score column was introduced. New runs never use this
    # branch because preflight writes the audited score components.
    rmse = observation_pnp_rmse(row)
    area = max(
        0.0,
        _finite_float(
            row,
            "marker_area_ratio",
            _finite_float(row, "area_px2", 0.0),
        ),
    )
    if not math.isfinite(rmse):
        return 0.0
    return float(math.sqrt(area) / (1.0 + rmse))


def edge_metadata(row: dict[str, str]) -> dict[str, object]:
    distance = _finite_float(row, "distance_m", float("nan"))
    return {
        "selection_score": edge_quality(row),
        "pnp_reprojection_rmse_px": observation_pnp_rmse(row),
        "marker_area_ratio": _finite_float(
            row, "marker_area_ratio", 0.0
        ),
        "area_px2": _finite_float(row, "area_px2", 0.0),
        "distance_m": distance if math.isfinite(distance) else None,
        "observer_id": str(row.get("observer_id", "")),
        "marker_id": int(float(row["marker_id"])),
        "frame_id": str(row.get("frame_id", "")),
        "image_path": str(row.get("image_path", "")),
    }


def build_graph(
    rows: list[dict[str, str]],
    *,
    preserve_input_order: bool = True,
):
    adjacency = defaultdict(list)

    for row in rows:
        marker_id = int(float(row["marker_id"]))
        observer_id = row["observer_id"]
        marker = marker_node(marker_id)
        observer = observer_node(observer_id)

        adjacency[marker].append(
            (observer, row)
        )

        adjacency[observer].append(
            (marker, row)
        )

    if not preserve_input_order:
        for node in adjacency:
            adjacency[node].sort(
                key=lambda item: (
                    item[0][0],
                    str(item[0][1]),
                    str(item[1].get("frame_id", "")),
                    str(item[1].get("image_path", "")),
                )
            )
    return adjacency


def deterministic_breadth_first_tree(
    adjacency,
    start: Node,
):
    """Build a stable unweighted tree from the reference marker."""
    visited = {start}
    parent: dict[Node, tuple[Node, dict[str, str]]] = {}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor, row in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = (current, row)
            queue.append(neighbor)
    return parent


def _node_text(node: Node) -> str:
    return f"{node[0]}:{node[1]}"


def maximum_bottleneck_tree(
    adjacency,
    start: Node,
    *,
    edge_weight_policy: str = "wizard_selection_score_v2",
):
    """Choose one deterministic maximum-bottleneck path per graph node.

    The comparison order is: larger bottleneck, fewer hops, larger mean
    selection score, smaller worst PnP RMSE, larger minimum marker-area ratio,
    then a lexicographically stable full path signature.
    """

    initial = {
        "bottleneck_score": float("inf"),
        "path_length": 0,
        "score_sum": 0.0,
        "mean_score": float("inf"),
        "maximum_pnp_rmse_px": 0.0,
        "minimum_marker_area_ratio": float("inf"),
        "path_signature": (_node_text(start),),
    }

    def rank(metrics: dict[str, object]) -> tuple[object, ...]:
        return (
            -float(metrics["bottleneck_score"]),
            int(metrics["path_length"]),
            -float(metrics["mean_score"]),
            float(metrics["maximum_pnp_rmse_px"]),
            -float(metrics["minimum_marker_area_ratio"]),
            tuple(metrics["path_signature"]),
        )

    best: dict[Node, dict[str, object]] = {start: initial}
    parent: dict[Node, tuple[Node, dict[str, str]]] = {}
    heap: list[tuple[tuple[object, ...], str, Node]] = [
        (rank(initial), _node_text(start), start)
    ]

    while heap:
        queued_rank, _stable_node, current = heapq.heappop(heap)
        current_metrics = best.get(current)
        if current_metrics is None or queued_rank != rank(current_metrics):
            continue
        for neighbor, row in adjacency.get(current, []):
            quality = edge_quality(row, edge_weight_policy)
            if not math.isfinite(quality) or quality <= 0.0:
                continue
            rmse = observation_pnp_rmse(row)
            area_ratio = _finite_float(
                row, "marker_area_ratio", 0.0
            )
            hops = int(current_metrics["path_length"]) + 1
            score_sum = float(current_metrics["score_sum"]) + quality
            signature = (
                *tuple(current_metrics["path_signature"]),
                (
                    f"{_node_text(neighbor)}@"
                    f"{row.get('observer_id', '')}:"
                    f"{row.get('frame_id', '')}:"
                    f"{row.get('image_path', '')}"
                ),
            )
            candidate: dict[str, object] = {
                "bottleneck_score": min(
                    float(current_metrics["bottleneck_score"]), quality
                ),
                "path_length": hops,
                "score_sum": score_sum,
                "mean_score": score_sum / hops,
                "maximum_pnp_rmse_px": max(
                    float(current_metrics["maximum_pnp_rmse_px"]), rmse
                ),
                "minimum_marker_area_ratio": min(
                    float(current_metrics["minimum_marker_area_ratio"]),
                    area_ratio,
                ),
                "path_signature": signature,
            }
            previous = best.get(neighbor)
            if previous is not None and rank(previous) <= rank(candidate):
                continue
            best[neighbor] = candidate
            parent[neighbor] = (current, row)
            heapq.heappush(
                heap, (rank(candidate), _node_text(neighbor), neighbor)
            )
    return parent, best


def main_compat_widest_path_tree(
    adjacency,
    start: Node,
    *,
    edge_weight_policy: str = "wizard_selection_score_v2",
):
    """Reproduce the validated ``main`` maximum-frontier initialization.

    This is Prim's rooted maximum-spanning-tree construction used by the
    former ``05_initialize_ref_marker_pose_graph_v2.py`` workflow.  It is kept
    separate from :func:`maximum_bottleneck_tree`, whose richer path-level
    tie-breakers remain useful as an independent diagnostic.
    """

    visited = {start}
    parent: dict[Node, tuple[Node, dict[str, str]]] = {}
    metrics: dict[Node, dict[str, object]] = {
        start: {
            "bottleneck_score": float("inf"),
            "path_length": 0,
            "mean_score": float("inf"),
        }
    }
    frontier: list[tuple[object, ...]] = []
    counter = itertools.count()

    def push(node: Node) -> None:
        for neighbor, row in adjacency.get(node, []):
            if neighbor in visited:
                continue
            score = edge_quality(row, edge_weight_policy)
            if not math.isfinite(score) or score <= 0.0:
                continue
            heapq.heappush(
                frontier,
                (
                    -score,
                    next(counter),
                    node,
                    neighbor,
                    row,
                ),
            )

    push(start)
    while frontier:
        (
            _negative_score,
            _counter,
            source,
            target,
            row,
        ) = heapq.heappop(frontier)
        if target in visited:
            continue
        if source not in visited:
            raise RuntimeError(
                "Main-compatible frontier contains an uninitialized parent: "
                f"{source}"
            )
        score = edge_quality(row, edge_weight_policy)
        visited.add(target)
        parent[target] = (source, row)
        source_metrics = metrics[source]
        length = int(source_metrics["path_length"]) + 1
        previous_mean = float(source_metrics["mean_score"])
        score_sum = (
            0.0
            if not math.isfinite(previous_mean)
            else previous_mean * int(source_metrics["path_length"])
        )
        metrics[target] = {
            "bottleneck_score": min(
                float(source_metrics["bottleneck_score"]), score
            ),
            "path_length": length,
            "mean_score": (score_sum + score) / length,
        }
        push(target)
    return parent, metrics


def main_observation_score(row: dict[str, str]) -> float:
    """Legacy-main GT-free observation score for parity evidence only."""

    if not is_success(row):
        return 0.0
    area = _finite_float(row, "area_px2", 0.0)
    distance = _finite_float(row, "distance_m", 99.0)
    depth = _finite_float(row, "tvec_z_m", -1.0)
    rmse = observation_pnp_rmse(row)
    if (
        area < 64.0
        or distance <= 0.0
        or depth <= 0.0
        or not math.isfinite(rmse)
        or rmse > 25.0
    ):
        return 0.0
    corners: list[tuple[float, float]] = []
    for index in range(4):
        u = _finite_float(row, f"corner{index}_u", float("nan"))
        v = _finite_float(row, f"corner{index}_v", float("nan"))
        if not math.isfinite(u) or not math.isfinite(v):
            return 0.0
        corners.append((u, v))
    width = _finite_float(
        row, "image_width", 2.0 * _finite_float(row, "cx", 0.0)
    )
    height = _finite_float(
        row, "image_height", 2.0 * _finite_float(row, "cy", 0.0)
    )
    margin = max(
        0.0,
        min(
            value
            for u, v in corners
            for value in (u, width - 1.0 - u, v, height - 1.0 - v)
        ),
    )
    return float(
        math.sqrt(area)
        * (1.0 / math.sqrt(max(distance, 0.25)))
        * (1.0 / ((1.0 + rmse) ** 2))
        * min(1.0, max(0.20, margin / 40.0))
    )


