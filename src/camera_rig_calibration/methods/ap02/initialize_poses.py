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



from .initialize_graph import (
    _node_text,
    edge_metadata,
    edge_quality,
    marker_node,
    observation_pnp_rmse,
    observer_node,
)
def initialize_from_tree(
    parent,
    ref_marker_id: int,
    *,
    path_metrics: dict[Node, dict[str, object]] | None = None,
    algorithm: str = "unweighted_first_hit_bfs",
    edge_weight_policy: str = "legacy_observation_quality_v1",
):
    start = marker_node(ref_marker_id)

    marker_poses = {
        int(ref_marker_id): make_T(
            np.eye(3),
            np.zeros(3),
        )
    }

    observer_poses = {}

    children = defaultdict(list)

    for child, (parent_node, row) in parent.items():
        children[parent_node].append(
            (
                child,
                row,
            )
        )

    queue = deque([start])

    init_log = []
    used_edges = []

    while queue:
        current = queue.popleft()

        for child, row in children.get(
            current,
            [],
        ):
            current_kind, current_id = current
            child_kind, child_id = child

            T_observer_marker = T_from_detection_row(row)

            if T_observer_marker is None:
                continue

            if (
                current_kind == "marker"
                and child_kind == "observer"
            ):
                T_ref_marker = marker_poses[int(current_id)]

                T_ref_observer = (
                    make_observer_known_from_marker(
                        T_ref_marker,
                        T_observer_marker,
                    )
                )

                observer_poses[str(child_id)] = T_ref_observer

            elif (
                current_kind == "observer"
                and child_kind == "marker"
            ):
                T_ref_observer = observer_poses[str(current_id)]

                T_ref_marker = (
                    make_marker_known_from_observer(
                        T_ref_observer,
                        T_observer_marker,
                    )
                )

                marker_poses[int(child_id)] = T_ref_marker

            else:
                raise RuntimeError(
                    f"Invalid bipartite edge: "
                    f"{current} -> {child}"
                )

            init_log.append(
                {
                    "initialization_algorithm": algorithm,
                    "initialized_type": child_kind,
                    "initialized_id": child_id,
                    "from_type": current_kind,
                    "from_id": current_id,
                    "observed_marker_id": int(
                        float(row["marker_id"])
                    ),
                    "observer_id": row["observer_id"],
                    "frame_id": row.get("frame_id", ""),
                    "edge_quality": edge_quality(row, edge_weight_policy),
                    "path_length": (
                        path_metrics.get(child, {}).get("path_length", "")
                        if path_metrics is not None
                        else ""
                    ),
                    "path_bottleneck_score": (
                        path_metrics.get(child, {}).get(
                            "bottleneck_score", ""
                        )
                        if path_metrics is not None
                        else ""
                    ),
                    "pnp_reprojection_rmse_px": (
                        observation_pnp_rmse(row)
                    ),
                    "area_px2": row.get("area_px2", ""),
                    "distance_m": row.get("distance_m", ""),
                }
            )

            used_edges.append(row)
            queue.append(child)

    return (
        marker_poses,
        observer_poses,
        init_log,
        used_edges,
    )


def _path_to_node(
    parent: dict[Node, tuple[Node, dict[str, str]]],
    node: Node,
) -> list[tuple[Node, Node, dict[str, str]]]:
    reversed_edges: list[tuple[Node, Node, dict[str, str]]] = []
    current = node
    seen: set[Node] = set()
    while current in parent:
        if current in seen:
            raise RuntimeError(
                f"Initialization parent graph contains a cycle at {current}"
            )
        seen.add(current)
        previous, row = parent[current]
        reversed_edges.append((previous, current, row))
        current = previous
    return list(reversed(reversed_edges))


def _tree_path_metrics(
    parent: dict[Node, tuple[Node, dict[str, str]]],
    start: Node,
) -> dict[Node, dict[str, object]]:
    metrics: dict[Node, dict[str, object]] = {
        start: {
            "bottleneck_score": float("inf"),
            "path_length": 0,
            "mean_score": float("inf"),
        }
    }
    for node in parent:
        edges = _path_to_node(parent, node)
        scores = [edge_quality(row) for _source, _target, row in edges]
        metrics[node] = {
            "bottleneck_score": min(scores) if scores else float("inf"),
            "path_length": len(edges),
            "mean_score": (
                float(sum(scores) / len(scores))
                if scores
                else float("inf")
            ),
        }
    return metrics


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = max(
        -1.0, min(1.0, float((np.trace(relative) - 1.0) / 2.0))
    )
    return math.degrees(math.acos(cosine))


def _camera_path_diagnostics(
    parent: dict[Node, tuple[Node, dict[str, str]]],
    metrics: dict[Node, dict[str, object]],
    poses: dict[str, np.ndarray],
    *,
    algorithm: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for camera_id, transform in sorted(poses.items()):
        node = observer_node(camera_id)
        edges = _path_to_node(parent, node)
        weakest = min(
            edges,
            key=lambda item: (
                edge_quality(item[2]),
                _node_text(item[0]),
                _node_text(item[1]),
            ),
            default=None,
        )
        path_metrics = metrics.get(node, {})
        rows.append(
            {
                "camera_id": camera_id,
                "initialization_algorithm": algorithm,
                "selected_graph_path": [
                    {
                        "from": _node_text(source),
                        "to": _node_text(target),
                        "observation": edge_metadata(row),
                    }
                    for source, target, row in edges
                ],
                "path_length": len(edges),
                "path_bottleneck_score": path_metrics.get(
                    "bottleneck_score"
                ),
                "path_mean_score": path_metrics.get("mean_score"),
                "weakest_edge": (
                    {
                        "from": _node_text(weakest[0]),
                        "to": _node_text(weakest[1]),
                        "observation": edge_metadata(weakest[2]),
                    }
                    if weakest is not None
                    else None
                ),
                "initial_pose": transform.tolist(),
            }
        )
    return rows


