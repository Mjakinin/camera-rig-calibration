"""Deterministic AP02 pose-graph initialization."""

from __future__ import annotations

import argparse
import heapq
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

    def rank(row: dict[str, str]) -> tuple[float, float, float, str]:
        score = edge_quality(row)
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


def edge_quality(row: dict[str, str]) -> float:
    """Return the accepted, GT-free observation selection score."""
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


def maximum_bottleneck_tree(adjacency, start: Node):
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
            quality = edge_quality(row)
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


def initialize_from_tree(
    parent,
    ref_marker_id: int,
    *,
    path_metrics: dict[Node, dict[str, object]] | None = None,
    algorithm: str = "unweighted_first_hit_bfs",
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
                    "edge_quality": edge_quality(row),
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "static_only",
            "with_moving",
        ],
        required=True,
    )

    parser.add_argument(
        "--ref-marker-id",
        type=int,
        default=DEFAULT_REF_MARKER_ID,
    )

    parser.add_argument(
        "--out-root",
        default=str(
            AP02_ROOT
            / "05_graph_initialization"
        ),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=OBS_CSV,
    )

    args = parser.parse_args()

    out = ensure_dir(
        Path(args.out_root) / args.mode
    )

    all_rows = read_csv(args.observations)
    mode_rows = filter_mode(all_rows, args.mode)
    selected_rows = best_observations(mode_rows)

    adjacency = build_graph(selected_rows)

    start = marker_node(args.ref_marker_id)

    bfs_parent = deterministic_breadth_first_tree(
        adjacency,
        start,
    )
    parent, path_metrics = maximum_bottleneck_tree(adjacency, start)

    (
        marker_poses,
        observer_poses,
        init_log,
        used_edges,
    ) = initialize_from_tree(
        parent,
        args.ref_marker_id,
        path_metrics=path_metrics,
        algorithm="maximum_bottleneck",
    )
    (
        _bfs_marker_poses,
        bfs_observer_poses,
        bfs_init_log,
        _bfs_used_edges,
    ) = initialize_from_tree(
        bfs_parent,
        args.ref_marker_id,
        path_metrics=_tree_path_metrics(bfs_parent, start),
        algorithm="unweighted_first_hit_bfs",
    )

    static_pose_rows = []
    moving_pose_rows = []
    static_observer_ids = {
        row["observer_id"]
        for row in selected_rows
        if row.get("observer_type") == "static"
    }

    for observer_id, transform in sorted(
        observer_poses.items()
    ):
        if observer_id in static_observer_ids:
            static_pose_rows.append(
                pose_row(
                    "static_camera",
                    observer_id,
                    transform,
                    source=f"{args.mode}_maximum_bottleneck",
                )
            )

        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(
                pose_row(
                    "moving_frame",
                    observer_id,
                    transform,
                    source=f"{args.mode}_maximum_bottleneck",
                )
            )

    marker_pose_rows = [
        pose_row(
            "marker",
            str(marker_id),
            transform,
            source=f"{args.mode}_maximum_bottleneck",
        )
        for marker_id, transform
        in sorted(marker_poses.items())
    ]

    write_csv(
        out / "initial_static_camera_poses_ref_marker.csv",
        static_pose_rows,
        pose_fields(),
    )

    write_csv(
        out / "initial_moving_frame_poses_ref_marker.csv",
        moving_pose_rows,
        pose_fields(),
    )

    write_csv(
        out / "initial_marker_poses_ref_marker.csv",
        marker_pose_rows,
        pose_fields(),
    )

    init_fields = [
        "initialization_algorithm",
        "initialized_type",
        "initialized_id",
        "from_type",
        "from_id",
        "observed_marker_id",
        "observer_id",
        "frame_id",
        "edge_quality",
        "path_length",
        "path_bottleneck_score",
        "pnp_reprojection_rmse_px",
        "area_px2",
        "distance_m",
    ]

    write_csv(
        out / "initialization_log.csv",
        init_log,
        init_fields,
    )

    if used_edges:
        write_csv(
            out / "used_initialization_edges.csv",
            used_edges,
            list(used_edges[0].keys()),
        )
    else:
        write_csv(
            out / "used_initialization_edges.csv",
            [],
            [
                "observer_type",
                "observer_id",
                "frame_id",
                "marker_id",
            ],
        )

    productive_static_poses = {
        camera_id: transform
        for camera_id, transform in observer_poses.items()
        if camera_id in static_observer_ids
    }
    bfs_static_poses = {
        camera_id: transform
        for camera_id, transform in bfs_observer_poses.items()
        if camera_id in static_observer_ids
    }
    productive_paths = _camera_path_diagnostics(
        parent,
        path_metrics,
        productive_static_poses,
        algorithm="maximum_bottleneck",
    )
    bfs_metrics = _tree_path_metrics(bfs_parent, start)
    bfs_paths = _camera_path_diagnostics(
        bfs_parent,
        bfs_metrics,
        bfs_static_poses,
        algorithm="unweighted_first_hit_bfs",
    )
    productive_by_camera = {
        str(row["camera_id"]): row for row in productive_paths
    }
    bfs_by_camera = {str(row["camera_id"]): row for row in bfs_paths}
    comparison_rows: list[dict[str, object]] = []
    for camera_id in sorted(
        set(productive_static_poses) | set(bfs_static_poses)
    ):
        productive_pose = productive_static_poses.get(camera_id)
        bfs_pose = bfs_static_poses.get(camera_id)
        productive_path = productive_by_camera.get(camera_id, {})
        bfs_path = bfs_by_camera.get(camera_id, {})
        comparison_rows.append(
            {
                "camera_id": camera_id,
                "productive_algorithm": "maximum_bottleneck",
                "productive_path": " -> ".join(
                    str(edge["to"])
                    for edge in productive_path.get(
                        "selected_graph_path", []
                    )
                ),
                "productive_bottleneck_score": productive_path.get(
                    "path_bottleneck_score"
                ),
                "diagnostic_algorithm": "unweighted_first_hit_bfs",
                "diagnostic_path": " -> ".join(
                    str(edge["to"])
                    for edge in bfs_path.get("selected_graph_path", [])
                ),
                "diagnostic_bottleneck_score": bfs_path.get(
                    "path_bottleneck_score"
                ),
                "translation_difference_m": (
                    float(
                        np.linalg.norm(
                            productive_pose[:3, 3] - bfs_pose[:3, 3]
                        )
                    )
                    if productive_pose is not None and bfs_pose is not None
                    else ""
                ),
                "rotation_difference_deg": (
                    _rotation_difference_deg(productive_pose, bfs_pose)
                    if productive_pose is not None and bfs_pose is not None
                    else ""
                ),
            }
        )
    write_csv(
        out / "initialization_algorithm_comparison.csv",
        comparison_rows,
        list(comparison_rows[0])
        if comparison_rows
        else [
            "camera_id",
            "productive_algorithm",
            "diagnostic_algorithm",
        ],
    )
    diagnostics_payload = {
        "schema_version": 5,
        "mode": args.mode,
        "reference_marker_id": args.ref_marker_id,
        "productive_algorithm": "maximum_bottleneck",
        "diagnostic_algorithm": "unweighted_first_hit_bfs",
        "path_tie_breakers": [
            "maximum bottleneck selection score",
            "minimum path length",
            "maximum mean selection score",
            "minimum worst PnP reprojection RMSE",
            "maximum minimum marker area ratio",
            "lexicographic node/frame/image path",
        ],
        "productive_camera_paths": productive_paths,
        "diagnostic_camera_paths": bfs_paths,
        "comparison": comparison_rows,
    }
    (out / "initialization_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    bfs_root = ensure_dir(
        out / "diagnostics" / "unweighted_first_hit_bfs"
    )
    write_csv(
        bfs_root / "initialization_log.csv",
        bfs_init_log,
        init_fields,
    )
    (bfs_root / "camera_paths.json").write_text(
        json.dumps(bfs_paths, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    successful_rows = [
        row
        for row in mode_rows
        if is_success(row)
    ]

    all_static = sorted({
        row["observer_id"]
        for row in successful_rows
        if row.get("observer_type") == "static"
    })

    all_moving = sorted({
        row["observer_id"]
        for row in successful_rows
        if row.get("observer_type") == "moving"
    })

    all_markers = sorted({
        int(float(row["marker_id"]))
        for row in successful_rows
    })

    initialized_static = sorted(
        row["entity_id"]
        for row in static_pose_rows
    )

    initialized_moving = sorted(
        row["entity_id"]
        for row in moving_pose_rows
    )

    initialized_markers = sorted(
        int(row["entity_id"])
        for row in marker_pose_rows
    )

    missing_static = sorted(
        set(all_static) - set(initialized_static)
    )

    missing_moving = sorted(
        set(all_moving) - set(initialized_moving)
    )

    missing_markers = sorted(
        set(all_markers) - set(initialized_markers)
    )

    report = [
        "AP02 maximum-bottleneck pose graph initialization",
        "=================================================",
        "",
        f"Mode: {args.mode}",
        f"Reference marker id: {args.ref_marker_id}",
        "Productive algorithm: maximum_bottleneck",
        "Diagnostic algorithm: unweighted_first_hit_bfs",
        "",
        f"Raw observations in mode: {len(mode_rows)}",
        f"Observer-marker initialization edges: {len(selected_rows)}",
        f"Used initialization edges: {len(used_edges)}",
        "",
        (
            "Initialized static cameras: "
            f"{len(initialized_static)} / {len(all_static)}"
        ),
        (
            "Initialized moving frames: "
            f"{len(initialized_moving)} / {len(all_moving)}"
        ),
        (
            "Initialized markers: "
            f"{len(initialized_markers)} / {len(all_markers)}"
        ),
        "",
        f"Initialized static cameras: {initialized_static}",
        f"Missing static cameras: {missing_static}",
        "",
        f"Initialized markers: {initialized_markers}",
        f"Missing markers: {missing_markers}",
        "",
        f"Missing moving frames count: {len(missing_moving)}",
        "",
        "Interpretation:",
        (
            "- Productive BA poses use the deterministic maximum-bottleneck "
            "paths from the configured reference marker."
        ),
        (
            "- First-hit BFS is retained only in diagnostics and never "
            "initializes bundle adjustment."
        ),
        "- Ground truth is not read by either initialization.",
        "",
    ]

    (
        out / "graph_connectivity_report.txt"
    ).write_text(
        "\n".join(report) + "\n"
    )

    print(f"[OK] wrote {out}")
    print(
        "[OK] initialized static cameras:",
        initialized_static,
    )
    print(
        "[OK] initialized markers:",
        len(initialized_markers),
        "/",
        len(all_markers),
    )


if __name__ == "__main__":
    main()
