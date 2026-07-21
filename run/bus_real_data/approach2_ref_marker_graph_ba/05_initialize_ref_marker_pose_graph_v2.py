#!/usr/bin/env python3

from __future__ import annotations

import argparse
import heapq
import itertools
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from ap02_common import (
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

from ap02_observation_quality import (
    is_success,
    observation_score,
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
    best: dict[
        tuple[str, int],
        dict[str, str],
    ] = {}

    for row in rows:
        if not is_success(row):
            continue

        try:
            marker_id = int(float(row["marker_id"]))
        except (KeyError, ValueError):
            continue

        score = observation_score(row)

        if score <= 0.0:
            continue

        key = (
            row["observer_id"],
            marker_id,
        )

        if (
            key not in best
            or score > observation_score(best[key])
        ):
            best[key] = row

    return list(best.values())


def build_graph(
    rows: list[dict[str, str]],
):
    adjacency = defaultdict(list)

    for row in rows:
        marker_id = int(float(row["marker_id"]))
        observer_id = row["observer_id"]
        score = observation_score(row)

        marker = marker_node(marker_id)
        observer = observer_node(observer_id)

        adjacency[marker].append(
            (observer, row, score)
        )

        adjacency[observer].append(
            (marker, row, score)
        )

    return adjacency


def widest_path_tree(
    adjacency,
    start: Node,
):
    """
    Rooted maximum-spanning tree.

    At each step, add the strongest observation edge connecting the
    already initialized Ref14 component to one previously unseen node.

    The resulting tree is acyclic and provides maximum-bottleneck
    paths from the reference marker.
    """

    counter = itertools.count()
    visited = {start}

    parent: dict[
        Node,
        tuple[Node, dict[str, str], float],
    ] = {}

    best: dict[
        Node,
        tuple[float, float, int],
    ] = {
        start: (
            float("inf"),
            float("inf"),
            0,
        )
    }

    heap = []

    def push_frontier(node: Node) -> None:
        for neighbor, row, edge_score in adjacency.get(
            node,
            [],
        ):
            if neighbor in visited:
                continue

            if not math.isfinite(edge_score) or edge_score <= 0.0:
                continue

            heapq.heappush(
                heap,
                (
                    -edge_score,
                    next(counter),
                    node,
                    neighbor,
                    row,
                ),
            )

    push_frontier(start)

    while heap:
        (
            negative_score,
            _,
            from_node,
            to_node,
            row,
        ) = heapq.heappop(heap)

        if to_node in visited:
            continue

        if from_node not in visited:
            raise RuntimeError(
                "Tree frontier contains an uninitialized parent: "
                f"{from_node}"
            )

        edge_score = -negative_score

        visited.add(to_node)

        parent[to_node] = (
            from_node,
            row,
            edge_score,
        )

        parent_bottleneck = best[from_node][0]
        parent_hops = best[from_node][2]

        best[to_node] = (
            min(parent_bottleneck, edge_score),
            edge_score,
            parent_hops + 1,
        )

        push_frontier(to_node)

    return parent, best


def initialize_from_tree(
    parent,
    best,
    ref_marker_id: int,
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

    for child, (
        parent_node,
        row,
        edge_score,
    ) in parent.items():
        children[parent_node].append(
            (
                child,
                row,
                edge_score,
            )
        )

    queue = deque([start])

    init_log = []
    used_edges = []

    while queue:
        current = queue.popleft()

        for child, row, edge_score in children.get(
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

            path_bottleneck = best[child][0]

            init_log.append(
                {
                    "initialized_type": child_kind,
                    "initialized_id": child_id,
                    "from_type": current_kind,
                    "from_id": current_id,
                    "observed_marker_id": int(
                        float(row["marker_id"])
                    ),
                    "observer_id": row["observer_id"],
                    "frame_id": row.get("frame_id", ""),
                    "edge_quality": edge_score,
                    "path_bottleneck": path_bottleneck,
                    "pnp_reprojection_rmse_px": (
                        pnp_reprojection_rmse(row)
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

    args = parser.parse_args()

    out = ensure_dir(
        Path(args.out_root) / args.mode
    )

    all_rows = read_csv(OBS_CSV)
    mode_rows = filter_mode(all_rows, args.mode)
    selected_rows = best_observations(mode_rows)

    adjacency = build_graph(selected_rows)

    start = marker_node(args.ref_marker_id)

    parent, best = widest_path_tree(
        adjacency,
        start,
    )

    (
        marker_poses,
        observer_poses,
        init_log,
        used_edges,
    ) = initialize_from_tree(
        parent,
        best,
        args.ref_marker_id,
    )

    static_pose_rows = []
    moving_pose_rows = []

    for observer_id, transform in sorted(
        observer_poses.items()
    ):
        if observer_id.startswith("cam_edge_"):
            static_pose_rows.append(
                pose_row(
                    "static_camera",
                    observer_id,
                    transform,
                    source=f"{args.mode}_widest_path",
                )
            )

        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(
                pose_row(
                    "moving_frame",
                    observer_id,
                    transform,
                    source=f"{args.mode}_widest_path",
                )
            )

    marker_pose_rows = [
        pose_row(
            "marker",
            str(marker_id),
            transform,
            source=f"{args.mode}_widest_path",
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
        "initialized_type",
        "initialized_id",
        "from_type",
        "from_id",
        "observed_marker_id",
        "observer_id",
        "frame_id",
        "edge_quality",
        "path_bottleneck",
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

    camera_bottlenecks = {}

    for camera in initialized_static:
        node = observer_node(camera)

        if node in best:
            camera_bottlenecks[camera] = best[node][0]

    report = [
        "AP02 maximum-bottleneck pose graph initialization",
        "=================================================",
        "",
        f"Mode: {args.mode}",
        f"Reference marker id: {args.ref_marker_id}",
        "",
        f"Raw observations in mode: {len(mode_rows)}",
        f"Quality-valid observer-marker edges: {len(selected_rows)}",
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
        f"Camera path bottlenecks: {camera_bottlenecks}",
        "",
        f"Initialized markers: {initialized_markers}",
        f"Missing markers: {missing_markers}",
        "",
        f"Missing moving frames count: {len(missing_moving)}",
        "",
        "Interpretation:",
        (
            "- Every entity is initialized through the path whose "
            "weakest observation has the highest possible quality."
        ),
        (
            "- This replaces first-hit BFS and avoids committing "
            "cam5 to the first weak bridge encountered."
        ),
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
        "[OK] camera path bottlenecks:",
        camera_bottlenecks,
    )
    print(
        "[OK] initialized markers:",
        len(initialized_markers),
        "/",
        len(all_markers),
    )


if __name__ == "__main__":
    main()
