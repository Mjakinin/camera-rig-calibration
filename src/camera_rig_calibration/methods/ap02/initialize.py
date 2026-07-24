"""Deterministic AP02 pose-graph initialization."""

from __future__ import annotations

import argparse
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

    def rank(row: dict[str, str]) -> tuple[float, float, str]:
        rmse = pnp_reprojection_rmse(row)
        try:
            area = float(row.get("area_px2", 0.0) or 0.0)
        except (TypeError, ValueError):
            area = 0.0
        stable = "|".join(
            (
                str(row.get("observer_id", "")),
                str(row.get("frame_id", "")),
                str(row.get("image_path", "")),
            )
        )
        return (rmse, -area, stable)

    for row in sorted(rows, key=rank):
        if not is_success(row):
            continue

        try:
            marker_id = int(float(row["marker_id"]))
        except (KeyError, ValueError):
            continue

        if not math.isfinite(pnp_reprojection_rmse(row)):
            continue

        key = (
            row["observer_id"],
            marker_id,
        )

        if key not in best or rank(row) < rank(best[key]):
            best[key] = row

    return [best[key] for key in sorted(best)]


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


def initialize_from_tree(
    parent,
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
                    "initialized_type": child_kind,
                    "initialized_id": child_id,
                    "from_type": current_kind,
                    "from_id": current_id,
                    "observed_marker_id": int(
                        float(row["marker_id"])
                    ),
                    "observer_id": row["observer_id"],
                    "frame_id": row.get("frame_id", ""),
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

    parent = deterministic_breadth_first_tree(
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
        args.ref_marker_id,
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
                    source=f"{args.mode}_deterministic_bfs",
                )
            )

        elif observer_id.startswith("moving_frame_"):
            moving_pose_rows.append(
                pose_row(
                    "moving_frame",
                    observer_id,
                    transform,
                    source=f"{args.mode}_deterministic_bfs",
                )
            )

    marker_pose_rows = [
        pose_row(
            "marker",
            str(marker_id),
            transform,
            source=f"{args.mode}_deterministic_bfs",
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

    report = [
        "AP02 deterministic pose graph initialization",
        "============================================",
        "",
        f"Mode: {args.mode}",
        f"Reference marker id: {args.ref_marker_id}",
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
            "- Connectivity is traversed deterministically from the "
            "configured reference marker."
        ),
        (
            "- PnP RMSE and marker area only select a reproducible "
            "representative for duplicate observer-marker edges."
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
        "[OK] initialized markers:",
        len(initialized_markers),
        "/",
        len(all_markers),
    )


if __name__ == "__main__":
    main()
