#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median


ROOT = Path(
    "results/bus_real_data/ablation/world/lighting"
)

VARIANTS = [
    "ceiling_dark_extreme",
    "ceiling_low",
    "ceiling_normal",
    "ceiling_bright",
]

STATIC_CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]

REF_MARKER = 14


def number(row: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def successful(row: dict[str, str]) -> bool:
    return row.get("pnp_success", "").strip().lower() in {
        "true",
        "1",
        "yes",
    }


def quality(row: dict[str, str]) -> float:
    area = max(0.0, number(row, "area_px2", 0.0))
    distance = number(row, "distance_m", 99.0)

    if distance <= 0:
        distance = 99.0

    return area / (distance * distance + 1e-9)


def best_observations(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    best: dict[tuple[str, int], dict[str, str]] = {}

    for row in rows:
        if not successful(row):
            continue

        try:
            marker_id = int(float(row["marker_id"]))
        except (KeyError, ValueError):
            continue

        key = (row["observer_id"], marker_id)

        if key not in best or quality(row) > quality(best[key]):
            best[key] = row

    return list(best.values())


def marker_node(marker_id: int) -> str:
    return f"M:{marker_id}"


def observer_node(observer_id: str) -> str:
    return f"O:{observer_id}"


def build_graph(
    rows: list[dict[str, str]],
):
    adjacency = defaultdict(list)

    for row in best_observations(rows):
        marker_id = int(float(row["marker_id"]))
        observer_id = row["observer_id"]

        marker = marker_node(marker_id)
        observer = observer_node(observer_id)

        edge = {
            "marker_id": marker_id,
            "observer_id": observer_id,
            "quality": quality(row),
            "area_px2": number(row, "area_px2", float("nan")),
            "distance_m": number(row, "distance_m", float("nan")),
            "frame_id": row.get("frame_id", ""),
        }

        adjacency[marker].append((observer, edge))
        adjacency[observer].append((marker, edge))

    return adjacency


def bfs_path(adjacency, start: str, target: str):
    queue = deque([start])
    parent = {start: None}
    parent_edge = {}

    while queue:
        current = queue.popleft()

        if current == target:
            break

        for neighbor, edge in adjacency.get(current, []):
            if neighbor in parent:
                continue

            parent[neighbor] = current
            parent_edge[neighbor] = edge
            queue.append(neighbor)

    if target not in parent:
        return None

    nodes = []
    edges = []
    current = target

    while current is not None:
        nodes.append(current)

        if current in parent_edge:
            edges.append(parent_edge[current])

        current = parent[current]

    nodes.reverse()
    edges.reverse()

    return nodes, edges


for variant in VARIANTS:
    csv_path = (
        ROOT
        / variant
        / "aruco_observations"
        / "shared_all_aruco_observations.csv"
    )

    if not csv_path.exists():
        raise SystemExit(f"[ERROR] Missing {csv_path}")

    with csv_path.open(newline="", errors="replace") as file:
        rows = list(csv.DictReader(file))

    valid = [row for row in rows if successful(row)]

    ref_rows = [
        row
        for row in valid
        if int(float(row["marker_id"])) == REF_MARKER
    ]

    cam5_rows = [
        row
        for row in valid
        if row.get("observer_id") == "cam_edge_5"
    ]

    cam5_markers = sorted({
        int(float(row["marker_id"]))
        for row in cam5_rows
    })

    graph = build_graph(valid)

    print()
    print("=" * 110)
    print(variant)
    print("=" * 110)
    print(f"valid observations: {len(valid)}")
    print(f"Ref14 observations: {len(ref_rows)}")
    print(
        "Ref14 observers:    "
        + str(sorted({
            row["observer_id"]
            for row in ref_rows
        }))
    )
    print(f"cam5 observations:   {len(cam5_rows)}")
    print(f"cam5 marker IDs:     {cam5_markers}")

    if cam5_rows:
        areas = [
            number(row, "area_px2", 0.0)
            for row in cam5_rows
        ]
        distances = [
            number(row, "distance_m", 99.0)
            for row in cam5_rows
        ]

        print(f"cam5 median area:    {median(areas):.2f} px²")
        print(f"cam5 median distance:{median(distances):.3f} m")

    for camera in STATIC_CAMERAS:
        result = bfs_path(
            graph,
            marker_node(REF_MARKER),
            observer_node(camera),
        )

        if result is None:
            print(f"{camera:12s}: NOT CONNECTED TO REF14")
            continue

        nodes, edges = result
        bottleneck = min(
            (edge["quality"] for edge in edges),
            default=float("nan"),
        )

        print(f"{camera:12s}: {' -> '.join(nodes)}")
        print(f"{'':14s} bottleneck quality={bottleneck:.6f}")

        for edge in edges:
            print(
                f"{'':16s}"
                f"observer={edge['observer_id']:<24s} "
                f"marker={edge['marker_id']:<2d} "
                f"frame={edge['frame_id']:<8s} "
                f"area={edge['area_px2']:9.2f} "
                f"dist={edge['distance_m']:7.3f} "
                f"q={edge['quality']:.6f}"
            )
