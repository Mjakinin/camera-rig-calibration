#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
AP02_DIR = REPO_ROOT / "run" / "bus_real_data" / "approach2_ref_marker_graph_ba"
sys.path.insert(0, str(AP02_DIR))

from ap02_observation_quality import (  # noqa: E402
    is_success,
    pnp_reprojection_rmse,
    safe_float,
)

DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "bus_real_data"
    / "02_ref_marker_graph_ba"
    / "02_aruco_observations"
    / "ap02_all_aruco_observations.csv"
)
DEFAULT_PROFILES = HERE / "quality_profiles.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "bus_real_data"
    / "quality_check"
    / "cross_approach_inputs"
)
APPROACHES = ("AP01", "AP02", "AP03")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def marker_id(row: dict[str, str]) -> int | None:
    try:
        return int(float(row["marker_id"]))
    except (KeyError, TypeError, ValueError):
        return None


def accepts(row: dict[str, str], profile: dict[str, float]) -> bool:
    if not is_success(row):
        return False

    area = safe_float(row, "area_px2", 0.0)
    distance = safe_float(row, "distance_m", float("inf"))
    reprojection = pnp_reprojection_rmse(row)

    return (
        area >= float(profile["min_area_px2"])
        and distance > 0.0
        and distance <= float(profile["max_distance_m"])
        and math.isfinite(reprojection)
        and reprojection <= float(profile["max_reprojection_rmse_px"])
    )


def component_metrics(
    rows: list[dict[str, str]],
    ref_marker_id: int,
) -> dict[str, int | bool]:
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        current_marker = marker_id(row)
        observer = str(row.get("observer_id", "")).strip()
        if current_marker is None or not observer:
            continue
        marker_node = ("marker", str(current_marker))
        observer_node = ("observer", observer)
        adjacency[marker_node].add(observer_node)
        adjacency[observer_node].add(marker_node)

    start = ("marker", str(ref_marker_id))
    if start not in adjacency:
        return {
            "reference_marker_present": False,
            "connected_markers": 0,
            "connected_observers": 0,
            "connected_edges": 0,
        }

    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    marker_nodes = {node for node in visited if node[0] == "marker"}
    observer_nodes = {node for node in visited if node[0] == "observer"}
    connected_edges = sum(
        1
        for row in rows
        if marker_id(row) is not None
        and ("marker", str(marker_id(row))) in marker_nodes
        and ("observer", str(row.get("observer_id", ""))) in observer_nodes
    )
    return {
        "reference_marker_present": True,
        "connected_markers": len(marker_nodes),
        "connected_observers": len(observer_nodes),
        "connected_edges": connected_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create identical quality-filtered ArUco observation inputs for "
            "AP01, AP02 and AP03, and report retention/connectivity."
        )
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ref-marker-id", type=int, default=14)
    parser.add_argument("--profile", action="append", dest="selected_profiles")
    args = parser.parse_args()

    rows = read_csv(args.observations)
    if not rows:
        raise RuntimeError(f"No observations found in {args.observations}")

    with args.profiles.open("r", encoding="utf-8") as handle:
        profiles: dict[str, dict[str, object]] = json.load(handle)

    selected = args.selected_profiles or list(profiles)
    unknown = sorted(set(selected) - set(profiles))
    if unknown:
        raise ValueError(f"Unknown profiles: {', '.join(unknown)}")

    successful = [row for row in rows if is_success(row)]
    summaries: list[dict[str, object]] = []

    for profile_name in selected:
        profile = profiles[profile_name]
        accepted = [row for row in rows if accepts(row, profile)]
        metrics = component_metrics(accepted, args.ref_marker_id)

        markers = {value for row in accepted if (value := marker_id(row)) is not None}
        observers = {
            str(row.get("observer_id", ""))
            for row in accepted
            if str(row.get("observer_id", ""))
        }

        for approach in APPROACHES:
            target = args.out_root / profile_name / approach / "aruco_observations.csv"
            write_csv(target, accepted, list(rows[0].keys()))

            summaries.append(
                {
                    "approach": approach,
                    "filter_profile": profile_name,
                    "description": profile.get("description", ""),
                    "input_observations": len(rows),
                    "successful_observations": len(successful),
                    "accepted_observations": len(accepted),
                    "retention_fraction": (
                        len(accepted) / len(successful) if successful else 0.0
                    ),
                    "unique_markers": len(markers),
                    "unique_observers": len(observers),
                    **metrics,
                    "min_area_px2": profile["min_area_px2"],
                    "max_distance_m": profile["max_distance_m"],
                    "max_reprojection_rmse_px": profile[
                        "max_reprojection_rmse_px"
                    ],
                    "observations_path": str(target),
                }
            )

    fields = list(summaries[0].keys()) if summaries else []
    write_csv(args.out_root / "cross_approach_quality_input_summary.csv", summaries, fields)

    manifest = {
        "source_observations": str(args.observations),
        "profiles_file": str(args.profiles),
        "profiles": selected,
        "approaches": list(APPROACHES),
        "ref_marker_id": args.ref_marker_id,
        "note": (
            "Each approach receives byte-equivalent filtered observation rows. "
            "Approach-specific pipeline adapters should consume the generated CSV."
        ),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Wrote cross-approach quality inputs to: {args.out_root}")
    for profile_name in selected:
        row = next(item for item in summaries if item["filter_profile"] == profile_name)
        print(
            f"{profile_name}: accepted={row['accepted_observations']}, "
            f"retention={row['retention_fraction']:.3f}, "
            f"connected_markers={row['connected_markers']}, "
            f"connected_observers={row['connected_observers']}"
        )


if __name__ == "__main__":
    main()
