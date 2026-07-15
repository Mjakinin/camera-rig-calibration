#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
AP02_DIR = HERE.parent / "approach2_ref_marker_graph_ba"
sys.path.insert(0, str(AP02_DIR))

from ap02_observation_quality import is_success, pnp_reprojection_rmse, safe_float  # noqa: E402

DEFAULT_INPUT = Path("results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv")
DEFAULT_MATRIX = HERE / "benchmark_matrix.json"
DEFAULT_OUTPUT = Path("results/bus_real_data/quality_check/benchmark_cases")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def marker_id(row: dict[str, str]) -> int | None:
    try:
        return int(float(row.get("marker_id", "")))
    except (TypeError, ValueError):
        return None


def filter_rows(rows: list[dict[str, str]], *, min_area: float = 0.0, max_distance: float = math.inf, max_reprojection: float = math.inf) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if not is_success(row):
            continue
        area = safe_float(row, "area_px2", 0.0)
        distance = safe_float(row, "distance_m", math.inf)
        reprojection = pnp_reprojection_rmse(row)
        if area < min_area:
            continue
        if not (0.0 < distance <= max_distance):
            continue
        if not (math.isfinite(reprojection) and reprojection <= max_reprojection):
            continue
        selected.append(dict(row))
    return selected


def add_corner_noise(rows: list[dict[str, str]], sigma: float, rng: random.Random) -> list[dict[str, str]]:
    output = []
    for source in rows:
        row = dict(source)
        if sigma > 0.0:
            for corner in range(4):
                for axis in ("u", "v"):
                    key = f"corner{corner}_{axis}"
                    try:
                        row[key] = str(float(row[key]) + rng.gauss(0.0, sigma))
                    except (KeyError, TypeError, ValueError):
                        pass
        output.append(row)
    return output


def inject_outliers(rows: list[dict[str, str]], fraction: float, rng: random.Random) -> list[dict[str, str]]:
    output = [dict(row) for row in rows]
    count = min(len(output), round(len(output) * fraction))
    for index in rng.sample(range(len(output)), count) if count else []:
        row = output[index]
        for corner in range(4):
            for axis in ("u", "v"):
                key = f"corner{corner}_{axis}"
                try:
                    row[key] = str(float(row[key]) + rng.choice((-1.0, 1.0)) * rng.uniform(10.0, 50.0))
                except (KeyError, TypeError, ValueError):
                    pass
    return output


def graph_metrics(rows: list[dict[str, str]], ref_marker: int) -> dict[str, object]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_count = 0
    for row in rows:
        mid = marker_id(row)
        observer = str(row.get("observer_id", "")).strip()
        if mid is None or not observer:
            continue
        marker = f"m:{mid}"
        observer_node = f"o:{observer}"
        adjacency[marker].add(observer_node)
        adjacency[observer_node].add(marker)
        edge_count += 1

    components = 0
    visited: set[str] = set()
    for node in adjacency:
        if node in visited:
            continue
        components += 1
        queue = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    start = f"m:{ref_marker}"
    ref_nodes: set[str] = set()
    distances: dict[str, int] = {}
    if start in adjacency:
        queue = deque([start])
        ref_nodes.add(start)
        distances[start] = 0
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in ref_nodes:
                    ref_nodes.add(neighbor)
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

    degrees = [len(neighbors) for neighbors in adjacency.values()]
    return {
        "components": components,
        "nodes": len(adjacency),
        "edges": edge_count,
        "unique_markers": sum(node.startswith("m:") for node in adjacency),
        "unique_observers": sum(node.startswith("o:") for node in adjacency),
        "reference_marker_present": start in adjacency,
        "reference_component_markers": sum(node.startswith("m:") for node in ref_nodes),
        "reference_component_observers": sum(node.startswith("o:") for node in ref_nodes),
        "mean_degree": sum(degrees) / len(degrees) if degrees else 0.0,
        "max_shortest_path_from_reference": max(distances.values()) if distances else -1,
    }


def save_case(root: Path, case_id: str, rows: list[dict[str, str]], source_count: int, test_type: str, test_level: object, ref_marker: int, summary: list[dict[str, object]]) -> None:
    case_dir = root / case_id
    fields = list(rows[0].keys()) if rows else []
    for approach in ("AP01", "AP02", "AP03"):
        write_csv(case_dir / approach / "aruco_observations.csv", rows, fields)
    metrics = graph_metrics(rows, ref_marker)
    summary.append({
        "case_id": case_id,
        "test_type": test_type,
        "test_level": test_level,
        "reference_marker_id": ref_marker,
        "input_observations": source_count,
        "accepted_observations": len(rows),
        "retention_fraction": len(rows) / source_count if source_count else 0.0,
        **metrics,
        "case_root": str(case_dir),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_csv(args.observations)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rng = random.Random(int(matrix.get("seed", 42)))
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    default_ref = int(matrix.get("reference_marker_candidates", [14])[0])

    save_case(out, "baseline", filter_rows(rows), len(rows), "baseline", 1.0, default_ref, summary)

    for value in matrix["threshold_sweeps"]["min_area_px2"]:
        save_case(out, f"area_{value}", filter_rows(rows, min_area=float(value)), len(rows), "min_area_px2", value, default_ref, summary)
    for value in matrix["threshold_sweeps"]["max_distance_m"]:
        save_case(out, f"distance_{value}m", filter_rows(rows, max_distance=float(value)), len(rows), "max_distance_m", value, default_ref, summary)
    for value in matrix["threshold_sweeps"]["max_reprojection_rmse_px"]:
        save_case(out, f"reprojection_{value}px", filter_rows(rows, max_reprojection=float(value)), len(rows), "max_reprojection_rmse_px", value, default_ref, summary)

    successful = filter_rows(rows)
    for fraction in matrix["subsampling"]["fractions"]:
        for repetition in range(int(matrix["subsampling"]["repetitions"])):
            count = round(len(successful) * float(fraction))
            selected = rng.sample(successful, count) if count < len(successful) else list(successful)
            save_case(out, f"subsample_{fraction}_rep{repetition}", selected, len(rows), "subsample_fraction", fraction, default_ref, summary)

    marker_ids = sorted({mid for row in successful if (mid := marker_id(row)) is not None})
    if matrix.get("leave_one_marker_out", False):
        for removed in marker_ids:
            selected = [row for row in successful if marker_id(row) != removed]
            save_case(out, f"leave_marker_{removed}_out", selected, len(rows), "leave_one_marker_out", removed, default_ref, summary)

    for ref_marker in matrix.get("reference_marker_candidates", [default_ref]):
        save_case(out, f"reference_marker_{ref_marker}", successful, len(rows), "reference_marker", ref_marker, int(ref_marker), summary)

    for sigma in matrix.get("corner_noise_px", []):
        selected = add_corner_noise(successful, float(sigma), rng)
        save_case(out, f"corner_noise_{sigma}px", selected, len(rows), "corner_noise_px", sigma, default_ref, summary)

    for fraction in matrix.get("outlier_fractions", []):
        selected = inject_outliers(successful, float(fraction), rng)
        save_case(out, f"outliers_{fraction}", selected, len(rows), "outlier_fraction", fraction, default_ref, summary)

    fields = list(summary[0].keys()) if summary else []
    write_csv(out / "benchmark_case_summary.csv", summary, fields)
    (out / "benchmark_manifest.json").write_text(json.dumps({"source": str(args.observations), "matrix": matrix, "cases": len(summary)}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(summary)} benchmark cases in {out}")


if __name__ == "__main__":
    main()
