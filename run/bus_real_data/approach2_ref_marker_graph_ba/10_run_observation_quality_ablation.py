#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median

from ap02_common import (
    AP02_ROOT,
    DEFAULT_REF_MARKER_ID,
    ensure_dir,
    read_csv,
    write_csv,
)
from ap02_observation_quality import (
    is_success,
    observation_score,
    pnp_reprojection_rmse,
    safe_float,
)


DEFAULT_OBSERVATIONS = (
    AP02_ROOT
    / "02_aruco_observations"
    / "ap02_all_aruco_observations.csv"
)
DEFAULT_OUTPUT = AP02_ROOT / "10_observation_quality_ablation"

VARIANTS = (
    "baseline_success",
    "marker_area_only",
    "distance_only",
    "reprojection_only",
    "marker_area_distance",
    "marker_area_reprojection",
    "distance_reprojection",
    "all_hard_filters",
    "weighted_score",
)


def percentile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    if len(finite) == 1:
        return finite[0]

    position = (len(finite) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]

    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def marker_id(row: dict[str, str]) -> int | None:
    try:
        return int(float(row["marker_id"]))
    except (KeyError, TypeError, ValueError):
        return None


def quality_values(row: dict[str, str]) -> dict[str, float]:
    return {
        "area_px2": safe_float(row, "area_px2", 0.0),
        "distance_m": safe_float(row, "distance_m", float("inf")),
        "reprojection_rmse_px": pnp_reprojection_rmse(row),
        "weighted_score": observation_score(row),
    }


def acceptance_reasons(
    row: dict[str, str],
    *,
    min_area_px2: float,
    max_distance_m: float,
    max_reprojection_rmse_px: float,
    min_weighted_score: float,
) -> dict[str, bool]:
    values = quality_values(row)
    return {
        "success": is_success(row),
        "marker_area": values["area_px2"] >= min_area_px2,
        "distance": (
            values["distance_m"] > 0.0
            and values["distance_m"] <= max_distance_m
        ),
        "reprojection": (
            math.isfinite(values["reprojection_rmse_px"])
            and values["reprojection_rmse_px"]
            <= max_reprojection_rmse_px
        ),
        "weighted_score": (
            math.isfinite(values["weighted_score"])
            and values["weighted_score"] >= min_weighted_score
        ),
    }


def accepts(variant: str, reasons: dict[str, bool]) -> bool:
    if not reasons["success"]:
        return False

    required = {
        "baseline_success": (),
        "marker_area_only": ("marker_area",),
        "distance_only": ("distance",),
        "reprojection_only": ("reprojection",),
        "marker_area_distance": ("marker_area", "distance"),
        "marker_area_reprojection": ("marker_area", "reprojection"),
        "distance_reprojection": ("distance", "reprojection"),
        "all_hard_filters": (
            "marker_area",
            "distance",
            "reprojection",
        ),
        "weighted_score": ("weighted_score",),
    }

    if variant not in required:
        raise ValueError(f"Unknown quality variant: {variant}")

    return all(reasons[name] for name in required[variant])


def ref_component_metrics(
    rows: list[dict[str, str]],
    ref_marker_id: int,
) -> dict[str, int | bool]:
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    for row in rows:
        current_marker_id = marker_id(row)
        observer_id = str(row.get("observer_id", "")).strip()
        if current_marker_id is None or not observer_id:
            continue

        marker_node = ("marker", str(current_marker_id))
        observer_node = ("observer", observer_id)
        adjacency[marker_node].add(observer_node)
        adjacency[observer_node].add(marker_node)

    start = ("marker", str(ref_marker_id))
    if start not in adjacency:
        return {
            "reference_marker_present": False,
            "reference_component_markers": 0,
            "reference_component_observers": 0,
            "reference_component_edges": 0,
        }

    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    component_markers = {
        node for node in visited if node[0] == "marker"
    }
    component_observers = {
        node for node in visited if node[0] == "observer"
    }
    component_edges = sum(
        1
        for row in rows
        if marker_id(row) is not None
        and ("marker", str(marker_id(row))) in component_markers
        and ("observer", str(row.get("observer_id", "")))
        in component_observers
    )

    return {
        "reference_marker_present": True,
        "reference_component_markers": len(component_markers),
        "reference_component_observers": len(component_observers),
        "reference_component_edges": component_edges,
    }


def summarize_variant(
    variant: str,
    source_rows: list[dict[str, str]],
    accepted_rows: list[dict[str, str]],
    ref_marker_id: int,
) -> dict[str, object]:
    successful_rows = [row for row in source_rows if is_success(row)]
    areas = [safe_float(row, "area_px2") for row in accepted_rows]
    distances = [safe_float(row, "distance_m") for row in accepted_rows]
    reprojections = [pnp_reprojection_rmse(row) for row in accepted_rows]
    scores = [observation_score(row) for row in accepted_rows]

    observers = {
        str(row.get("observer_id", ""))
        for row in accepted_rows
        if str(row.get("observer_id", ""))
    }
    markers = {
        value
        for row in accepted_rows
        if (value := marker_id(row)) is not None
    }
    static_observers = {
        str(row.get("observer_id", ""))
        for row in accepted_rows
        if row.get("observer_type") == "static"
    }
    moving_observers = {
        str(row.get("observer_id", ""))
        for row in accepted_rows
        if row.get("observer_type") == "moving"
    }

    successful_count = len(successful_rows)
    accepted_count = len(accepted_rows)
    summary: dict[str, object] = {
        "variant": variant,
        "input_observations": len(source_rows),
        "successful_observations": successful_count,
        "accepted_observations": accepted_count,
        "rejected_successful_observations": successful_count - accepted_count,
        "retention_fraction_of_successful": (
            accepted_count / successful_count if successful_count else 0.0
        ),
        "unique_markers": len(markers),
        "unique_observers": len(observers),
        "unique_static_observers": len(static_observers),
        "unique_moving_observers": len(moving_observers),
        "area_px2_median": median(areas) if areas else float("nan"),
        "area_px2_p10": percentile(areas, 0.10),
        "distance_m_median": median(distances) if distances else float("nan"),
        "distance_m_p90": percentile(distances, 0.90),
        "reprojection_rmse_px_median": (
            median(reprojections) if reprojections else float("nan")
        ),
        "reprojection_rmse_px_p90": percentile(reprojections, 0.90),
        "weighted_score_median": median(scores) if scores else float("nan"),
    }
    summary.update(ref_component_metrics(accepted_rows, ref_marker_id))
    return summary


def annotated_row(
    row: dict[str, str],
    reasons: dict[str, bool],
) -> dict[str, object]:
    values = quality_values(row)
    result: dict[str, object] = dict(row)
    result.update(values)
    result.update({f"passes_{key}": value for key, value in reasons.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ablation over marker-observation quality filters and "
            "report retention plus reference-marker graph connectivity."
        )
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ref-marker-id", type=int, default=DEFAULT_REF_MARKER_ID)
    parser.add_argument("--min-area-px2", type=float, default=64.0)
    parser.add_argument("--max-distance-m", type=float, default=12.0)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=4.0)
    parser.add_argument("--min-weighted-score", type=float, default=0.01)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    args = parser.parse_args()

    rows = read_csv(args.observations)
    out_root = ensure_dir(args.out_root)
    summaries: list[dict[str, object]] = []
    rejection_counts: dict[str, Counter[str]] = {}

    for variant in args.variants:
        accepted_rows: list[dict[str, str]] = []
        annotated_rows: list[dict[str, object]] = []
        rejected = Counter()

        for row in rows:
            reasons = acceptance_reasons(
                row,
                min_area_px2=args.min_area_px2,
                max_distance_m=args.max_distance_m,
                max_reprojection_rmse_px=args.max_reprojection_rmse_px,
                min_weighted_score=args.min_weighted_score,
            )
            accepted = accepts(variant, reasons)
            enriched = annotated_row(row, reasons)
            enriched["accepted"] = accepted
            annotated_rows.append(enriched)

            if accepted:
                accepted_rows.append(row)
            else:
                for reason, passed in reasons.items():
                    if not passed:
                        rejected[reason] += 1

        variant_dir = ensure_dir(out_root / variant)
        accepted_fields = list(rows[0].keys()) if rows else []
        write_csv(
            variant_dir / "accepted_observations.csv",
            accepted_rows,
            accepted_fields,
        )

        annotation_fields = list(annotated_rows[0].keys()) if annotated_rows else []
        write_csv(
            variant_dir / "observation_decisions.csv",
            annotated_rows,
            annotation_fields,
        )

        summary = summarize_variant(
            variant,
            rows,
            accepted_rows,
            args.ref_marker_id,
        )
        summaries.append(summary)
        rejection_counts[variant] = rejected

        with (variant_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=True)
            handle.write("\n")

    summary_fields = list(summaries[0].keys()) if summaries else []
    write_csv(out_root / "quality_ablation_summary.csv", summaries, summary_fields)

    manifest = {
        "observations": str(args.observations),
        "ref_marker_id": args.ref_marker_id,
        "thresholds": {
            "min_area_px2": args.min_area_px2,
            "max_distance_m": args.max_distance_m,
            "max_reprojection_rmse_px": args.max_reprojection_rmse_px,
            "min_weighted_score": args.min_weighted_score,
        },
        "variants": list(args.variants),
        "rejection_counts": {
            variant: dict(counts)
            for variant, counts in rejection_counts.items()
        },
    }
    with (out_root / "quality_ablation_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Wrote observation quality ablation to: {out_root}")
    for summary in summaries:
        print(
            f"{summary['variant']}: "
            f"accepted={summary['accepted_observations']}, "
            f"retention={summary['retention_fraction_of_successful']:.3f}, "
            f"ref_component_markers={summary['reference_component_markers']}, "
            f"ref_component_observers={summary['reference_component_observers']}"
        )


if __name__ == "__main__":
    main()
