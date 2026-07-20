#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_INPUT = REPO / "results/bus_real_data/02_ref_marker_graph_ba/02_aruco_observations/ap02_all_aruco_observations.csv"
DEFAULT_OUT = REPO / "results/bus_real_data/quality_check/benchmark_cases"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def successful(row: dict[str, str]) -> bool:
    value = str(row.get("pnp_success", row.get("success", ""))).strip().lower()
    return value in {"1", "true", "yes", "ok", "success"}


def finite_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def image_size(row: dict[str, str]) -> tuple[float, float] | None:
    aliases = [
        ("image_width", "image_height"),
        ("width", "height"),
        ("img_width", "img_height"),
    ]
    for wk, hk in aliases:
        width = finite_float(row, wk)
        height = finite_float(row, hk)
        if width and height and width > 0 and height > 0:
            return width, height
    return None


def border_distance_px(row: dict[str, str]) -> float | None:
    size = image_size(row)
    if size is None:
        return None
    width, height = size
    distances: list[float] = []
    for corner in range(4):
        u = finite_float(row, f"corner{corner}_u")
        v = finite_float(row, f"corner{corner}_v")
        if u is None or v is None:
            return None
        distances.extend([u, v, width - 1.0 - u, height - 1.0 - v])
    return min(distances)


def observer_key(row: dict[str, str]) -> str:
    for key in ("observer_id", "frame_id", "image_name", "image", "frame"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def save_case(out_root: Path, case_id: str, rows: list[dict[str, str]], fields: list[str], metadata: dict) -> None:
    case_root = out_root / case_id
    for approach in ("AP01", "AP02", "AP03"):
        write_csv(case_root / approach / "aruco_observations.csv", rows, fields)
    (case_root / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate border-distance and minimum-markers-per-frame benchmark cases.")
    parser.add_argument("--observations", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--border", type=float, action="append", default=[5.0, 10.0, 20.0, 40.0])
    parser.add_argument("--min-markers", type=int, action="append", default=[2, 3, 4])
    args = parser.parse_args()

    rows = read_csv(args.observations)
    if not rows:
        raise SystemExit(f"No observations found: {args.observations}")
    fields = list(rows[0].keys())
    base = [dict(row) for row in rows if successful(row)]

    summary: list[dict[str, object]] = []

    for threshold in sorted(set(args.border)):
        selected = [row for row in base if (distance := border_distance_px(row)) is not None and distance >= threshold]
        case_id = f"border_{threshold:g}px"
        metadata = {
            "case_id": case_id,
            "test_type": "min_border_distance_px",
            "test_level": threshold,
            "input_observations": len(base),
            "accepted_observations": len(selected),
            "retention_fraction": len(selected) / len(base) if base else 0.0,
        }
        save_case(args.out_root, case_id, selected, fields, metadata)
        summary.append(metadata)

    counts = Counter(observer_key(row) for row in base if observer_key(row))
    for minimum in sorted(set(args.min_markers)):
        valid_observers = {observer for observer, count in counts.items() if count >= minimum}
        selected = [row for row in base if observer_key(row) in valid_observers]
        case_id = f"min_markers_{minimum}"
        metadata = {
            "case_id": case_id,
            "test_type": "min_markers_per_observer",
            "test_level": minimum,
            "input_observations": len(base),
            "accepted_observations": len(selected),
            "accepted_observers": len(valid_observers),
            "retention_fraction": len(selected) / len(base) if base else 0.0,
        }
        save_case(args.out_root, case_id, selected, fields, metadata)
        summary.append(metadata)

    summary_path = args.out_root / "additional_quality_case_summary.csv"
    summary_fields: list[str] = []
    for row in summary:
        for key in row:
            if key not in summary_fields:
                summary_fields.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)

    print(f"Generated {len(summary)} cases")
    print(f"Summary: {summary_path}")
    for row in summary:
        print(f"{row['case_id']}: {row['accepted_observations']}/{row['input_observations']} observations")


if __name__ == "__main__":
    main()
