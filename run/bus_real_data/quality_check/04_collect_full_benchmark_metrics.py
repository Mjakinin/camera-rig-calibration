#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

DEFAULT_ROOT = Path("results/bus_real_data/quality_check/full_approach_benchmark")

ALIASES = {
    "translation_error_m": (
        "translation_error_m", "translation_error", "trans_error_m",
        "position_error_m", "translation_l2_m", "t_error_m",
    ),
    "rotation_error_deg": (
        "rotation_error_deg", "rotation_error", "rot_error_deg",
        "angular_error_deg", "orientation_error_deg", "r_error_deg",
    ),
    "reprojection_rmse_px": (
        "reprojection_rmse_px", "reprojection_rmse", "reproj_rmse_px",
        "rmse_px", "mean_reprojection_error_px",
    ),
    "scale_error_percent": (
        "scale_error_percent", "scale_error_pct", "scale_percent_error",
    ),
    "scale": ("scale", "scale_factor", "estimated_scale", "metric_scale"),
}


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def flatten_json(value: object, prefix: str = "") -> dict[str, object]:
    output: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(flatten_json(child, f"{prefix}[{index}]"))
    else:
        output[prefix.lower()] = value
    return output


def collect_candidates(result_root: Path) -> dict[str, list[tuple[float, str]]]:
    candidates: dict[str, list[tuple[float, str]]] = {name: [] for name in ALIASES}
    for path in result_root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    lowered = {str(key).lower(): value for key, value in row.items()}
                    for metric, aliases in ALIASES.items():
                        for alias in aliases:
                            if alias in lowered:
                                value = finite_float(lowered[alias])
                                if value is not None:
                                    candidates[metric].append((value, str(path)))
                                break
        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    for path in result_root.rglob("*.json"):
        try:
            flattened = flatten_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for metric, aliases in ALIASES.items():
            for key, raw in flattened.items():
                leaf = key.rsplit(".", 1)[-1]
                if leaf in aliases:
                    value = finite_float(raw)
                    if value is not None:
                        candidates[metric].append((value, str(path)))
    return candidates


def summarize(values: list[tuple[float, str]]) -> dict[str, object]:
    if not values:
        return {"count": 0, "mean": "", "median": "", "max": "", "source": ""}
    numbers = [item[0] for item in values]
    sources = sorted({item[1] for item in values})
    return {
        "count": len(numbers),
        "mean": mean(numbers),
        "median": median(numbers),
        "max": max(numbers),
        "source": ";".join(sources[:5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AP01/AP02/AP03 metrics from benchmark snapshots.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    run_summary = args.root / "pipeline_run_summary.csv"
    if not run_summary.is_file():
        raise FileNotFoundError(f"Missing pipeline run summary: {run_summary}")

    with run_summary.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))

    output_rows: list[dict[str, object]] = []
    for run in runs:
        result_root = Path(run["result_dir"])
        candidates = collect_candidates(result_root) if result_root.exists() else {name: [] for name in ALIASES}
        row: dict[str, object] = dict(run)
        for metric in ALIASES:
            stats = summarize(candidates[metric])
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        output_rows.append(row)

    output = args.root / "full_benchmark_metrics.csv"
    if output_rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)
    print(f"Wrote: {output}")
    print("Review metric source columns: the collector records every file used for each aggregate.")


if __name__ == "__main__":
    main()
