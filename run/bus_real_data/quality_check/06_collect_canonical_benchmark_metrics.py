#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "results/bus_real_data/quality_check/full_approach_benchmark"


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def summarize(values: list[float], prefix: str) -> dict[str, object]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": "",
            f"{prefix}_median": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": mean(values),
        f"{prefix}_median": median(values),
        f"{prefix}_max": max(values),
    }


def values_from_rows(rows, aliases):
    output = []
    for row in rows:
        for key in aliases:
            value = finite_float(row.get(key))
            if value is not None:
                output.append(value)
                break
    return output


def first_existing(root: Path, relative_candidates: list[str]) -> Path | None:
    for relative in relative_candidates:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def recursive_find_exact(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def nested_numbers(data, aliases: set[str]) -> list[float]:
    found = []
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if normalized in aliases:
                number = finite_float(value)
                if number is not None:
                    found.append(number)
            found.extend(nested_numbers(value, aliases))
    elif isinstance(data, list):
        for value in data:
            found.extend(nested_numbers(value, aliases))
    return found


def canonical_pairwise_metrics(result_root: Path, approach: str):
    candidates = [
        "99_FINAL_RESULTS_FOR_REPORT/data/primary/BASELINE_FINAL_PAIRWISE_DETAIL.csv",
        "99_FINAL_RESULTS_FOR_REPORT/data/primary/BASELINE_FINAL_PAIRWISE_RESULT.csv",
        "99_FINAL_RESULTS_FOR_REPORT/data/primary/AP01_PAIRWISE_RESULT.csv",
        "99_FINAL_RESULTS_FOR_REPORT/data/primary/AP02_PAIRWISE_RESULT.csv",
        "99_FINAL_RESULTS_FOR_REPORT/data/primary/AP03_PAIRWISE_RESULT.csv",
        "07_final_extrinsics_cam3_reference/FINAL_PAIRWISE_DETAIL.csv",
        "07_final_extrinsics_cam3_reference/PAIRWISE_RESULT.csv",
    ]
    source = first_existing(result_root, candidates)
    if source is None:
        exact_names = [
            f"{approach}_PAIRWISE_RESULT.csv",
            f"{approach}_PAIRWISE_DETAIL.csv",
            "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
        ]
        for name in exact_names:
            source = recursive_find_exact(result_root, name)
            if source:
                break
    rows = read_csv(source) if source else []
    translation = values_from_rows(rows, [
        "translation_error_m", "translation_error", "trans_error_m",
        "position_error_m", "translation_l2_m",
    ])
    rotation = values_from_rows(rows, [
        "rotation_error_deg", "rotation_geodesic_deg", "rot_error_deg",
        "angular_error_deg", "rotation_error",
    ])
    result = {}
    result.update(summarize(translation, "translation_error_m"))
    result.update(summarize(rotation, "rotation_error_deg"))
    result["pose_metric_source"] = str(source.relative_to(result_root)) if source else ""
    return result


def collect_ap01(root: Path):
    result = canonical_pairwise_metrics(root, "AP01")
    result.update({
        "reprojection_rmse_px": "",
        "scale_estimate": "",
        "scale_error_percent": "",
        "scale_inliers": "",
        "scale_spread": "",
        "approach_metric_note": "Canonical final pairwise static-camera evaluation only; trajectory and candidate tables excluded.",
    })
    return result


def collect_ap02(root: Path):
    source = first_existing(root, [
        "08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv",
    ]) or recursive_find_exact(root, "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv")
    rows = read_csv(source) if source else []
    translation = values_from_rows(rows, [
        "translation_error_m", "position_error_m", "trans_error_m",
        "translation_l2_m", "center_error_m",
    ])
    rotation = values_from_rows(rows, [
        "rotation_error_deg", "rotation_geodesic_deg", "rot_error_deg",
        "angular_error_deg",
    ])
    result = {}
    result.update(summarize(translation, "translation_error_m"))
    result.update(summarize(rotation, "rotation_error_deg"))
    result["pose_metric_source"] = str(source.relative_to(root)) if source else ""

    ba_source = first_existing(root, [
        "07_graph_ba/with_moving/ba_summary.txt",
        "07_graph_ba/static_only/ba_summary.txt",
    ])
    reprojection = None
    if ba_source:
        text = ba_source.read_text(encoding="utf-8", errors="replace")
        patterns = [
            r"(?:final|optimized).*?(?:reprojection|rmse)[^0-9+-]*([0-9]+(?:\.[0-9]+)?)",
            r"(?:reprojection|rmse)[^0-9+-]*([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                reprojection = finite_float(match.group(1))
                if reprojection is not None:
                    break
    result.update({
        "reprojection_rmse_px": reprojection if reprojection is not None else "",
        "reprojection_metric_source": str(ba_source.relative_to(root)) if ba_source else "",
        "scale_estimate": "",
        "scale_error_percent": "",
        "scale_inliers": "",
        "scale_spread": "",
        "approach_metric_note": "Only AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv and one BA summary are used.",
    })
    return result


def collect_ap03(root: Path):
    result = canonical_pairwise_metrics(root, "AP03")
    metadata_path = first_existing(root, [
        "07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json",
    ]) or recursive_find_exact(root, "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json")
    metadata = read_json(metadata_path)
    scale_values = nested_numbers(metadata, {
        "scale", "scale_factor", "estimated_scale", "global_scale",
        "metric_scale", "final_scale",
    }) if metadata is not None else []
    scale_error_values = nested_numbers(metadata, {
        "scale_error_percent", "scale_error_pct", "relative_scale_error_percent",
    }) if metadata is not None else []
    inlier_values = nested_numbers(metadata, {
        "num_inliers", "inlier_count", "scale_inliers", "n_inliers",
    }) if metadata is not None else []
    spread_values = nested_numbers(metadata, {
        "scale_std", "scale_mad", "scale_spread", "scale_sigma",
    }) if metadata is not None else []
    result.update({
        "reprojection_rmse_px": "",
        "scale_estimate": scale_values[0] if scale_values else "",
        "scale_error_percent": scale_error_values[0] if scale_error_values else "",
        "scale_inliers": int(inlier_values[0]) if inlier_values else "",
        "scale_spread": spread_values[0] if spread_values else "",
        "scale_metric_source": str(metadata_path.relative_to(root)) if metadata_path else "",
        "approach_metric_note": "Only canonical AP03 pose evaluation and marker-size-scale metadata are used.",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect canonical, non-duplicated AP01/AP02/AP03 benchmark metrics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_summary = args.root / "pipeline_run_summary.csv"
    runs = read_csv(run_summary)
    if not runs:
        raise SystemExit(f"Missing or empty run summary: {run_summary}")
    collectors = {"AP01": collect_ap01, "AP02": collect_ap02, "AP03": collect_ap03}
    output_rows = []
    for run in runs:
        approach = run.get("approach", "")
        result_dir = Path(run.get("result_dir", ""))
        if not result_dir.is_absolute():
            result_dir = REPO / result_dir
        base = dict(run)
        collector = collectors.get(approach)
        if collector and result_dir.is_dir() and run.get("status") == "success":
            base.update(collector(result_dir))
        else:
            base.update({
                "translation_error_m_count": 0,
                "translation_error_m_mean": "",
                "translation_error_m_median": "",
                "translation_error_m_max": "",
                "rotation_error_deg_count": 0,
                "rotation_error_deg_mean": "",
                "rotation_error_deg_median": "",
                "rotation_error_deg_max": "",
                "reprojection_rmse_px": "",
                "scale_estimate": "",
                "scale_error_percent": "",
                "scale_inliers": "",
                "scale_spread": "",
                "pose_metric_source": "",
                "approach_metric_note": "No canonical metrics collected because the run failed or the result directory is missing.",
            })
        output_rows.append(base)
    output = args.output or (args.root / "canonical_full_benchmark_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in output_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote: {output}")
    for row in output_rows:
        print(
            f"{row.get('case_id')} {row.get('approach')}: "
            f"T={row.get('translation_error_m_count', 0)} "
            f"R={row.get('rotation_error_deg_count', 0)} "
            f"source={row.get('pose_metric_source', '')}"
        )


if __name__ == "__main__":
    main()
