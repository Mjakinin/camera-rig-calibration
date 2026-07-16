#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "results/bus_real_data/quality_check/full_approach_benchmark"
BASE_COLLECTOR = REPO / "run/bus_real_data/quality_check/07_collect_verified_benchmark_metrics.py"


def load_base():
    spec = importlib.util.spec_from_file_location("verified_metrics", BASE_COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load collector: {BASE_COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def collect_ap03(root: Path):
    metadata_source = root / "07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    pose_source = root / "07_final_results/AP03_GT_PAIRWISE_POSE_ERRORS.csv"
    metadata = base.read_json(metadata_source)
    pose_rows = [row for row in base.read_csv(pose_source) if row.get("status") == "OK"]
    translation = base.values(pose_rows, ["translation_error_cm"], 0.01)
    rotation = base.values(pose_rows, ["rotation_error_deg"])

    result = {}
    result.update(base.summarize(translation, "translation_error_m"))
    result.update(base.summarize(rotation, "rotation_error_deg"))
    result.update({
        "reprojection_rmse_px": "",
        "scale_estimate": metadata.get("scale_m_per_colmap_unit", ""),
        "scale_raw_median": metadata.get("raw_median_scale", ""),
        "scale_observations_total": metadata.get("num_scale_observations_total", ""),
        "scale_observations_used": metadata.get("num_scale_observations_used", ""),
        "scale_spread": metadata.get("used_std_scale", ""),
        "scale_relative_spread": metadata.get("used_rel_std_scale", ""),
        "scale_raw_mad": metadata.get("raw_mad_scale", ""),
        "registered_images": metadata.get("registered_images", ""),
        "registered_static_cameras": metadata.get("registered_static_cameras", ""),
        "registered_moving_frames": metadata.get("registered_moving_frames", ""),
        "pose_metric_source": base.relative(pose_source, root),
        "scale_metric_source": base.relative(metadata_source, root),
        "approach_metric_note": "AP03 pose errors use six GT pairwise static-camera extrinsics; GT is used only after estimation. Scale uses marker-size-only metadata.",
    })
    return result


def read_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect verified metrics and append AP03 snapshots discovered on disk.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.root / "verified_full_benchmark_metrics.csv"

    # First let the verified base collector refresh all runs represented in pipeline_run_summary.csv.
    base.collect_ap03 = collect_ap03
    original_argv = __import__("sys").argv
    try:
        __import__("sys").argv = [str(BASE_COLLECTOR), "--root", str(args.root), "--output", str(output)]
        base.main()
    finally:
        __import__("sys").argv = original_argv

    rows = read_rows(output)
    by_key = {(row.get("case_id", ""), row.get("approach", "")): row for row in rows}

    # Add or replace AP03 rows directly from existing snapshot directories, independent of run-log contents.
    for case_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        result_root = case_dir / "AP03/results"
        pose_file = result_root / "07_final_results/AP03_GT_PAIRWISE_POSE_ERRORS.csv"
        static_pose_file = result_root / "07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv"
        if not pose_file.is_file() or not static_pose_file.is_file():
            continue

        row = by_key.get((case_dir.name, "AP03"), {
            "case_id": case_dir.name,
            "approach": "AP03",
            "status": "success",
            "result_dir": str(result_root.relative_to(REPO)),
        })
        row.update(collect_ap03(result_root))
        by_key[(case_dir.name, "AP03")] = row

    merged = list(by_key.values())
    order = {"AP01": 1, "AP02": 2, "AP03": 3}
    merged.sort(key=lambda row: (row.get("case_id", ""), order.get(row.get("approach", ""), 99)))
    write_rows(output, merged)

    print(f"Wrote merged metrics: {output}")
    for row in merged:
        if row.get("approach") == "AP03":
            print(
                f"{row.get('case_id')} AP03: "
                f"T={row.get('translation_error_m_mean', '')} "
                f"R={row.get('rotation_error_deg_mean', '')} "
                f"scale={row.get('scale_estimate', '')}"
            )


if __name__ == "__main__":
    main()
