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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def summarize(values, prefix):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": "",
            f"{prefix}_median": "",
            f"{prefix}_max": "",
        }
    return {
        f"{prefix}_count": len(clean),
        f"{prefix}_mean": mean(clean),
        f"{prefix}_median": median(clean),
        f"{prefix}_max": max(clean),
    }


def values(rows, aliases, scale=1.0):
    output = []
    for row in rows:
        for alias in aliases:
            number = finite_float(row.get(alias))
            if number is not None:
                output.append(number * scale)
                break
    return output


def relative(path, root):
    return str(path.relative_to(root)) if path and path.is_file() else ""


def collect_ap01(root: Path):
    # Static direct-relay quality: one final aggregate estimate, not candidate rows.
    static_source = root / "05_direct_static_cam3_cam1_multimarker/05_multimarker_aggregate_estimates.csv"
    static_rows = read_csv(static_source)
    static_translation = values(static_rows, [
        "translation_error_m", "translation_error", "translation_l2_m",
        "position_error_m", "trans_error_m",
        "translation_error_cm", "position_error_cm",
    ])
    # Convert explicit centimetre fields only when metre fields were absent.
    if not static_translation:
        static_translation = values(static_rows, ["translation_error_cm", "position_error_cm"], 0.01)
    static_rotation = values(static_rows, [
        "rotation_error_deg", "rotation_geodesic_deg", "angular_error_deg",
        "rot_error_deg",
    ])

    # Moving trajectory quality: canonical per-frame Sim(3)-aligned trajectory files.
    trajectory_source = root / "04_moving_camera_colmap_trajectory/sim3_eval_vs_gt/sim3_aligned_trajectory_errors.csv"
    trajectory_rows = read_csv(trajectory_source)
    trajectory_translation = values(trajectory_rows, [
        "translation_error_m", "position_error_m", "error_m",
        "euclidean_error_m", "l2_error_m",
    ])

    rotation_source = root / "04_moving_camera_colmap_trajectory/sim3_eval_vs_gt/rotation_eval_orientation_aligned/rotation_errors_by_frame.csv"
    rotation_rows = read_csv(rotation_source)
    trajectory_rotation = values(rotation_rows, [
        "rotation_error_deg", "angular_error_deg", "geodesic_error_deg",
    ])

    result = {}
    result.update(summarize(static_translation, "static_translation_error_m"))
    result.update(summarize(static_rotation, "static_rotation_error_deg"))
    result.update(summarize(trajectory_translation, "trajectory_translation_error_m"))
    result.update(summarize(trajectory_rotation, "trajectory_rotation_error_deg"))
    # Common columns use static extrinsic quality for cross-approach camera calibration.
    result.update(summarize(static_translation, "translation_error_m"))
    result.update(summarize(static_rotation, "rotation_error_deg"))
    result.update({
        "reprojection_rmse_px": "",
        "scale_estimate": "",
        "scale_observations_total": "",
        "scale_observations_used": "",
        "scale_spread": "",
        "pose_metric_source": relative(static_source, root),
        "trajectory_translation_source": relative(trajectory_source, root),
        "trajectory_rotation_source": relative(rotation_source, root),
        "approach_metric_note": "AP01 static aggregate and moving trajectory are reported separately; candidate and sorted duplicate tables are excluded.",
    })
    return result


def collect_ap02(root: Path):
    pose_source = root / "08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv"
    rows = read_csv(pose_source)
    # The verified file stores translation in centimetres.
    translation = values(rows, ["translation_error_cm"], 0.01)
    rotation = values(rows, ["rotation_error_deg"])

    ba_source = root / "07_graph_ba/with_moving/ba_summary.txt"
    final_mean = final_median = final_max = None
    observations_used = None
    optimizer_success = ""
    if ba_source.is_file():
        text = ba_source.read_text(encoding="utf-8", errors="replace")
        block = re.search(
            r"Final reprojection error \[px\]:\s*\n"
            r"- mean:\s*([0-9.eE+-]+)\s*\n"
            r"- median:\s*([0-9.eE+-]+)\s*\n"
            r"- max:\s*([0-9.eE+-]+)",
            text,
        )
        if block:
            final_mean = finite_float(block.group(1))
            final_median = finite_float(block.group(2))
            final_max = finite_float(block.group(3))
        match = re.search(r"Marker observations used:\s*(\d+)", text)
        if match:
            observations_used = int(match.group(1))
        match = re.search(r"- success:\s*(True|False)", text)
        if match:
            optimizer_success = match.group(1).lower()

    result = {}
    result.update(summarize(translation, "translation_error_m"))
    result.update(summarize(rotation, "rotation_error_deg"))
    result.update({
        "reprojection_rmse_px": final_mean if final_mean is not None else "",
        "reprojection_median_px": final_median if final_median is not None else "",
        "reprojection_max_px": final_max if final_max is not None else "",
        "optimized_observations_used": observations_used if observations_used is not None else "",
        "optimizer_success": optimizer_success,
        "scale_estimate": "",
        "scale_observations_total": "",
        "scale_observations_used": "",
        "scale_spread": "",
        "pose_metric_source": relative(pose_source, root),
        "reprojection_metric_source": relative(ba_source, root),
        "approach_metric_note": "AP02 translation_error_cm is converted to metres; final BA reprojection statistics are parsed from the verified final block.",
    })
    return result


def collect_ap03(root: Path):
    metadata_source = root / "07_final_results/AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json"
    metadata = read_json(metadata_source)
    result = {}
    result.update(summarize([], "translation_error_m"))
    result.update(summarize([], "rotation_error_deg"))
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
        "pose_metric_source": "",
        "scale_metric_source": relative(metadata_source, root),
        "approach_metric_note": "AP03 scale metrics use the verified marker-size-only metadata keys. Pose error remains empty until a canonical GT evaluation is generated.",
    })
    return result


def empty_metrics(note):
    result = {}
    result.update(summarize([], "translation_error_m"))
    result.update(summarize([], "rotation_error_deg"))
    result.update({
        "reprojection_rmse_px": "",
        "scale_estimate": "",
        "scale_observations_total": "",
        "scale_observations_used": "",
        "scale_spread": "",
        "pose_metric_source": "",
        "approach_metric_note": note,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect verified, approach-specific benchmark metrics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = read_csv(args.root / "pipeline_run_summary.csv")
    if not runs:
        raise SystemExit(f"Missing run summary: {args.root / 'pipeline_run_summary.csv'}")

    collectors = {"AP01": collect_ap01, "AP02": collect_ap02, "AP03": collect_ap03}
    output_rows = []
    for run in runs:
        row = dict(run)
        result_root = Path(run.get("result_dir", ""))
        if not result_root.is_absolute():
            result_root = REPO / result_root
        collector = collectors.get(run.get("approach", ""))
        if run.get("status") == "success" and collector and result_root.is_dir():
            row.update(collector(result_root))
        else:
            row.update(empty_metrics("Run failed, approach unknown, or result directory missing."))
        output_rows.append(row)

    output = args.output or args.root / "verified_full_benchmark_metrics.csv"
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
            f"reproj={row.get('reprojection_rmse_px', '')} "
            f"scale={row.get('scale_estimate', '')}"
        )


if __name__ == "__main__":
    main()
