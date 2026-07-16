#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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


base.collect_ap03 = collect_ap03

if __name__ == "__main__":
    base.main()
