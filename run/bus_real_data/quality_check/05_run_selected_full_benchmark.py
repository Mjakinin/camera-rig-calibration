#!/usr/bin/env python3
"""Corrected entry point for the selected full AP01/AP02/AP03 benchmark.

This wrapper keeps the implementation in 03_run_selected_full_benchmark.py,
but aligns the selected case IDs with 02_generate_quality_benchmark_cases.py
and forces a complete AP03 run. A full AP03 run is required because the current
AP03 shell pipeline deletes generated outputs unless it performs preparation and
COLMAP reconstruction again.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE / "03_run_selected_full_benchmark.py"

spec = importlib.util.spec_from_file_location("quality_full_benchmark", IMPLEMENTATION)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load benchmark implementation: {IMPLEMENTATION}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.APPROACHES["AP03"]["command"] = [
    "bash",
    "run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh",
]

module.DEFAULT_CASES = [
    "baseline",
    "area_750",
    "area_1000",
    "area_2000",
    "distance_4m",
    "distance_5m",
    "distance_6m",
    "reprojection_0.2px",
    "reprojection_0.3px",
    "reprojection_0.5px",
    "subsample_0.75_rep0",
    "subsample_0.5_rep0",
    "subsample_0.25_rep0",
    "corner_noise_1.0px",
    "corner_noise_2.0px",
    "corner_noise_5.0px",
    "outliers_0.01",
    "outliers_0.05",
    "outliers_0.1",
    "outliers_0.2",
    "reference_marker_10",
    "reference_marker_14",
    "reference_marker_20",
]

if __name__ == "__main__":
    module.main()
