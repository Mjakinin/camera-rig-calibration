#!/usr/bin/env python3
"""Compatibility entry point for the staged combined AP03 implementation.

Single and multi scaling share one COLMAP reconstruction and one scale
configuration. Multi-scale is the primary result; single-scale is diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig_calibration.methods.ap03 import (
    estimate_scale,
    inspect_stage,
    prepare_colmap,
    reconstruct_stage,
    report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--moving-camera-id", required=True)
    parser.add_argument(
        "--matcher", choices=["exhaustive", "sequential"], default="exhaustive"
    )
    parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--single-marker-id", type=int, required=True)
    parser.add_argument("--multi-marker-ids", required=True)
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--reprojection-threshold-px", type=float, default=5.0)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--minimum-inliers", type=int, default=4)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--colmap-executable", default="colmap")
    parser.add_argument("--max-image-size", type=int, default=2400)
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    parser.add_argument("--reuse-colmap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    dataset = args.dataset.resolve()
    observations = args.observations_root.resolve()
    output = args.out.resolve()
    cameras = tuple(
        value.strip() for value in args.cameras.split(",") if value.strip()
    )
    multi_markers = tuple(
        int(value.strip())
        for value in args.multi_marker_ids.split(",")
        if value.strip()
    )
    if not cameras:
        raise RuntimeError("--cameras must contain at least one camera ID")
    if not multi_markers:
        raise RuntimeError("--multi-marker-ids must contain at least one marker")

    prepare_colmap.run(
        dataset=dataset,
        output_root=output,
        camera_ids=cameras,
        moving_camera_id=args.moving_camera_id,
    )
    reconstruct_stage.run(
        dataset=dataset,
        output_root=output,
        camera_ids=cameras,
        moving_camera_id=args.moving_camera_id,
        matcher=args.matcher,
        use_gpu=bool(args.use_gpu),
        maximum_image_size=args.max_image_size,
        maximum_features=args.max_features,
        sequential_overlap=args.sequential_overlap,
        loop_detection=bool(args.loop_detection),
        mapper_minimum_matches=args.mapper_min_matches,
        colmap_executable=args.colmap_executable,
        reuse=args.reuse_colmap,
    )
    inspect_stage.run(output_root=output, camera_ids=cameras)
    estimate_scale.run(
        repository_root=repository,
        observations_root=observations,
        output_root=output,
        camera_ids=cameras,
        mode="single",
        marker_ids=(args.single_marker_id,),
        marker_length_m=args.marker_length_m,
        reprojection_threshold_px=args.reprojection_threshold_px,
        ransac_iterations=args.ransac_iterations,
        minimum_inliers=args.minimum_inliers,
        dictionary=args.dictionary,
    )
    estimate_scale.run(
        repository_root=repository,
        observations_root=observations,
        output_root=output,
        camera_ids=cameras,
        mode="multi",
        marker_ids=multi_markers,
        marker_length_m=args.marker_length_m,
        reprojection_threshold_px=args.reprojection_threshold_px,
        ransac_iterations=args.ransac_iterations,
        minimum_inliers=args.minimum_inliers,
        dictionary=args.dictionary,
    )
    report.run(output_root=output)


if __name__ == "__main__":
    main()
