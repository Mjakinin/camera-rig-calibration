#!/usr/bin/env python3
"""Compatibility entry point for the staged AP02 implementation.

New rigcal runs invoke the importable stages individually. This wrapper keeps the
historical aggregate command usable without dynamic imports or simulated argv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from camera_rig_calibration.methods.ap02 import (
    build_graph,
    initialize_stage,
    optimize_stage,
    report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    parser.add_argument("--max-nfev-static", type=int, default=100)
    parser.add_argument("--max-nfev-moving", type=int, default=120)
    parser.add_argument(
        "--ba-robust-loss",
        choices=["soft_l1", "huber", "linear"],
        default="soft_l1",
    )
    parser.add_argument("--ba-robust-loss-scale-px", type=float, default=3.0)
    parser.add_argument("--cameras", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    cameras = tuple(
        value.strip() for value in args.cameras.split(",") if value.strip()
    )
    if not cameras:
        raise RuntimeError("--cameras must contain at least one camera ID")

    build_graph.run(
        observations_root=args.observations_root.resolve(),
        output_root=output,
        camera_ids=cameras,
        reference_marker_id=args.ref_marker_id,
    )
    static_initialization = initialize_stage.run(
        output_root=output,
        reference_marker_id=args.ref_marker_id,
        mode="static_only",
    )
    if static_initialization.status == "COMPLETED":
        optimize_stage.run(
            output_root=output,
            reference_marker_id=args.ref_marker_id,
            mode="static_only",
            maximum_function_evaluations=args.max_nfev_static,
            robust_loss=args.ba_robust_loss,
            robust_loss_scale_px=args.ba_robust_loss_scale_px,
        )
    initialize_stage.run(
        output_root=output,
        reference_marker_id=args.ref_marker_id,
        mode="with_moving",
    )
    optimize_stage.run(
        output_root=output,
        reference_marker_id=args.ref_marker_id,
        mode="with_moving",
        maximum_function_evaluations=args.max_nfev_moving,
        robust_loss=args.ba_robust_loss,
        robust_loss_scale_px=args.ba_robust_loss_scale_px,
    )
    report.run(
        output_root=output,
        camera_ids=cameras,
        reference_marker_id=args.ref_marker_id,
    )


if __name__ == "__main__":
    main()
