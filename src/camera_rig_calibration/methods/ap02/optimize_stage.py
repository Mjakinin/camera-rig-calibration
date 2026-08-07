from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def run(
    *,
    output_root: Path,
    reference_marker_id: int,
    mode: str,
    maximum_function_evaluations: int,
    robust_loss: str,
    robust_loss_scale_px: float,
    reprojection_model: str = "legacy_pinhole_v1",
    moving_frame_selection_policy: str = (
        "legacy_smart_at_ba_boundary_v1"
    ),
    reference_marker_maximum_frames: int | None = None,
    top_per_marker: int | None = 8,
    top_per_marker_pair: int | None = 4,
    maximum_total_frames: int | None = None,
    historical_reproduction: bool = False,
    log_path: Path | None = None,
) -> StageResult:
    stage_root = output_root / "07_graph_ba" / mode
    observations = (
        output_root
        / "02_aruco_observations"
        / "ap02_all_aruco_observations.csv"
    )
    initialization = output_root / "05_graph_initialization"

    def action() -> dict[str, Path]:
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.methods.ap02.optimize",
            "--mode",
            mode,
            "--ref-marker-id",
            str(reference_marker_id),
            "--max-nfev",
            str(maximum_function_evaluations),
            "--ap02-root",
            str(output_root),
            "--observations",
            str(observations),
            "--initialization-root",
            str(initialization),
            "--robust-loss",
            robust_loss,
            "--robust-loss-scale-px",
            str(robust_loss_scale_px),
            "--reprojection-model",
            reprojection_model,
            "--moving-frame-selection-policy",
            moving_frame_selection_policy,
        ]
        if historical_reproduction:
            command.append("--historical-reproduction")
        for option, value in (
            (
                "--reference-marker-maximum-frames",
                reference_marker_maximum_frames,
            ),
            ("--top-per-marker", top_per_marker),
            ("--top-per-marker-pair", top_per_marker_pair),
            ("--maximum-total-frames", maximum_total_frames),
        ):
            if value is not None:
                command.extend([option, str(value)])
        if log_path is None:
            subprocess.run(command, check=True)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command,
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        return {
            "static_poses": stage_root
            / "optimized_static_camera_poses_ref_marker.csv",
            "moving_poses": stage_root
            / "optimized_moving_frame_poses_ref_marker.csv",
            "marker_poses": stage_root
            / "optimized_marker_poses_ref_marker.csv",
            "optimizer_report": stage_root / "optimizer_report.json",
            "optimization_summary": stage_root
            / "ap02_optimization_summary.json",
            "optimization_history": stage_root
            / "ap02_optimization_history.csv",
        }

    return run_stage(
        f"ap02.optimize.{mode}",
        stage_root,
        action,
        inputs={
            "observations": observations,
            "initialization": initialization / mode,
        },
        parameters={
            "reference_marker_id": reference_marker_id,
            "mode": mode,
            "maximum_function_evaluations": maximum_function_evaluations,
            "robust_loss": robust_loss,
            "robust_loss_scale_px": robust_loss_scale_px,
            "reprojection_model": reprojection_model,
            "moving_frame_selection_policy": moving_frame_selection_policy,
            "reference_marker_maximum_frames": (
                reference_marker_maximum_frames
            ),
            "top_per_marker": top_per_marker,
            "top_per_marker_pair": top_per_marker_pair,
            "maximum_total_frames": maximum_total_frames,
            "observation_input": (
                "quality-ranked, graph-preserving AP02 frame selection"
            ),
            "historical_reproduction": historical_reproduction,
        },
        failure_is_diagnostic=mode == "static_only",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref-marker-id", type=int, required=True)
    parser.add_argument(
        "--mode", choices=["static_only", "with_moving"], required=True
    )
    parser.add_argument("--max-nfev", type=int, required=True)
    parser.add_argument(
        "--robust-loss",
        choices=["soft_l1", "huber", "linear"],
        required=True,
    )
    parser.add_argument("--robust-loss-scale-px", type=float, required=True)
    parser.add_argument(
        "--reprojection-model", default="legacy_pinhole_v1"
    )
    parser.add_argument(
        "--moving-frame-selection-policy",
        default="legacy_smart_at_ba_boundary_v1",
    )
    parser.add_argument("--reference-marker-maximum-frames", type=int)
    parser.add_argument("--top-per-marker", type=int, default=8)
    parser.add_argument("--top-per-marker-pair", type=int, default=4)
    parser.add_argument("--maximum-total-frames", type=int)
    parser.add_argument("--historical-reproduction", action="store_true")
    args = parser.parse_args()
    run(
        output_root=args.out.resolve(),
        reference_marker_id=args.ref_marker_id,
        mode=args.mode,
        maximum_function_evaluations=args.max_nfev,
        robust_loss=args.robust_loss,
        robust_loss_scale_px=args.robust_loss_scale_px,
        reprojection_model=args.reprojection_model,
        moving_frame_selection_policy=args.moving_frame_selection_policy,
        reference_marker_maximum_frames=(
            args.reference_marker_maximum_frames
        ),
        top_per_marker=args.top_per_marker,
        top_per_marker_pair=args.top_per_marker_pair,
        maximum_total_frames=args.maximum_total_frames,
        historical_reproduction=args.historical_reproduction,
    )


if __name__ == "__main__":
    main()
