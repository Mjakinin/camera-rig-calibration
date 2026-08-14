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
    initialization_algorithm: str = "maximum_frontier_v1",
    edge_weight_policy: str = "geometric_observation_quality_v1",
    log_path: Path | None = None,
) -> StageResult:
    stage_root = output_root / "05_graph_initialization" / mode
    observations = (
        output_root
        / "02_aruco_observations"
        / "ap02_all_aruco_observations.csv"
    )

    def action() -> dict[str, Path]:
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.methods.ap02.initialize",
            "--mode",
            mode,
            "--ref-marker-id",
            str(reference_marker_id),
            "--out-root",
            str(output_root / "05_graph_initialization"),
            "--observations",
            str(observations),
            "--initialization-algorithm",
            initialization_algorithm,
            "--edge-weight-policy",
            edge_weight_policy,
        ]
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
            / "initial_static_camera_poses_ref_marker.csv",
            "marker_poses": stage_root
            / "initial_marker_poses_ref_marker.csv",
            "moving_poses": stage_root
            / "initial_moving_frame_poses_ref_marker.csv",
        }

    return run_stage(
        f"ap02.initialize.{mode}",
        stage_root,
        action,
        inputs={"observations": observations},
        parameters={
            "reference_marker_id": reference_marker_id,
            "mode": mode,
            "initialization_algorithm": initialization_algorithm,
            "edge_weight_policy": edge_weight_policy,
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
    parser.add_argument(
        "--initialization-algorithm",
        default="maximum_frontier_v1",
    )
    parser.add_argument(
        "--edge-weight-policy",
        default="geometric_observation_quality_v1",
    )
    args = parser.parse_args()
    run(
        output_root=args.out.resolve(),
        reference_marker_id=args.ref_marker_id,
        mode=args.mode,
        initialization_algorithm=args.initialization_algorithm,
        edge_weight_policy=args.edge_weight_policy,
    )


if __name__ == "__main__":
    main()
