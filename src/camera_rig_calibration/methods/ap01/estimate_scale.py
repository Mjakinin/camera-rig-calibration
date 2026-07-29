from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, parser, prepared_observations


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
    scale_top_per_marker: int | None,
) -> StageResult:
    stage_root = output_root / "02_metric_scale"

    class Arguments:
        pass

    arguments = Arguments()
    arguments.dataset = dataset
    arguments.observations_root = observations_root
    arguments.out = output_root
    arguments.root_camera = root_camera
    arguments.moving_camera_id = moving_camera_id

    def action() -> dict[str, Path | float | int]:
        _, moving_rows, colmap_poses = prepared_observations(arguments)
        scale, statistics, pairs = core.robust_scale(
            moving_rows,
            colmap_poses,
            maximum_observations_per_marker=scale_top_per_marker,
        )
        stage_root.mkdir(parents=True, exist_ok=True)
        scale_file = stage_root / "metric_scale.txt"
        scale_file.write_text(f"{scale:.12g}\n", encoding="utf-8")
        diagnostics = stage_root / "SCALE_DIAGNOSTICS.json"
        diagnostics.write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        pairs_file = stage_root / "scale_pairs.csv"
        core.write_csv(pairs_file, pairs)
        return {
            "metric_scale": scale_file,
            "diagnostics": diagnostics,
            "pairs": pairs_file,
            "used_observation_pairs": int(statistics["used_pairs"]),
        }

    return run_stage(
        "ap01.estimate_scale",
        stage_root,
        action,
        inputs={
            "observations": observations_root,
            "colmap": output_root / "01_moving_colmap",
        },
        parameters={
            "selection": "quality_ranked_per_marker_before_pairing",
            "scale_top_per_marker": scale_top_per_marker,
        },
    )


def main() -> None:
    args = parser(__doc__ or "Estimate AP01 metric scale").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
        scale_top_per_marker=args.scale_top_per_marker,
    )


if __name__ == "__main__":
    main()
