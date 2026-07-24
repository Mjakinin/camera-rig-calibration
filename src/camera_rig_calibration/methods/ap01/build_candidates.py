from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, encode_candidate, parser, prepared_observations


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
) -> StageResult:
    stage_root = output_root / "03_candidates"

    class Arguments:
        pass

    arguments = Arguments()
    arguments.dataset = dataset
    arguments.observations_root = observations_root
    arguments.out = output_root
    arguments.root_camera = root_camera
    arguments.moving_camera_id = moving_camera_id

    def action() -> dict[str, Path | int]:
        static_rows, moving_rows, poses = prepared_observations(arguments)
        scale = float(
            (output_root / "02_metric_scale/metric_scale.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        static = core.best_static_by_camera_marker(static_rows)
        moving = core.moving_by_marker(moving_rows, set(poses))
        records: list[dict] = []
        for target in camera_ids:
            if target == root_camera:
                continue
            records.extend(
                encode_candidate(item)
                for item in core.direct_candidates(root_camera, target, static)
            )
            records.extend(
                encode_candidate(item)
                for item in core.relay_candidates(
                    root_camera,
                    target,
                    static,
                    moving,
                    poses,
                    scale,
                )
            )
        stage_root.mkdir(parents=True, exist_ok=True)
        path = stage_root / "transform_candidates.json"
        path.write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        return {"candidates": path, "candidate_count": len(records)}

    return run_stage(
        "ap01.build_candidates",
        stage_root,
        action,
        inputs={
            "observations": observations_root,
            "colmap": output_root / "01_moving_colmap",
            "scale": output_root / "02_metric_scale/metric_scale.txt",
        },
        parameters={"uses_all_quality_accepted_observations": True},
    )


def main() -> None:
    args = parser(__doc__ or "Build AP01 transform candidates").parse_args()
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=cameras(args),
        root_camera=args.root_camera,
        moving_camera_id=args.moving_camera_id,
    )


if __name__ == "__main__":
    main()
