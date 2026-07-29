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
    top_moving_per_marker: int | None,
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
        registered_by_marker: dict[int, list[dict]] = {}
        for row in moving_rows:
            if row["_frame"] in poses:
                registered_by_marker.setdefault(
                    int(row["_marker"]), []
                ).append(row)
        moving = core.moving_by_marker(
            moving_rows,
            set(poses),
            top_per_marker=top_moving_per_marker,
        )
        selection_rows: list[dict] = []
        for marker_id, registered in sorted(
            registered_by_marker.items()
        ):
            ranked = sorted(
                registered,
                key=lambda row: (
                    -float(row["_quality"]),
                    int(row["_frame"]),
                ),
            )
            selected_frames = {
                int(row["_frame"])
                for row in moving.get(marker_id, [])
            }
            for rank, row in enumerate(ranked, 1):
                selection_rows.append(
                    {
                        "marker_id": marker_id,
                        "frame_id": int(row["_frame"]),
                        "quality_rank": rank,
                        "selection_score": float(row["_quality"]),
                        "selected": (
                            int(row["_frame"]) in selected_frames
                        ),
                        "registered_observations_for_marker": len(
                            ranked
                        ),
                        "selected_observations_for_marker": len(
                            selected_frames
                        ),
                        "top_moving_per_marker": (
                            top_moving_per_marker
                        ),
                        "tie_breaker": (
                            "stable ascending moving-frame number"
                        ),
                    }
                )
        stage_root.mkdir(parents=True, exist_ok=True)
        core.write_csv(
            stage_root / "AP01_RELAY_SELECTION.csv",
            selection_rows,
        )
        (stage_root / "AP01_RELAY_SELECTION.json").write_text(
            json.dumps(selection_rows, indent=2) + "\n",
            encoding="utf-8",
        )
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
        parameters={
            "selection": "quality_ranked_per_marker",
            "top_moving_per_marker": top_moving_per_marker,
        },
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
        top_moving_per_marker=args.top_moving_per_marker,
    )


if __name__ == "__main__":
    main()
