from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, decode_candidate, parser


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
    moving_camera_id: str,
) -> StageResult:
    stage_root = output_root / "03_static_extrinsics"

    def action() -> dict[str, Path | int]:
        source = output_root / "03_candidates/transform_candidates.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for encoded in records:
            item = decode_candidate(encoded)
            grouped[str(item["target_camera"])][str(item["mode"])].append(
                item
            )

        poses = {root_camera: np.eye(4, dtype=np.float64)}
        methods = {root_camera: "gauge_identity"}
        diagnostics: dict[str, dict] = {}
        flattened: list[dict] = []
        for target in camera_ids:
            if target == root_camera:
                continue
            direct = grouped[target]["direct"]
            relay = grouped[target]["relay"]
            direct_pose = direct_stats = None
            relay_pose = relay_stats = None
            if direct:
                direct_pose, direct_stats = core.aggregate_candidates(
                    direct, translation_floor=0.12, rotation_floor=4.0
                )
            if relay:
                relay_pose, relay_stats = core.aggregate_candidates(
                    relay, translation_floor=0.30, rotation_floor=7.0
                )
            if len(direct) >= 2:
                selected, source_name = direct_pose, "direct_multimarker"
            elif relay_pose is not None:
                selected, source_name = relay_pose, "moving_colmap_relay"
            elif direct_pose is not None:
                selected, source_name = direct_pose, "direct_single_marker"
            else:
                selected, source_name = None, "unavailable"
            if selected is not None:
                poses[target] = selected
                methods[target] = source_name
            diagnostics[target] = {
                "selected_method": source_name,
                "direct": direct_stats,
                "relay": relay_stats,
                "relay_candidates": len(relay),
            }
            flattened.extend(core.serializable_candidate(item) for item in direct)
            flattened.extend(core.serializable_candidate(item) for item in relay)

        stage_root.mkdir(parents=True, exist_ok=True)
        pose_fields = [
            "entity_type",
            "entity_id",
            "source",
            "x_m",
            "y_m",
            "z_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "rvec_x",
            "rvec_y",
            "rvec_z",
        ]
        pose_rows = [
            core.pose_row(camera, pose, methods[camera])
            for camera, pose in sorted(poses.items())
        ]
        pose_file = (
            stage_root / "AP01_STATIC_CAMERA_POSES_ROOT_REFERENCE.csv"
        )
        core.write_csv(pose_file, pose_rows, pose_fields)
        if root_camera == "cam_edge_3":
            core.write_csv(
                stage_root / "AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv",
                pose_rows,
                pose_fields,
            )
        pairwise = stage_root / "AP01_PAIRWISE_DISTANCES.csv"
        core.CAMERAS = list(camera_ids)
        core.write_csv(
            pairwise,
            core.pairwise_rows(poses),
            ["camera_a", "camera_b", "distance_m"],
        )
        core.write_csv(
            stage_root / "AP01_TRANSFORM_CANDIDATES.csv", flattened
        )
        solution = stage_root / "solution_summary.json"
        solution.write_text(
            json.dumps(
                {
                    "root_camera": root_camera,
                    "camera_methods": methods,
                    "per_target_diagnostics": diagnostics,
                    "available_static_cameras": sorted(poses),
                    "missing_static_cameras": sorted(
                        set(camera_ids) - set(poses)
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "poses": pose_file,
            "pairwise": pairwise,
            "solution_summary": solution,
            "solved_cameras": len(poses),
        }

    return run_stage(
        "ap01.solve_extrinsics",
        stage_root,
        action,
        inputs={
            "candidates": (
                output_root / "03_candidates/transform_candidates.json"
            )
        },
        parameters={"root_camera": root_camera},
    )


def main() -> None:
    args = parser(__doc__ or "Solve AP01 static extrinsics").parse_args()
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
